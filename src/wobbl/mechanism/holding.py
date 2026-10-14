"""Whether a mechanism's motors can hold it and move it: the torque each driven joint needs, at its worst pose and
on fast moves, against what its motor gives. Works for any fixed base model, serial or with loops.

Holding a pose takes the torque that balances gravity, MuJoCo's bias force with nothing moving, less any spring in
the joint. In a mechanism with loops only the independent joints are driven, so the dependent joints' share comes
back through the loops by virtual work. The worst pose for each joint is searched for, and the search is run
twice from different seeds as a check. A hand moment sum over every body the joint carries checks the torque.
"""
from __future__ import annotations

import math

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .findings import Finding, Setup, amount
from .install import require as _mj
from .motors import stronger
from .poses import _eq_rows, describe_pose, sample, set_pose, space, worst

SEARCH = 300             # random poses before each refinement
MOVES = 8                # point to point moves at the speed limit


def quiet(model):
    """The model with contacts, joint limits and loop forces left out, for statics and inverse dynamics."""
    mujoco = _mj()
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    return model


def needed(sp, d) -> np.ndarray:
    """Torque each free coordinate's drive needs to hold ``d``'s pose still (gravity less springs), with the loops'
    share carried back to the free joints by virtual work."""
    mujoco = _mj()

    m = sp.model
    d.qvel[:] = 0
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    f = np.zeros(m.nv)
    mujoco.mj_rne(m, d, 0, f)
    mujoco.mj_passive(m, d)
    f -= d.qfrc_passive
    s = [j["dof"] for j in sp.free]
    if not sp.dep:
        return f[s]
    J, _ = _eq_rows(m, d)
    dd = [j["dof"] for j in sp.dep]
    G = -np.linalg.pinv(J[:, dd]) @ J[:, s]
    return f[s] + G.T @ f[dd]


def hand_moment(model, d, j: dict) -> float:
    """The torque a joint needs against gravity alone, by summing the moment of every body it carries about its
    axis (or their weight along it, for a slide), independent of MuJoCo's dynamics."""
    g = model.opt.gravity
    b0 = j["body"]
    inside = [b for b in range(1, model.nbody) if _carries(model, b0, b)]
    axis = d.xaxis[j["id"]]
    anchor = d.xanchor[j["id"]]
    tot = 0.0
    for b in inside:
        w = model.body_mass[b] * g
        tot += float(axis @ (np.cross(d.xipos[b] - anchor, w) if j["kind"] == "hinge" else w))
    return -tot


def _carries(model, top: int, b: int) -> bool:
    while b > 0:
        if b == top:
            return True
        b = int(model.body_parentid[b])
    return False


def holding(su: Setup) -> Finding:
    mujoco = _mj()

    model = quiet(su.ld.compile())
    sp = space(model, su.info)
    if not sp.n:
        return Finding("torque", "info", "The mechanism has no joint a motor could drive.")
    names = [j["name"] for j in sp.free]
    driven = [j for j in sp.free if j["name"] in su.motors] or list(sp.free)
    driven += [j for j in sp.free if j not in driven][:max(0, 12 - len(driven))]
    su.log("solve", f"searching {len(names)} joints for the poses that take the most torque")
    worst_of = {}
    for j in driven:
        k = names.index(j["name"])
        best, x, _ = worst(sp, lambda d, k=k: abs(needed(sp, d)[k]), n=su.n(SEARCH), seed=1)
        d = mujoco.MjData(model)
        set_pose(sp, d, x)
        worst_of[j["name"]] = {"j": j, "need": float(needed(sp, d)[k]), "x": x}
    judged = {n: w for n, w in worst_of.items() if n in su.motors and math.isfinite(su.motors[n].stall_torque)}
    for n, w in judged.items():
        w["limit"] = su.motors[n].stall_torque
        w["usage"] = abs(w["need"]) / max(w["limit"], 1e-12)
    hard = max(judged, key=lambda n: judged[n]["usage"]) if judged else max(worst_of,
                                                                            key=lambda n: abs(worst_of[n]["need"]))
    W = worst_of[hard]
    checks = [_search_check(sp, names.index(hard), W, su.n(SEARCH)), *_hand_check(sp, model, W)]
    dyn = _moves(su, sp, model, names) if not sp.dep else {}
    for n, w in judged.items():
        w["moving"] = dyn.get(n, {}).get("usage", 0.0)
    over = sorted((n for n in judged if judged[n]["usage"] > 1), key=lambda n: -judged[n]["usage"])
    stated = not su.cache.get("plain_speed")      # a speed Wobbl assumed is shown, but does not set the verdict
    fast = sorted((n for n in judged if stated and judged[n]["moving"] > 1 and n not in over),
                  key=lambda n: -judged[n]["moving"])
    near = [n for n in judged if 0.8 < judged[n]["usage"] <= 1]
    load = f" with {fmt(su.payload[0] * 1000)} g at the {su.payload[1]}" if su.payload else ""
    g = float(np.linalg.norm(model.opt.gravity))
    light = max(abs(w["need"]) for w in worst_of.values()) < 1e-6 * max(su.info.mass * max(g, 1.0) *
                                                                         float(model.stat.extent), 1e-9)
    if light:
        why = "the model has no gravity" if g == 0 else "every joint turns about a vertical axis or slides level"
        moving = max(judged, key=lambda n: judged[n]["moving"]) if judged and dyn else None
        verdict = "info" if not moving else ("fails" if judged[moving]["moving"] > 1 else
                                              "marginal" if judged[moving]["moving"] > 0.8 else "ok")
        sentence = f"Holding it still takes no torque, because {why}."
        if moving:
            sentence += (f" Moving fast, the {moving} needs {100 * judged[moving]['moving']:.0f}% of what its drive "
                         "gives at that speed.")
    elif not judged:
        verdict = "info"
        sentence = (f"The model gives no motor limits, so here is what its drives must supply{load}: the "
                    f"{hard} needs up to {amount(W['j'], abs(W['need']))} to hold its worst pose.")
    elif over:
        verdict = "fails"
        w = judged[over[0]]
        sentence = (f"The {over[0]} cannot hold the mechanism{load}: at its worst pose it needs "
                    f"{amount(w['j'], abs(w['need']))} against the {amount(w['j'], w['limit'])} its drive gives.")
        if len(over) > 1:
            sentence += f" The {', '.join(over[1:3])} {'are' if len(over) > 2 else 'is'} short as well."
    elif near or fast:
        verdict = "marginal"
        if near:
            w = judged[near[0]]
            sentence = (f"Every joint holds the mechanism{load}, but the {near[0]} has little to spare, needing "
                        f"{amount(w['j'], abs(w['need']))} of its {amount(w['j'], w['limit'])} at its worst pose.")
        else:
            w = judged[fast[0]]
            sentence = (f"Every joint holds the mechanism{load}, but moving at full speed the {fast[0]} needs "
                        f"{100 * w['moving']:.0f}% of what its drive gives at that speed.")
    else:
        verdict = "ok"
        w = judged[hard]
        sentence = (f"Every driven joint holds the mechanism{load} anywhere in its range, the busiest, the {hard}, "
                    f"needing {amount(w['j'], abs(w['need']))} of its {amount(w['j'], w['limit'])}.")
    unpowered = [n for n, w in worst_of.items() if n not in su.motors and abs(w["need"]) > 1e-6] if judged else []
    if unpowered:
        sentence += (f" The {', '.join(unpowered[:3])} {'have' if len(unpowered) > 1 else 'has'} no motor, so it "
                     "swings to hang wherever gravity leaves it.")
    fixes = _fixes(su, sp, model, judged, over + near) if verdict in ("fails", "marginal") else []
    values = {"joints": {n: {"needed": w["need"], "limit": w.get("limit"), "usage": w.get("usage"),
                             "moving_usage": w.get("moving"), "worst_pose": describe_pose(sp, w["x"]),
                             "unit": "N m" if w["j"]["kind"] == "hinge" else "N"} for n, w in worst_of.items()},
              "hardest": hard, "worst_usage": max((w["usage"] for w in judged.values()), default=None),
              "_poses": {f"worst for {n}": {j["name"]: float(v) for j, v in zip(sp.free, w["x"])}
                         for n, w in list(worst_of.items())[:3]}}
    metrics = [("hardest joint", hard), ("needs", amount(W["j"], abs(W["need"])))]
    if hard in judged:
        metrics += [("drive gives", amount(W["j"], judged[hard]["limit"])),
                    ("use", f"{100 * judged[hard]['usage']:.0f}%")]
    figs, images = [], {}
    if judged:
        su.img.mkdir(parents=True, exist_ok=True)
        rows = [{"name": n, "motor": su.motors[n], "w": np.r_[0.0, dyn.get(n, {}).get("w", [])],
                 "tau": np.r_[w["need"], dyn.get(n, {}).get("tau", [])]} for n, w in judged.items()][:9]
        images["torque_speed"] = plots.torque_speed(su.img / "torque_speed.png", rows)
        figs.append(["torque_speed", "Torque and speed each driven joint needs against what its drive gives"])
    assumptions = [f"The worst pose for each joint was searched over its whole range, {SEARCH} random poses then "
                   "a joint at a time, with contacts and joint stops left out."]
    if dyn:
        assumptions.append(f"Fast moves are {MOVES} point to point moves between random poses, each timed so the "
                           "fastest joint just reaches its speed limit" +
                           (", taken as 1 rad/s or 0.5 m/s since no drive states one." if su.cache.get("plain_speed")
                            else "."))
    if sp.dep:
        assumptions.append(f"The loops tie {', '.join(j['name'] for j in sp.dep)} to the driven joints.")
    return Finding("torque", verdict, sentence, metrics, checks, values, fixes, [], figs, images, assumptions)


def _search_check(sp, k: int, W: dict, n: int = SEARCH) -> Check:
    """The worst pose search again from another seed: a search that found the true worst lands in the same
    place."""
    a = abs(W["need"])
    b, _, _ = worst(sp, lambda d: abs(needed(sp, d)[k]), n=n, seed=7)
    gap = abs(a - b) / max(a, b, 1e-12)
    st = "pass" if gap < 0.01 else ("warn" if gap < 0.05 else "fail")
    return Check("worst pose search", st, f"searching again from another start finds {fmt(max(a, b), 4)} against "
                                          f"{fmt(min(a, b), 4)} at the {W['j']['name']} ({100 * gap:.2g}% apart)",
                 {"first": a, "second": b})


def _hand_check(sp, model, W: dict) -> list[Check]:
    if sp.dep:
        return []
    mujoco = _mj()

    d = mujoco.MjData(model)
    set_pose(sp, d, W["x"])
    mujoco.mj_comPos(model, d)
    mujoco.mj_comVel(model, d)
    f = np.zeros(model.nv)
    mujoco.mj_rne(model, d, 0, f)
    mj = float(f[W["j"]["dof"]])
    hand = hand_moment(model, d, W["j"])
    err = abs(mj - hand) / max(abs(hand), 1e-9)
    st = "pass" if err < 0.005 or abs(mj - hand) < 1e-6 else ("warn" if err < 0.02 else "fail")
    return [Check("hand statics", st, f"at the {W['j']['name']}'s worst pose the moments of the weights it carries "
                                      f"come to {amount(W['j'], hand, 4)}, and MuJoCo's dynamics gives "
                                      f"{amount(W['j'], mj, 4)} ({100 * err:.2g}% apart)",
                  {"hand": hand, "mujoco": mj})]


def _moves(su: Setup, sp, model, names: list[str]) -> dict:
    """Torque and speed along point to point moves between random poses, each timed so the fastest joint reaches
    its speed limit, from MuJoCo's inverse dynamics. Skipped when no drive states a speed limit."""
    mujoco = _mj()

    vmax = np.array([su.motors[n].no_load_speed if n in su.motors else math.inf for n in names])
    if not np.isfinite(vmax).any():     # no drive states a speed, so a brisk 1 rad/s or 0.5 m/s
        vmax = np.array([1.0 if j["kind"] == "hinge" else 0.5 for j in sp.free])
        su.cache["plain_speed"] = True
    su.log("solve", "fast moves between random poses at the speed limits")
    poses = sample(sp, 2 * MOVES + 1, seed=3)[1:]
    d = mujoco.MjData(model)
    out = {n: {"tau": [], "w": []} for n in names}
    dofs = [j["dof"] for j in sp.free]
    adr = [j["qadr"] for j in sp.free]
    for a, b in zip(poses[::2], poses[1::2]):
        dq = b - a
        T = max(float(np.max(np.where(np.isfinite(vmax), 1.875 * np.abs(dq) / vmax, 0.0))), 0.05)
        for s in np.linspace(0, 1, 41):
            d.qpos[:] = sp.q0
            d.qpos[adr] = a + dq * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
            d.qvel[:] = 0
            d.qacc[:] = 0
            d.qvel[dofs] = dq * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
            d.qacc[dofs] = dq * (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
            mujoco.mj_inverse(model, d)
            for k, n in enumerate(names):
                out[n]["tau"].append(float(d.qfrc_inverse[dofs[k]]))
                out[n]["w"].append(float(d.qvel[dofs[k]]))
    for n in names:
        tau, w = np.array(out[n]["tau"]), np.array(out[n]["w"])
        out[n] = {"tau": tau, "w": w, "usage": float(np.max(su.motors[n].usage(tau, w))) if n in su.motors else 0.0}
    return out


def _fixes(su: Setup, sp, model, judged: dict, short: list[str]) -> list[dict]:
    fixes = []
    for n in short[:4]:
        w = judged[n]
        need = 1.25 * abs(w["need"])
        line = {"kind": "drive", "title": f"Give the {n} at least {amount(w['j'], need)}",
                "summary": f"Its worst pose, {describe_pose(sp, w['x'])}, takes {amount(w['j'], abs(w['need']))}, "
                           "and a quarter more leaves room to move.", "auto": False,
                "after": {"joint": n, "needed": abs(w["need"]), "suggested": need}}
        pick = stronger(su.motors[n], need) if su.motors[n].key != "file" else None
        if pick is not None:
            line["summary"] += f" The {pick.name} clears it."
        fixes.append(line)
    if su.payload and short:
        top = su.cache.get("max_payload")
        if top is None and "rebuild" in su.cache:
            top = _max_payload(su, judged, short)
        if top is not None:
            fixes.append({"kind": "payload", "title": f"Keep the payload under about {fmt(top * 1000)} g",
                          "summary": "The torque each joint needs grows in step with the payload, so at each joint's "
                                     "worst pose this is the heaviest load its drive can hold, checked by searching "
                                     "the range again with it.", "auto": False, "after": {"max_payload_kg": top}})
    return fixes


def _max_payload(su: Setup, judged: dict, short: list[str]) -> float | None:
    """The heaviest payload every judged joint can hold: torque is affine in the payload at a fixed pose, so two
    payloads give the line at each joint's worst pose, and a search of the range with the answer confirms it."""
    mujoco = _mj()

    kg = su.payload[0]
    lines = []
    for p in (0.0, kg):
        m = quiet(su.cache["rebuild"](p).compile())
        sp = space(m, su.info)
        names = [j["name"] for j in sp.free]
        d = mujoco.MjData(m)
        vals = {}
        for n in judged:
            set_pose(sp, d, judged[n]["x"])
            vals[n] = float(needed(sp, d)[names.index(n)])
        lines.append(vals)
    top = math.inf
    for n, w in judged.items():
        t0, t1 = lines[0][n], lines[1][n]
        slope = (t1 - t0) / kg
        if abs(slope) < 1e-12:
            continue
        lim = 0.999 * w["limit"]
        for target in (lim, -lim):
            m_ = (target - t0) / slope
            if m_ >= 0:
                top = min(top, m_)
    if not math.isfinite(top):
        return None
    for _ in range(4):     # the worst pose can shift with the payload, so search the range again with it
        top = max(0.0, math.floor(top * 1000) / 1000)
        m = quiet(su.cache["rebuild"](top).compile())
        sp = space(m, su.info)
        names = [j["name"] for j in sp.free]
        use = max(worst(sp, lambda d, k=names.index(n): abs(needed(sp, d)[k]), n=SEARCH, seed=1)[0] /
                  judged[n]["limit"] for n in judged)
        if use <= 1.0 or top == 0:
            return top
        top /= use
    return top
