"""Standing on a floor: what each foot or wheel carries, the torque the joints need to hold the stance, and whether
the whole thing tips, for any model with a base that can lift off, or any fixed base robot set loose on a floor.

The model is put on a floor and run until it settles, with every joint that has a drive held at its starting pose
by that drive and the others left free. Once settled, the floor's push has to add up to the weight, which is the
hand check, and the center of mass has to sit inside the outline of the contacts. For a robot that stands on a
base and moves an arm, every pose of the arm is searched for the one that brings the center of mass nearest the
edge, and that pose is run too, to see whether it really tips.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .findings import Finding, Setup, amount
from .holding import quiet
from .install import require as _mj
from .model import describe, loaded, start_key, start_qpos
from .poses import describe_pose, set_pose, space, worst
from .sim import simulate
from .spec import Drive
from .study import contact_check, event, typical_size, warnings_check

SETTLE = 1.5             # s to stand and settle
FALLEN = 45.0            # degrees of tilt that count as fallen over


def _mech(su: Setup, extra_edits: dict | None = None, hold: dict | None = None, quick: bool = False):
    """The mechanism standing on a floor, each driven joint held at ``hold`` (its starting angle by default)."""
    m = su.mech
    edits = {**m.edits, "stand": True, **(extra_edits or {})}
    if not su.info.floating:
        edits["free_base"] = True
    base = dataclasses.replace(m, edits=edits)
    model = quiet_copy(base)
    info = describe(model)
    q0 = {j["name"]: float(start_qpos(model)[j["qadr"]]) for j in info.joints}
    drives, move = [], []
    if not su.passive:
        for j in info.joints:
            if j["name"] in su.motors:
                drives.append(Drive(j["name"], su.motors[j["name"]], "servo"))
        if drives:
            start = {d.joint: q0[d.joint] for d in drives}
            if hold:
                move = [(0.0, start), (0.6, {**start, **hold})]
            else:
                move = [(0.0, start)]
    T = 0.8 if quick else SETTLE
    return dataclasses.replace(base, drives=drives, move=move, hold=T if move else 0.0, duration=T, spin={}), info


def quiet_copy(m):
    from .mjcf import build
    return build(m)


def settle(su: Setup, quick: bool = False) -> dict:
    """Stand the model and run it until it settles, with every driven joint held rigidly, which shows what it
    stands on and what holding the stance takes. When that is more than a drive gives, it is run again with each
    drive held to its limit, to see whether it sags or falls. Cached, since several analyses read it."""
    if "settle" in su.cache:
        return su.cache["settle"]
    su.log("solve", "standing it on the floor until it settles")
    mech, info = _mech(su, quick=quick)
    axis = planar_axis(quiet_copy(mech))
    run = simulate(mech, frames=9, late=mech.end_time * 0.7, ideal=True)
    s = _read(run, info, axis)
    s["mech"], s["info"], s["axis"] = mech, info, axis
    short = []
    late = run.t >= run.t[-1] * 0.7
    for d in mech.drives:
        if d.motor is not None and math.isfinite(d.motor.stall_torque):
            need = float(np.mean(np.abs(run.tau_need[late, run.col(d.joint)])))
            if need > d.motor.stall_torque:
                short.append(d.joint)
    s["short"] = short
    if short:
        su.log("solve", "again with each drive held to its limit")
        s["real"] = _read(simulate(mech, frames=9, late=mech.end_time * 0.7), info, axis)
    su.cache["settle"] = s
    return s


def planar_axis(model):
    """The direction a planar model moves along the floor, from its root's level slide, or None for a model that
    is free in every direction."""
    mujoco = _mj()
    for i in range(model.njnt):
        if model.body_parentid[model.jnt_bodyid[i]] == 0 and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return None
    for i in range(model.njnt):
        if model.body_parentid[model.jnt_bodyid[i]] == 0 and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE \
                and abs(model.jnt_axis[i, 2]) < 0.5:
            a = np.array(model.jnt_axis[i, :2], float)
            return a / max(np.linalg.norm(a), 1e-12)
    return None


def _read(run, info, axis=None) -> dict:
    """What a settle run shows: each body's share of the floor's push, the support outline, the center of mass,
    whether the base fell over, and how steady it got."""
    n = max(run.model_info.get("late_samples", 0), 1)
    feet = {}
    for k, c in run.contacts.items():
        if "world" not in c["bodies"] or "late_fn" not in c:
            continue
        body = [b for b in c["bodies"] if b != "world"][0]
        e = feet.setdefault(body, {"fz": 0.0, "fn": 0.0, "pos": np.zeros(3)})
        e["fz"] += c["late_fz"] / n
        e["fn"] += c["late_fn"] / n
        e["pos"] += c["late_pos"] / n
    for e in feet.values():
        e["pos"] = e["pos"] / max(e["fn"], 1e-12)
    root = next(iter(run.free.values()), None)
    tilt = float(np.max(root["tilt"])) if root else 0.0
    fell = tilt > FALLEN
    speed = float(np.linalg.norm(root["vel"][-1])) if root else 0.0
    weight = run.model_info["mass_kg"] * float(np.linalg.norm(run.model_info["gravity"]))
    com = run.model_info["com"]
    # the outline it stands on comes from every contact point at the end, since one foot can touch at several
    last = [c for c in run.model_info.get("final_contacts", []) if "world" in c["bodies"] and c["fz"] > 1e-4 * weight]
    pts = np.array([c["pos"][:2] for c in last]) if last else np.zeros((0, 2))
    if axis is not None and len(pts):       # a planar model can only tip along its plane
        p = pts @ axis
        c = float(com[:2] @ axis)
        mg = (min(c - p.min(), p.max() - c), float(np.ptp(p)) / 2)
    else:
        mg = margin(com[:2], pts)
    return {"run": run, "feet": feet, "fell": fell, "tilt": tilt, "speed": speed, "weight": weight, "com": com,
            "support": pts, "margin": mg}


def margin(c, pts) -> tuple[float, float]:
    """Signed distance from point ``c`` to the edge of the outline of ``pts``, inside positive, and the outline's
    half size, m. A single contact or a line of them gives no area, so the margin is how far off it the point is."""
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return -math.inf, 0.0
    if len(pts) >= 3:
        try:
            from scipy.spatial import ConvexHull

            h = ConvexHull(pts)
            eq = h.equations                  # a x + b y + c <= 0 inside, with (a, b) a unit normal
            dist = float(np.min(-(eq[:, :2] @ c + eq[:, 2])))
            size = float(np.sqrt(h.volume / math.pi))
            return dist, size
        except Exception:  # noqa: BLE001  collinear points fall through to the line case
            pass
    if len(pts) == 1:
        return -float(np.linalg.norm(c - pts[0])), 0.0
    a, b = pts.min(axis=0), pts.max(axis=0)
    seg = b - a
    t = float(np.clip((c - a) @ seg / max(seg @ seg, 1e-18), 0, 1))
    return -float(np.linalg.norm(c - (a + t * seg))), float(np.linalg.norm(seg) / 2)


def contact(su: Setup) -> Finding:
    s = settle(su, su.quick)
    run, feet, weight = s["run"], s["feet"], s["weight"]
    total = sum(e["fz"] for e in feet.values())
    settled = s["speed"] < 0.05 and not s["fell"]
    err = abs(total - weight) / max(weight, 1e-12)
    checks = [Check("hand statics", ("pass" if err < 0.02 else "warn" if err < 0.05 else "fail") if settled else "info",
                    f"once settled the floor pushes up with {fmt(total)} N in all, against the {fmt(weight)} N it "
                    f"weighs ({100 * err:.2g}% apart)" if settled else
                    f"it is still moving at the end, so the floor's {fmt(total)} N need not match its {fmt(weight)} N "
                    "weight yet", {"floor_N": total, "weight_N": weight})]
    su.log("verify", "standing it again at half the time step")
    s2 = _read(simulate(s["mech"], timestep=run.dt / 2, frames=0, poses=0, late=s["mech"].end_time * 0.7,
                        ideal=True), s["info"], s["axis"])
    if feet:
        diff = max(abs(feet[b]["fz"] - s2["feet"].get(b, {"fz": 0.0})["fz"]) for b in feet) / max(weight, 1e-12)
        st = "pass" if diff < 0.01 else ("warn" if diff < 0.05 else "fail")
        checks.append(Check("time step convergence", st, f"at half the time step each contact's share moves by at "
                                                         f"most {100 * diff:.2g}% of the weight", {"change": diff}))
    if s["fell"]:      # forces at rest were never reached, so how deep the fall pressed in tells nothing
        checks.append(Check("contact settings", "info", "it fell over, so the contact depth at rest was not reached"))
    else:
        checks += [c for c in (contact_check(run, typical_size(run), settled=True),) if c]
    checks += [c for c in (warnings_check(run),) if c]
    rows = sorted(feet.items(), key=lambda kv: -kv[1]["fz"])
    share = "; ".join(f"the {b} {fmt(e['fz'])} N ({100 * e['fz'] / max(total, 1e-12):.0f}%)" for b, e in rows[:6])
    if s["fell"]:
        verdict = "fails"
        sentence = (f"Even held rigid it does not stay standing: it tilts past {FALLEN:.0f} degrees within "
                    f"{fmt(run.t[-1])} s.")
    elif not rows:
        verdict = "fails"
        sentence = "Nothing touches the floor once it settles, so it is not standing on anything."
    else:
        verdict = "ok"
        sentence = f"Standing still, it rests on {len(rows)} contact{'s' if len(rows) > 1 else ''}: {share}."
    values = {"forces_N": {b: e["fz"] for b, e in rows}, "weight_N": weight, "floor_N": total,
              "settled": settled, "max_tilt_deg": s["tilt"]}
    events = [event("set down", 0.0)]
    if s["fell"]:
        root = next(iter(run.free.values()))
        events.append(event("falls over", run.t[int(np.argmax(root["tilt"] > FALLEN))]))
    su.img.mkdir(parents=True, exist_ok=True)
    images, figs = {}, []
    if rows:
        images["forces"] = plots.bars(su.img / "forces.png", [b for b, _ in rows[:10]],
                                      [e["fz"] for _, e in rows[:10]], "N", {"weight / contacts": weight / len(rows)})
        figs.append(["forces", "What each part touching the floor carries once settled"])
    images["frames"] = plots.strobe(su.img / "frames.png", run, (("x", "z"), ("x", "y")))
    figs.append(["frames", "The model settling on the floor"])
    a = [f"It starts just above the floor and settles for {fmt(run.t[-1])} s with each driven joint held rigidly at "
         "its starting angle and the rest left free, so the forces are those of the stance itself."]
    return Finding("contact", verdict, sentence, [("contacts", str(len(rows))), ("weight", f"{fmt(weight)} N")],
                   checks, values, [], events, figs, images, a, run=run)


def stance_torque(su: Setup) -> Finding:
    """The torque each driven joint needs to hold the stance, from the settled run, against its drive."""
    s = settle(su)
    run = s["run"]
    late = run.t >= run.t[-1] * 0.7
    rows = {}
    for j in s["info"].joints:
        if j["name"] not in run.joints or j["name"] not in su.motors:
            continue
        c = run.col(j["name"])
        need = float(np.mean(run.tau_need[late, c]))
        lim = su.motors[j["name"]].stall_torque
        rows[j["name"]] = {"j": j, "need": need, "limit": lim,
                           "usage": abs(need) / lim if math.isfinite(lim) and lim > 0 else 0.0}
    if not rows:
        return Finding("torque", "info", "Nothing in the model drives its joints, so there is no motor torque to "
                                         "check while it stands.")
    judged = {n: r for n, r in rows.items() if math.isfinite(r["limit"])}
    pool = judged or rows
    hard = max(pool, key=lambda n: pool[n]["usage"] if judged else abs(pool[n]["need"]))
    w = rows[hard]
    fell = s.get("real", {}).get("fell", False)
    if s["fell"]:
        return Finding("torque", "fails", "Even with every joint held rigid it falls over, so no drive can hold this "
                                          "stance, however strong.", [("hardest joint", hard)], [],
                       {"joints": {n: {"needed": r["need"], "limit": r["limit"]} for n, r in rows.items()},
                        "falls_rigid": True}, [], [], [], {}, [], run=run)
    if not judged:
        verdict = "info"
        sentence = (f"The model gives no motor limits, so here is what holding the stance takes: up to "
                    f"{amount(w['j'], abs(w['need']))} at the {hard}.")
    elif s["short"]:
        verdict = "fails"
        sentence = (f"The {hard} cannot hold the stance: it needs {amount(w['j'], abs(w['need']))} against the "
                    f"{amount(w['j'], w['limit'])} its drive gives, and held to that limit it "
                    + ("falls over." if fell else "sags."))
    elif w["usage"] > 0.8:
        verdict = "marginal"
        sentence = (f"The drives hold the stance, but the {hard} has little to spare, at "
                    f"{amount(w['j'], abs(w['need']))} of its {amount(w['j'], w['limit'])}.")
    else:
        verdict = "ok"
        sentence = (f"The drives hold the stance easily, the busiest, the {hard}, at "
                    f"{amount(w['j'], abs(w['need']))} of its {amount(w['j'], w['limit'])} ({100 * w['usage']:.0f}%).")
    tail = run.tau_need[late][:, run.col(hard)]
    spread = float(np.ptp(tail)) / max(abs(w["need"]), 1e-9)
    checks = [Check("settled", "pass" if spread < 0.05 else "warn",
                    f"over the last part of the run the {hard}'s torque stays within {100 * spread:.2g}% of its mean"
                    + (", so the stance has settled" if spread < 0.05 else ", so it was still moving"),
                    {"spread": spread})]
    values = {"joints": {n: {"needed": r["need"], "limit": r["limit"], "usage": r["usage"]} for n, r in rows.items()},
              "hardest": hard, "falls_at_limits": fell,
              "worst_usage": max((r["usage"] for r in judged.values()), default=None)}
    fixes = [{"kind": "drive", "title": f"Give the {n} at least {amount(rows[n]['j'], 1.25 * abs(rows[n]['need']))}",
              "summary": "That holds the stance with a quarter to spare.", "auto": False} for n in s["short"][:4]]
    return Finding("torque", verdict, sentence, [("hardest joint", hard), ("use", f"{100 * w['usage']:.0f}%")],
                   checks, values, fixes, [], [], {},
                   ["Torques are what each drive gives once it has settled holding its starting angle."],
                   run=s.get("real", s)["run"])


def stability(su: Setup) -> Finding:
    s = settle(su)
    run = s["run"]
    d0, size = s["margin"]
    checks = []
    if s["fell"]:
        verdict = "fails"
        sentence = f"It tips over standing still, tilting past {FALLEN:.0f} degrees."
    elif d0 < -1e-4:
        verdict = "fails"
        sentence = (f"Its center of mass sits {fmt(-d0 * 1000)} mm outside what it stands on, so it cannot stay up "
                    "for long.")
    else:
        verdict = "ok" if d0 > 0.1 * size else "marginal"
        sentence = (f"It stands, with its center of mass {fmt(max(d0, 0.0) * 1000)} mm inside the edge of what it "
                    "stands on" + ("." if verdict == "ok" else ", which is little to spare."))
    statics_tip = d0 < -1e-4
    checks.append(Check("tipping statics", "pass" if statics_tip == s["fell"] else "warn",
                        f"the center of mass sits {fmt(abs(d0) * 1000)} mm {'inside' if d0 >= 0 else 'outside'} the "
                        f"outline of the contacts and the run {'tips' if s['fell'] else 'stays up'}"
                        + (", as the moments say" if statics_tip == s["fell"] else ", against what the moments say"),
                        {"margin_m": d0}))
    values = {"margin_m": d0, "support_halfsize_m": size, "fell": s["fell"], "max_tilt_deg": s["tilt"]}
    fixes, events = [], []
    arm = [j for j in s["info"].joints if j["name"] in su.motors and j["limited"]]
    if not s["fell"] and not su.info.floating and arm and len(s["support"]) >= 1:
        w = _worst_pose(su, s, arm)
        values.update({"worst_margin_m": w["margin"], "worst_pose": w["pose"], "tips_in_worst": w["tips"],
                       "_poses": {"nearest the edge": w["hold"]}})
        checks.append(Check("tipping statics, worst pose", "pass" if (w["margin"] < 0) == w["tips"] else "warn",
                            f"at {w['pose']} the center of mass sits {fmt(abs(w['margin']) * 1000)} mm "
                            f"{'inside' if w['margin'] >= 0 else 'outside'} the base and running that pose it "
                            f"{'tips' if w['tips'] else 'stays up'}", {"margin_m": w["margin"]}))
        if w["tips"] or w["margin"] < 0:
            verdict = "marginal" if verdict == "ok" else verdict
            sentence += f" Moved to {w['pose']}, it tips over."
            fixes.append(_ballast(s, w))
        else:
            sentence += (f" In its worst pose, {w['pose']}, the center of mass is still {fmt(w['margin'] * 1000)} mm "
                         "inside.")
    if s["fell"]:
        root = next(iter(run.free.values()))
        events.append(event("tips past 45 degrees", run.t[int(np.argmax(root["tilt"] > FALLEN))]))
    images = {"tilt": plots.series(su.img / "tilt.png", run.t, {"tilt": next(iter(run.free.values()))["tilt"]},
                                   "degrees")} if run.free else {}
    return Finding("stability", verdict, sentence, [("margin", f"{fmt(d0 * 1000)} mm")], checks, values, fixes,
                   events, [["tilt", "Tilt of the base while it settles"]] if images else [], images,
                   ["The center of mass is judged against the outline of everything touching the floor once settled, "
                    "with the joints held rigid."], run=run)


def _worst_pose(su: Setup, s: dict, arm: list[dict]) -> dict:
    """The arm pose that brings the center of mass nearest the edge of the base, then run to see if it tips."""
    mujoco = _mj()

    model = quiet(loaded(s["mech"]).compile())
    model.key_qpos[start_key(model)] = s["run"].model_info["final_qpos"]
    info = describe(model)
    sp = space(model, info, [j["name"] for j in arm])
    pts = s["support"]
    best, x, _ = worst(sp, lambda d: -_com_margin(model, d, pts), n=300, seed=4)
    pose = describe_pose(sp, x)
    hold = {j["name"]: float(v) for j, v in zip(sp.free, x)}
    mech, _ = _mech(su, hold=hold)
    run = simulate(mech, frames=0, poses=0, ideal=True)
    root = next(iter(run.free.values()), None)
    tips = bool(root is not None and np.max(root["tilt"]) > FALLEN / 3)
    d = mujoco.MjData(model)
    set_pose(sp, d, x)
    return {"margin": -best, "pose": pose, "tips": tips, "com": d.subtree_com[0].copy(), "mass": model.body_mass.sum(),
            "hold": hold}


def _com_margin(model, d, pts) -> float:
    mujoco = _mj()
    mujoco.mj_comPos(model, d)
    return margin(d.subtree_com[0][:2], pts)[0]


def _ballast(s: dict, w: dict) -> dict:
    """Mass at the middle of the base that pulls the worst pose's center of mass back to 80% of the way to the
    edge."""
    pts = s["support"]
    c0 = pts.mean(axis=0)
    c = w["com"][:2]
    e = float(np.linalg.norm(c - c0))
    edge = e - w["margin"]                 # distance from the middle to the edge along that line, roughly
    m = float(w["mass"])
    extra = max(0.0, m * (e / max(0.8 * edge, 1e-9) - 1))
    extra = math.ceil(extra * 20) / 20
    return {"kind": "ballast", "title": f"Add about {fmt(extra * 1000)} g of ballast low in the middle of the base",
            "summary": "That pulls the center of mass back to 80% of the way to the edge in the worst pose, or fix "
                       "the base down.", "auto": False, "after": {"ballast_kg": extra}}
