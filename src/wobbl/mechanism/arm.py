"""A robot arm doing a pick and place: the torque and speed each servo needs against its torque speed line, whether
one stalls or overheats, whether the arm hits the table or tips its base, and fixes checked by solving again.

The needs come from MuJoCo's inverse dynamics along the move, which is exact for a rigid arm and takes a fraction
of a second, so every fix is checked by solving the whole move again with the change in place. A forward run with
the real servo limits then shows what actually happens when a servo falls short.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from . import theory as T
from .motors import stronger
from .quantities import fmt_torque
from .sim import Trajectory, inverse_along, simulate
from .study import Answer, contact_check, energy_check, event, model_info, step_check, warnings_check
from .templates import arm as build_arm

MARGIN = 0.8          # a servo past this share of its line, or of its continuous rating, has little to spare
TRACK_DEG = 2.0       # a forward run further than this behind the move did not make it


def needs(raw: dict, n: int = 601) -> dict:
    """What the move asks of each servo with the arm clamped down: torque and speed over time, the worst share of
    each motor's line, and the RMS torque over the cycle against the motor's continuous rating."""
    b = build_arm({**raw, "mount": "clamped"})
    m = b.mech
    times = np.linspace(0, m.end_time, n)
    names, tau, com = inverse_along(m, times)
    traj = Trajectory(m.move, dict(m.move[0][1]))
    out = {"t": times, "joints": {}, "built": b, "com": com}
    for c, j in enumerate(names):
        mot = m.drive(j).at_joint
        w = np.array([traj.at(j, t)[1] for t in times])
        u = mot.usage(tau[:, c], w)
        k = int(np.argmax(u))
        lo, hi = mot.limit(w[k])
        rms = float(np.sqrt(np.mean(tau[:, c] ** 2)))
        out["joints"][j] = {"tau": tau[:, c], "w": w, "usage": u, "peak": float(u[k]), "k": k, "rms": rms,
                            "rated": mot.rated, "heat": rms / mot.rated, "motor": mot,
                            "avail": hi if tau[k, c] >= 0 else -lo}
    out["worst"] = max(v["peak"] for v in out["joints"].values())
    out["hot"] = max(v["heat"] for v in out["joints"].values())
    return out


def _passes(n: dict) -> bool:
    return n["worst"] <= 1 and n["hot"] <= 1


def answer(b, img: Path, say) -> Answer:
    info, m, raw = b.info, b.mech, dict(b.params)
    pitch, joints = info["pitch"], ["base", *info["pitch"]]
    say("solve", "inverse dynamics along the move")
    need = needs(raw)
    J = need["joints"]
    checks, events, fixes = [], [], []

    # hand statics at the pick pose, where the arm starts at rest holding the payload
    pose = info["poses"][0]
    hand = T.arm_statics(info["links"], [pose[j] for j in pitch], info["static_masses"])
    mj0 = [float(J[j]["tau"][0]) for j in pitch]
    err = max(abs(a - h) for a, h in zip(mj0, hand)) / max(max(abs(h) for h in hand), 1e-9)
    checks.append(Check("hand statics", "pass" if err < 0.005 else ("warn" if err < 0.02 else "fail"),
                        "at the pick pose the moments of the weights about each joint come to " +
                        ", ".join(f"{j} {fmt(h)} N m" for j, h in zip(pitch, hand)) + ", and MuJoCo's inverse "
                        f"dynamics needs {', '.join(fmt(x) for x in mj0)} N m ({100 * err:.2g}% apart)",
                        {"hand_Nm": hand, "mujoco_Nm": mj0}))

    say("solve", "running the move against the servos' limits")
    run = simulate(m, frames=9)
    say("verify", "again at half the time step")
    run2 = simulate(m, timestep=m.timestep / 2, frames=0)
    lag = {j: float(np.nanmax(np.abs(np.degrees(run.q[:, run.col(j)] - run.q_des[:, run.col(j)])))) for j in joints}
    worst_j = max(joints, key=lambda j: J[j]["peak"])
    behind = max(lag.values())
    ok_inverse = need["worst"] <= 1
    if ok_inverse == (behind < TRACK_DEG):
        st, s = "pass", (f"the run with the servos' real limits follows the move to {fmt(behind)} degrees, as the "
                         "inverse dynamics says it can" if ok_inverse else
                         f"the run with the servos' real limits falls {fmt(behind)} degrees behind the move, as the "
                         "inverse dynamics says it must")
    elif ok_inverse:
        st, s = "warn", (f"the inverse dynamics says every servo keeps up, but the run falls {fmt(behind)} degrees "
                         "behind, so a free base or a contact is adding load")
    else:
        st, s = "info", (f"the {worst_j} servo runs out briefly but the arm catches up, ending within "
                         f"{fmt(behind)} degrees of the move")
    checks.append(Check("forward run", st, s, {"lag_deg": lag}))
    c1, c2 = run.col(worst_j), run2.col(worst_j)
    if behind < TRACK_DEG:
        checks.append(step_check(float(np.max(np.abs(run.tau[:, c1]))), float(np.max(np.abs(run2.tau[:, c2]))),
                                 f"peak {worst_j} torque", " N m"))
    else:   # a servo pinned at its limit gives the same peak at any step, so compare how far the arm falls behind
        lag2 = float(np.nanmax(np.abs(np.degrees(run2.q[:, c2] - run2.q_des[:, c2]))))
        checks.append(step_check(lag[worst_j], lag2, f"how far the {worst_j} falls behind", " deg"))
    checks.append(energy_check(run))
    checks += [c for c in (contact_check(run, 0.03, support_only=True), warnings_check(run)) if c is not None]

    # what the run shows: a servo pinned at its limit, parts touching, a base tipping
    for j in joints:
        c = run.col(j)
        sat = np.nonzero(np.abs(run.tau_need[:, c]) > np.abs(run.tau[:, c]) + 1e-6)[0]
        if len(sat):
            events.append(event(f"{j} servo at its limit", run.t[sat[0]],
                                f"asks {fmt_torque(abs(run.tau_need[sat[0], c]))}, gets "
                                f"{fmt_torque(abs(run.tau[sat[0], c]))}"))
    hits = {k: v for k, v in run.contacts.items() if not v["support"]}
    for k, v in hits.items():
        events.append(event(_touch(k), v["first_s"], f"up to {fmt(v['peak_N'])} N"))
    tilt = run.free["base_free"]["tilt"] if "base_free" in run.free else np.zeros(1)
    tipped = float(np.max(tilt)) > 5
    if tipped:
        events.append(event("base tips", run.t[int(np.argmax(tilt > 5))], "tilted past 5 degrees"))
    if info["mount"] == "free":
        checks.append(_static_tip(need, info, tipped))
    tseg = info["segment_s"]
    for name, k in (("pick", 0), ("lifted", 1), ("over the place point", 2), ("placed", 3)):
        events.append(event(name, k * tseg))
    events.append(event("end of the hold", 3 * tseg + info["dwell_s"]))
    events.sort(key=lambda e: e["t_s"])

    stall = sorted((j for j in joints if J[j]["peak"] > 1), key=lambda j: -J[j]["peak"])
    hot = sorted((j for j in joints if J[j]["heat"] > 1), key=lambda j: -J[j]["heat"])
    payload_g = info["payload"] * 1000
    if stall or hot or hits or tipped:
        verdict, parts = "fails", []
        if stall:
            j = stall[0]
            v = J[j]
            k = v["k"]
            rpm = abs(v["w"][k]) * 60 / (2 * math.pi)
            parts.append(f"The {j} servo stalls. Carrying {fmt(payload_g)} g through this move needs "
                         f"{fmt_torque(abs(v['tau'][k]))} " + ("almost at rest" if rpm < 1 else f"at {fmt(rpm)} rpm")
                         + f", where the {v['motor'].name} gives {fmt_torque(max(v['avail'], 0.0))}.")
            if len(stall) > 1:
                parts.append(f"The {' and '.join(stall[1:])} servo{'s' if len(stall) > 2 else ''} also fall"
                             f"{'' if len(stall) > 2 else 's'} short.")
        elif hot:
            j = hot[0]
            parts.append(f"Every servo can make the move once, but the {j} servo overheats if the cycle repeats, "
                         f"averaging {fmt(J[j]['rms'])} N m RMS against about {fmt(J[j]['rated'])} N m it can carry "
                         "continuously.")
        if tipped:
            parts.append(f"The base tips over at {fmt(next(e['t_s'] for e in events if e['name'] == 'base tips'))} s.")
        if hits:
            k, v = next(iter(hits.items()))
            parts.append(f"{_touch(k).capitalize()} at {fmt(v['first_s'])} s.")
        head = " ".join(parts)
    elif need["worst"] > MARGIN or need["hot"] > MARGIN:
        verdict = "marginal"
        j = max(joints, key=lambda x: max(J[x]["peak"], J[x]["heat"]))
        head = (f"The arm makes the move with little to spare, as the {j} servo uses {100 * J[j]['peak']:.0f}% of "
                f"what it can give at speed and {100 * J[j]['heat']:.0f}% of its continuous rating.")
    else:
        verdict = "ok"
        head = (f"Every servo keeps up with margin. The busiest, the {worst_j}, uses {100 * J[worst_j]['peak']:.0f}% "
                f"of what it can give at speed and {100 * J[worst_j]['heat']:.0f}% of its continuous rating.")
    if verdict != "ok":
        say("fixes", "solving the move again with each fix")
        fixes = _fixes(raw, need, info, stall or hot or [worst_j], tipped)

    v = J[worst_j]
    k = v["k"]
    metrics = [("hardest joint", worst_j), ("needs", f"{fmt(abs(v['tau'][k]))} N m"),
               ("servo gives", f"{fmt(max(v['avail'], 0.0))} N m"), ("heat", f"{100 * v['heat']:.0f}% of rated")]
    img.mkdir(parents=True, exist_ok=True)
    images = {
        "torque_speed": plots.torque_speed(img / "torque_speed.png", [
            {"name": f"{j} ({J[j]['motor'].name.split(' at ')[0]})", "motor": J[j]["motor"], "w": J[j]["w"],
             "tau": J[j]["tau"]} for j in joints]),
        "usage": plots.series(img / "usage.png", need["t"], {j: 100 * J[j]["usage"] for j in joints},
                              "% of the torque the servo has at that speed", {"stall": 100.0}),
        "frames": plots.strobe(img / "frames.png", run, (("x", "z"), ("x", "y")),
                               highlight={"upper arm", "forearm", "hand"} if stall else None),
    }
    groups: dict[str, list[str]] = {}
    for j in joints:
        groups.setdefault(J[j]["motor"].name, []).append(j)

    def _list(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    servos = (f"Every joint has a {J[joints[0]]['motor'].describe()}" if len(groups) == 1 else
              "The arm has " + _list([f"a {name} at the {_list(js)}" for name, js in groups.items()]))
    assumptions = [
        f"{servos}. Each can carry {100 / 3:.0f}% of its stall torque continuously unless a rating is given.",
        f"The move picks at {_pt(info['targets'][0])} mm, lifts "
        f"{fmt((info['targets'][1][2] - info['targets'][0][2]) * 1000)} mm, swings over to "
        f"{_pt(info['targets'][3])} mm and lowers, each step in {fmt(tseg)} s on a minimum jerk profile, then holds "
        f"{fmt(info['dwell_s'])} s. The heating estimate repeats that loaded cycle.",
        "The servos follow the move by computed torque, as a well tuned controller would, and a hobby servo's own "
        "loop lags a little more.",
        "Each servo's motor, felt through its gearbox, adds the inertia of a 20 ms mechanical time constant, "
        "typical of small brushed motors, since servo datasheets do not give it.",
        f"The {fmt(payload_g)} g payload and the gripper are point masses at the tip, and the base is "
        + ("screwed to the table." if info["mount"] == "clamped" else
           f"standing free on the table ({fmt(info['base_mass'] * 1000)} g, "
           f"{fmt(info['base_size'][0] * 1000)} x {fmt(info['base_size'][1] * 1000)} mm)."),
    ]
    values = {"joints": {j: {"peak_usage": J[j]["peak"], "torque_Nm": float(J[j]["tau"][J[j]["k"]]),
                             "speed_rpm": float(J[j]["w"][J[j]["k"]] * 60 / (2 * math.pi)),
                             "available_Nm": float(J[j]["avail"]), "rms_Nm": J[j]["rms"], "rated_Nm": J[j]["rated"],
                             "hold_Nm": float(J[j]["tau"][0]), "lag_deg": lag[j], "motor": J[j]["motor"].name}
                         for j in joints},
              "payload_kg": info["payload"], "mount": info["mount"], "tipped": tipped}
    return Answer(verdict, head, metrics, checks, assumptions, values, fixes, events,
                  [["torque_speed", "Torque and speed each servo needs against its torque speed line"],
                   ["usage", "Share of each servo's available torque through the move"],
                   ["frames", "The move at several instants, from the side and from above"]],
                  images, run, model=model_info(m, run))


def _touch(pair: str) -> str:
    """'floor / hand' reads as 'the hand hits the table', anything else as 'the forearm touches the base'."""
    a, b = pair.split(" / ")
    if "floor" in (a, b):
        return f"the {b if a == 'floor' else a} hits the table"
    return f"the {a} touches the {b}"


def _pt(p) -> str:
    return ", ".join(fmt(x * 1000) for x in p)


def _static_tip(need: dict, info: dict, tipped: bool) -> Check:
    """Whether the center of mass of the whole robot leaves the base's footprint at any point of the move."""
    bw, bd = info["base_size"]
    com = need["com"]
    out = np.maximum(np.abs(com[:, 0]) / (bw / 2), np.abs(com[:, 1]) / (bd / 2))
    k = int(np.argmax(out))
    static = out[k] > 1
    s = (f"the center of mass reaches {100 * out[k]:.0f}% of the way to the base's edge at {fmt(need['t'][k])} s"
         + (", past it, so the weight alone tips it" if static else ", so the weight alone keeps it down"))
    if static == tipped:
        return Check("hand statics, tipping", "pass", s + ", as the run shows", {"edge_share": float(out[k])})
    if tipped:
        return Check("hand statics, tipping", "info", s + ", and it tips in the run from the arm's acceleration",
                     {"edge_share": float(out[k])})
    return Check("hand statics, tipping", "warn", s + ", yet it stays up in the run", {"edge_share": float(out[k])})


RATIOS = (1.5, 2, 2.5, 3, 4, 5, 6, 8)


def _bad(n: dict) -> list[str]:
    return [j for j, v in n["joints"].items() if v["peak"] > MARGIN or v["heat"] > 1]


def _state(n: dict) -> str:
    return (f"the busiest servo uses {100 * n['worst']:.0f}% of its torque at speed and the warmest "
            f"{100 * n['hot']:.0f}% of its continuous rating"
            + ("." if _passes(n) else ", which is still not enough."))


def _fixes(raw: dict, need: dict, info: dict, failing: list[str], tipped: bool) -> list[dict]:
    J = need["joints"]
    fixes = []
    # the smallest belt or gear stage that brings each failing pitch joint inside its own servo, all together
    gears = {}
    for j in [x for x in _bad(need) if x != "base"]:
        for r in RATIOS:
            v = needs({**raw, f"{j}_ratio": r})["joints"][j]
            if v["peak"] <= MARGIN and v["heat"] <= 1:
                gears[j] = r
                break
    if gears:
        n = needs({**raw, **{f"{j}_ratio": r for j, r in gears.items()}})
        what = " and ".join(f"the {j} through {r:g}:1" for j, r in gears.items())
        fixes.append({"kind": "reduction", "title": f"Drive {what} with a belt or gear stage",
                      "summary": "Each servo then sees a fraction of the torque and turns that much faster, with 10% "
                                 f"lost in the stage. Solving the move again, {_state(n)}", "auto": False,
                      "after": {"ratios": gears, "worst_usage": n["worst"], "heat": n["hot"]}})
    # stronger servos, picked again while a heavier servo pushes the joints behind it over; a joint no servo in the
    # catalog can carry keeps the reduction found above
    picks, extra, n, none = {}, {}, need, []
    for _ in range(3):
        bad = _bad(n)
        if not bad:
            break
        for j in bad:
            v = n["joints"][j]
            p = stronger(v["motor"], abs(v["tau"][v["k"]]), abs(v["w"][v["k"]]), v["rms"])
            if p is not None and j not in extra:
                picks[j] = p
            elif j in gears:
                extra[j] = gears[j]
            elif j not in none:
                none.append(j)
        n = needs({**raw, **{f"{j}_servo": p.key for j, p in picks.items()},
                   **{f"{j}_ratio": r for j, r in extra.items()}})
    if picks:
        parts = [f"a {p.name.split(' at ')[0]} at the {j}" for j, p in picks.items()]
        parts += [f"a {r:g}:1 stage on the {j}" for j, r in extra.items()]
        fixes.append({"kind": "servo", "title": "Use " + " and ".join(parts),
                      "summary": (f"No servo in Wobbl's catalog carries the {' or '.join(none)} continuously. " if none
                                  else "") + f"Solving the move again with {'these' if len(parts) > 1 else 'it'}, "
                                 + _state(n), "auto": False,
                      "after": {"servos": {j: p.key for j, p in picks.items()}, "ratios": extra,
                                "worst_usage": n["worst"], "heat": n["hot"]}})
    # a spring or counterweight carrying half of the shoulder's holding torque
    if "shoulder" in _bad(need):
        hold = abs(float(J["shoulder"]["tau"][0]))
        arm = 0.1
        cw = math.ceil(0.5 * hold / (T.G * arm) * 100) / 100
        n = needs({**raw, "counterweight": f"{cw * 1000:g} g", "counterweight_arm": f"{arm * 1000:g} mm"})
        v = n["joints"]["shoulder"]
        fixes.append({"kind": "balance", "title": f"Balance the shoulder with a {fmt(hold / 2)} N m spring or "
                                                   f"{fmt(cw * 1000)} g on a {fmt(arm * 1000)} mm tail",
                      "summary": "Either carries about half the holding torque. Solving again with the counterweight, "
                                 f"the shoulder uses {100 * v['peak']:.0f}% of its torque at speed and "
                                 f"{100 * v['heat']:.0f}% of its continuous rating, and the base servo swings the "
                                 "extra mass, where a spring adds none.", "auto": False,
                      "after": {"counterweight_kg": cw, "usage": v["peak"], "heat": v["heat"]}})
    # the heaviest payload these servos carry through the same move
    if not _passes(need):
        top = _heaviest(raw, info["payload"], _passes)
        if top > 0.005:
            once = _heaviest(raw, info["payload"], lambda x: x["worst"] <= 1)
            fixes.append({"kind": "payload", "title": f"Keep the payload under about {fmt(top * 1000)} g",
                          "summary": "That is the heaviest load these servos carry through the same move over and "
                                     "over without running hot, found by solving the move at each weight. A single "
                                     f"move without a stall works up to about {fmt(once * 1000)} g.",
                          "auto": False, "after": {"max_payload_kg": top, "max_single_move_kg": once}})
        else:
            cool = _heaviest(raw, info["payload"], lambda x: x["worst"] <= 1)
            fixes.append({"kind": "payload", "title": "No payload makes these servos safe for this move",
                          "summary": (f"Below about {fmt(cool * 1000)} g nothing stalls, but " if cool > 0.005 else
                                      "Even empty a servo stalls, and ") + "holding the arm's own weight already runs "
                                     "a servo past its continuous rating, so a reduction or a stronger servo is "
                                     "needed whatever it carries.", "auto": False})
    if tipped:
        fixes.append(_ballast(raw, need, info))
    return fixes


def _heaviest(raw: dict, start: float, good) -> float:
    lo, hi = 0.0, start
    if not good(needs({**raw, "payload": "0 g"}, n=301)):
        return 0.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if good(needs({**raw, "payload": f"{mid * 1000:g} g"}, n=301)):
            lo = mid
        else:
            hi = mid
    return lo


def _ballast(raw: dict, need: dict, info: dict) -> dict:
    """Mass to add at the middle of the base so the robot's center of mass stays within 80% of its footprint."""
    bw, bd = info["base_size"]
    m_tot = sum(mm for _, _, mm in info["static_masses"]) + info["base_mass"] + 0.05
    extra = 0.0
    for c in need["com"]:
        r = math.hypot(c[0], c[1])
        if r < 1e-9:
            continue
        psi = math.atan2(c[1], c[0])
        edge = min(bw / 2 / max(abs(math.cos(psi)), 1e-9), bd / 2 / max(abs(math.sin(psi)), 1e-9))
        extra = max(extra, m_tot * (r / (0.8 * edge) - 1))
    extra = math.ceil(extra * 20) / 20
    run = simulate(build_arm({**raw, "base_mass": f"{(info['base_mass'] + extra) * 1000:g} g"}).mech, frames=0)
    stays = float(np.max(run.free["base_free"]["tilt"])) < 5 if "base_free" in run.free else True
    return {"kind": "ballast", "title": f"Add {fmt(extra * 1000)} g of ballast to the base, or screw it down",
            "summary": "That keeps the center of mass inside 80% of the footprint at every point of the move. "
                       + ("Running the move again with it, the base stays flat." if stays else
                          "Running the move again with it, the arm's swing still tips it, so screw it down."),
            "auto": False, "after": {"base_mass_kg": info["base_mass"] + extra, "stays": stays}}
