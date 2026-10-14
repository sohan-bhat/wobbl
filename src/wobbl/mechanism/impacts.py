"""Parts that hit and things that tip.

An impact's rebound and the momentum it trades are rigid body results that MuJoCo gets right whatever the contact
stiffness, but its peak force rests on that stiffness, so Wobbl tunes the contact to Hertz theory for the two
materials and says so. Tipping comes down to moments about an edge, which the run's floor reactions have to match.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from . import theory as T
from .analyses import with_
from .sim import simulate
from .study import Answer, contact_check, energy_check, event, model_info, step_check, warnings_check


def _hit(run) -> dict | None:
    hits = [v for v in run.contacts.values() if not v["support"]]
    return max(hits, key=lambda v: v["peak_N"]) if hits else None


def _after(run, info) -> tuple[float, float]:
    """Velocities along the line of impact once the parts separate (for a drop, the bounce speed, with the
    gravity since separation added back)."""
    part = run.free["part_free"]["vel"][-1]
    if info["fixed"]:
        h = _hit(run)
        since = run.t[-1] - (h["last_s"] if h else run.t[-1])
        return float(part[2] + T.G * since), 0.0
    return float(part[0]), float(run.free["target_free"]["vel"][-1][0])


def impact(b, img: Path, say) -> Answer:
    info, m = b.info, b.mech
    say("solve", "running the impact")
    run = simulate(m, samples=3000, frames=7)
    say("verify", "again at half the time step and with a softer contact")
    run2 = simulate(m, timestep=m.timestep / 2, samples=3000, frames=0)
    k, c = m.contact["stiffness"], m.contact["damping"]
    soft = with_(m, contact={**m.contact, "stiffness": k / 4, "damping": c / 2}, duration=2 * m.duration)
    run3 = simulate(soft, samples=3000, frames=0)
    h, h2, h3 = _hit(run), _hit(run2), _hit(run3)
    if h is None:
        raise ValueError("the parts never touched, so check the gap and the speeds")
    v1, v2, e = info["v1"], info["v2"], info["e"]
    vrel = v1 - v2
    a1, a2 = _after(run, info)
    s1, s2 = _after(run3, info)
    fixed = info["fixed"]
    checks = []
    e_sim = a1 / v1 if fixed else (a2 - a1) / vrel
    e_err = abs(e_sim - e) / max(e, 1e-9) if e > 0 else abs(e_sim)
    checks.append(Check("restitution", "pass" if e_err < 0.01 else ("warn" if e_err < 0.03 else "fail"),
                        f"the parts come apart at {fmt(e_sim, 4)} of their closing speed against the {fmt(e)} the "
                        f"contact was tuned for ({100 * e_err:.2g}% apart)", {"restitution": e_sim}))
    if not fixed:
        p0, p1 = info["m1"] * v1 + info["m2"] * v2, float(run.momentum[-1][0])
        perr = abs(p1 - p0) / max(abs(info["m1"] * v1) + abs(info["m2"] * v2), 1e-12)
        checks.append(Check("momentum", "pass" if perr < 1e-4 else ("warn" if perr < 1e-2 else "fail"),
                            f"total momentum is {fmt(p1, 5)} kg m/s after the hit and was {fmt(p0, 5)} before "
                            f"({100 * perr:.2g}% apart)", {"before": p0, "after": p1}))
        x1, x2 = T.collision(info["m1"], v1, info["m2"], v2, e)
        verr = max(abs(a1 - x1), abs(a2 - x2)) / abs(vrel)
        checks.append(Check("hand calculation", "pass" if verr < 0.01 else ("warn" if verr < 0.03 else "fail"),
                            f"momentum and restitution give {fmt(x1)} and {fmt(x2)} m/s afterward, and MuJoCo gives "
                            f"{fmt(a1)} and {fmt(a2)} m/s ({100 * verr:.2g}% of the closing speed apart)",
                            {"hand_mps": [x1, x2], "mujoco_mps": [a1, a2]}))
    lin = info["linear_peak_N"]
    ferr = abs(h["peak_N"] - lin) / lin
    hz = info["hertz"]
    mat1, mat2 = info["materials"]
    checks.append(Check("peak force", "pass" if ferr < 0.02 else ("warn" if ferr < 0.05 else "fail"),
                        f"the contact's spring and dashpot, solved exactly, peak at {fmt(lin)} N and MuJoCo peaks at "
                        f"{fmt(h['peak_N'])} N ({100 * ferr:.2g}% apart)", {"exact_N": lin}))
    checks.append(Check("Hertz contact", "info",
                        f"Hertz theory for {mat1.name} on {mat2.name} with a {fmt(info['radii'][0] * 1000)} mm round "
                        f"contact gives a {fmt(hz['force_N'])} N peak over {fmt(hz['time_s'] * 1000, 3)} ms, and "
                        "Wobbl's linear contact stores the same energy at that peak and lasts "
                        f"{fmt(info['contact_time_s'] * 1000, 3)} ms" if not info["k_given"] else
                        "the contact stiffness is the one given, not Hertz's",
                        {"hertz_N": hz["force_N"], "hertz_ms": hz["time_s"] * 1000}))
    same = max(abs(s1 - a1), abs(s2 - a2)) / abs(vrel)
    checks.append(Check("contact softness", "pass" if same < 0.01 else "warn",
                        f"a contact four times softer brings the peak down to {fmt(h3['peak_N'] if h3 else 0)} N, as a "
                        f"spring's square root law says, while the speeds afterward move {100 * same:.2g}% of the "
                        "closing speed, so the bounce does not depend on the softness while the force does",
                        {"soft_peak_N": h3["peak_N"] if h3 else None}))
    checks.append(step_check(h["peak_N"], h2["peak_N"], "peak contact force", " N"))
    if e >= 1:
        checks.append(energy_check(run, passive=True))
    else:
        lost = -float(run.work["constraints"][-1])
        want = (1 - e * e) * 0.5 * info["m_eff"] * vrel * vrel
        lerr = abs(lost - want) / want
        checks.append(Check("energy lost", "pass" if lerr < 0.02 else ("warn" if lerr < 0.05 else "fail"),
                            f"the contact takes {fmt(lost, 4)} J out of the motion, and (1 - e^2) of the closing "
                            f"kinetic energy is {fmt(want, 4)} J ({100 * lerr:.2g}% apart)", {"lost_J": lost}))
        checks.append(energy_check(run))
    checks += [x for x in (contact_check(run, 2 * info["radii"][0]), warnings_check(run)) if x]

    dur = h["last_s"] - h["first_s"]
    limit = info["limit"]
    weight = info["m1"] * T.G
    if fixed:
        head = (f"Dropped from {fmt(info['drop'] * 1000)} mm, the part lands at {fmt(v1)} m/s and pushes on the floor "
                f"with up to {fmt(h['peak_N'])} N, {fmt(h['peak_N'] / weight)} times its weight, for "
                f"{fmt(dur * 1000, 3)} ms, then bounces back at {fmt(a1)} m/s." if info["drop"] else
                f"The part hits at {fmt(v1)} m/s with up to {fmt(h['peak_N'])} N for {fmt(dur * 1000, 3)} ms and comes "
                f"back at {fmt(a1)} m/s.")
    else:
        head = (f"The parts hit with up to {fmt(h['peak_N'])} N for {fmt(dur * 1000, 3)} ms. Afterward the "
                f"{fmt(info['m1'] * 1000)} g part moves at {fmt(a1)} m/s and the {fmt(info['m2'] * 1000)} g part at "
                f"{fmt(a2)} m/s.")
    verdict = "ok"
    fixes = []
    if limit:
        if h["peak_N"] > limit:
            verdict = "fails"
            head += f" That is more than the {fmt(limit)} N limit."
        elif h["peak_N"] > 0.8 * limit:
            verdict = "marginal"
            head += f" That is {100 * h['peak_N'] / limit:.0f}% of the {fmt(limit)} N limit."
        else:
            head += f" That is well inside the {fmt(limit)} N limit."
        if verdict != "ok":
            ks = (0.8 * limit / h["peak_N"]) ** 2
            vs = v1 * (0.8 * limit / h["peak_N"]) ** (5 / 6)
            fixes.append({"kind": "pad", "title": f"Add a pad about {fmt(ks)} times as stiff as the bare contact",
                          "summary": "The peak force grows with the square root of the contact stiffness, so this "
                                     "brings it to 80% of the limit, as rubber or a sprung stop would.", "auto": False})
            fixes.append({"kind": "speed", "title": f"Bring the impact speed down to {fmt(vs)} m/s"
                          + (f", a drop of {fmt(vs * vs / (2 * T.G) * 1000)} mm" if fixed and info["drop"] else ""),
                          "summary": "Hertz contact force grows with speed to the power 1.2, so this brings the peak "
                                     "to 80% of the limit.", "auto": False})
    extra = [("medium", "the peak force rests on the contact stiffness, set from Hertz theory for a round contact "
                        f"of {fmt(info['radii'][0] * 1000)} mm, and a flat face, a rubber foot or a flexible mount "
                        "hits softer" if not info["k_given"] else
                        "the peak force rests on the contact stiffness given")]
    if not info["e_given"]:
        extra.append(("medium", f"restitution not given, so Wobbl took {fmt(e)}, typical of "
                                f"{'metal on metal' if e > 0.5 else 'plastic or wood'}"))
    img.mkdir(parents=True, exist_ok=True)
    t0 = h["first_s"]
    win = (run.t > t0 - dur) & (run.t < h["last_s"] + dur)
    images = {"force": plots.series(img / "force.png", (run.t[win] - t0) * 1000, {"contact force": run.peak_force[win]},
                                    "N", {"limit": limit} if limit else None, xlabel="ms from first touch"),
              "frames": plots.strobe(img / "frames.png", run, (("x", "z"),))}
    events = [event("first touch", h["first_s"]), event("peak force", h["peak_s"], f"{fmt(h['peak_N'])} N"),
              event("apart", h["last_s"])]
    assumptions = [
        f"{mat1.name} part of {fmt(info['m1'] * 1000)} g" + ("" if fixed else f" hitting a {mat2.name} part of "
                                                               f"{fmt(info['m2'] * 1000)} g")
        + (f" landing on {mat2.name}" if fixed else "") + f", meeting on a round contact of "
        f"{fmt(info['radii'][0] * 1000)} mm radius, head on and without friction.",
        f"The contact is a linear spring and a dashpot that can only push, tuned to Hertz's peak force and to a "
        f"restitution of {fmt(e)}.",
        "Both parts are rigid apart from the contact, so no energy goes into ringing.",
    ]
    if fixed and info["drop"]:
        assumptions.append("The fall itself is not simulated. The part starts just above the floor at the speed of a "
                           "fall from that height without air.")
    values = {"peak_force_N": h["peak_N"], "contact_ms": dur * 1000, "depth_mm": h["depth_m"] * 1000,
              "speeds_after_mps": [a1, a2], "restitution": e_sim, "hertz_peak_N": hz["force_N"],
              "impact_speed_mps": v1}
    return Answer(verdict, head, [("peak force", f"{fmt(h['peak_N'])} N"), ("contact", f"{fmt(dur * 1000, 3)} ms"),
                                  ("speed in", f"{fmt(vrel)} m/s"), ("bounce", f"{fmt(abs(e_sim), 3)}")],
                  checks, assumptions, values, fixes, events,
                  [["force", "Contact force through the impact"], ["frames", "The parts at several instants"]],
                  images, run, extra, model=model_info(m, run))


def tip(b, img: Path, say) -> Answer:
    info, m = b.info, b.mech
    hand = info["hand"]
    say("solve", "loading the stand")
    run = simulate(m, samples=1500, frames=7)
    say("verify", "again at half the time step and with a stiffer floor")
    run2 = simulate(m, timestep=m.timestep / 2, samples=1500, frames=0)
    run3 = simulate(with_(m, contact={"timeconst": 0.0025, "dampratio": 1.0}), samples=1500, frames=0)
    tilt = run.free["stand_free"]["tilt"]
    tipped = float(np.max(tilt)) > 10
    xy = run.free["stand_free"]["xy"]
    slid = float(np.max(np.linalg.norm(xy - xy[0], axis=1))) > 0.005
    mass, edge, fh, acc = hand["mass_kg"], info["edge"], info["force_height"], info["accel"]
    ramp = 0.5
    force = info["force"] * np.minimum(1.0, run.t / ramp)
    lean = (force * fh + mass * acc * hand["com_h_m"]) / (mass * T.G)       # how far the moments move the reaction
    expect = xy[:, 0] + lean
    # while it is flat and the push has not yet carried the floor's reaction to the edge, where it stops moving
    ok = (tilt < 0.3) & np.isfinite(run.cop[:, 0]) & (run.t > 0.05) & (hand["com_x_m"] + lean < 0.95 * edge)
    checks = []
    if ok.sum() >= 3:
        # the reaction rocks a little on the stiff contact springs, so compare the straight line it follows as the
        # push grows (or its mean when nothing changes) with the statics
        got = run.cop[ok, 0] - xy[ok, 0]
        if np.ptp(lean[ok]) > 1e-9:
            got = np.polyval(np.polyfit(lean[ok], got, 1), lean[ok])
        else:
            got = np.full(len(got), float(np.mean(got)))
        err = float(np.max(np.abs(got - lean[ok]))) / edge
        checks.append(Check("hand statics", "pass" if err < 0.02 else ("warn" if err < 0.05 else "fail"),
                            "while the base is flat, the floor's push sits where the moments of the weight, the push "
                            f"and the lean say it must, to {100 * err:.2g}% of the half width",
                            {"error_share": err}))
    tips_by_hand = hand["restoring_Nm"] <= 0 or hand["margin"] < 1
    if tips_by_hand == tipped:
        checks.append(Check("tipping statics", "pass", f"the moments about the front edge give a margin of "
                                                       f"{_margin(hand)} and the run {'tips' if tipped else 'stays up'}"
                                                       " as that says", {"margin": hand["margin"]}))
    else:
        checks.append(Check("tipping statics", "warn",
                            f"the moments about the front edge give a margin of {_margin(hand)} but the run "
                            f"{'tips' if tipped else 'stays up'}" + (" because it slid" if slid else ""),
                            {"margin": hand["margin"]}))
    t3 = run3.free["stand_free"]["tilt"]
    flat = tilt < 1.0
    sink = float(np.max(run.depth[flat])) * 1000 if flat.any() else 0.0      # before it goes over, if it does
    checks.append(Check("contact settings", "pass" if (float(np.max(t3)) > 10) == tipped else "warn",
                        f"with a floor twice as stiff it {'tips' if np.max(t3) > 10 else 'stays up'} too, and the "
                        f"base sinks at most {fmt(sink)} mm into the floor", {"depth_mm": sink}))
    t2 = run2.free["stand_free"]["tilt"]
    if tipped and float(np.max(t2)) > 10:
        checks.append(step_check(float(run.t[np.argmax(tilt > 10)]), float(run2.t[np.argmax(t2 > 10)]),
                                 "time to tip past 10 degrees", " s"))
    else:
        checks.append(step_check(float(np.max(tilt)) + 1.0, float(np.max(t2)) + 1.0, "largest tilt plus 1 degree",
                                 " deg"))
    checks.append(energy_check(run))
    checks += [x for x in (warnings_check(run),) if x]
    events = []
    if info["force"]:
        events.append(event("push at full strength", ramp, f"{fmt(info['force'])} N"))
    if tipped:
        events.append(event("tips past 10 degrees", run.t[int(np.argmax(tilt > 10))]))
    if slid:
        events.append(event("slides", run.t[int(np.argmax(np.linalg.norm(xy - xy[0], axis=1) > 0.005))]))
    events.sort(key=lambda e: e["t_s"])
    inside = (edge - hand["com_x_m"]) * 1000
    if tipped:
        verdict = "fails"
        head = (f"It tips over at {fmt(float(run.t[int(np.argmax(tilt > 10))]))} s, because "
                f"{fmt(hand['overturning_Nm'])} N m tips it over the front edge against "
                f"{fmt(max(hand['restoring_Nm'], 0))} N m from its weight." if hand["overturning_Nm"] > 0 else
                f"It tips over at once, since its center of mass sits {fmt(-inside)} mm beyond the front edge.")
    elif hand["margin"] < 1.5 or slid:
        verdict = "marginal"
        head = (f"It stays up with little margin, {fmt(hand['restoring_Nm'])} N m from its weight against "
                f"{fmt(hand['overturning_Nm'])} N m tipping it, a margin of {_margin(hand)}."
                + (" It slides along the floor." if slid else ""))
    else:
        verdict = "ok"
        head = (f"It stays up with a margin of {_margin(hand)}, {fmt(hand['restoring_Nm'])} N m from its weight "
                f"against {fmt(hand['overturning_Nm'])} N m tipping it." if hand["overturning_Nm"] > 0 else
                f"It stays up. Its center of mass sits {fmt(inside)} mm inside the front edge and nothing pushes it.")
    if math.isfinite(hand["critical_force_N"]) and hand["critical_force_N"] > 0:
        head += f" A push of {fmt(hand['critical_force_N'])} N at {fmt(fh * 1000)} mm tips it."
    fixes = _tip_fixes(b, hand) if verdict != "ok" else []
    img.mkdir(parents=True, exist_ok=True)
    images = {"tilt": plots.series(img / "tilt.png", run.t, {"tilt": tilt}, "degrees",
                                   {"10 degrees": 10.0} if tipped else None),
              "frames": plots.strobe(img / "frames.png", run, (("x", "z"),))}
    bw, bd = info["base"]
    assumptions = [f"A rigid stand on a {fmt(bw * 1000)} x {fmt(bd * 1000)} mm base, {fmt(mass * 1000)} g in all, with "
                   f"its center of mass {fmt(hand['com_x_m'] * 1000)} mm toward the front and "
                   f"{fmt(hand['com_h_m'] * 1000)} mm up, on a floor with friction {fmt(info['friction'])}.",
                   (f"The push of {fmt(info['force'])} N acts toward the front at {fmt(fh * 1000)} mm, growing over "
                    f"{fmt(ramp)} s." if info["force"] else "Nothing pushes it.") +
                   (f" It brakes at {fmt(acc)} m/s2, which leans it forward." if acc else "")]
    values = {"margin": hand["margin"], "critical_force_N": hand["critical_force_N"], "tipped": tipped,
              "slid": slid, "max_tilt_deg": float(np.max(tilt)), "com_m": [hand["com_x_m"], hand["com_h_m"]]}
    return Answer(verdict, head, [("margin", _margin(hand)), ("tips at", f"{fmt(hand['critical_force_N'])} N"),
                                  ("max tilt", f"{fmt(float(np.max(tilt)))} deg"), ("mass", f"{fmt(mass * 1000)} g")],
                  checks, assumptions, values, fixes, events,
                  [["tilt", "Tilt of the base over time"], ["frames", "The stand at several instants, from the side"]],
                  images, run, model=model_info(m, run))


def _margin(hand: dict) -> str:
    if hand["restoring_Nm"] <= 0:
        return "0"
    return "unlimited" if not math.isfinite(hand["margin"]) else fmt(hand["margin"])


def _tip_fixes(b, hand: dict) -> list[dict]:
    from .templates import tip as build_tip

    info = b.info
    bw = info["base"][0]
    g, acc = T.G, info["accel"]
    m_r, m_o, mass = hand["restoring_Nm"], hand["overturning_Nm"], hand["mass_kg"]
    out = []
    if m_o > 0:
        need = (1.5 * m_o - m_r) / max(g * (bw - 0.01) - 1.5 * acc * 0.01, 1e-9)
    else:
        need = mass * (hand["com_x_m"] - 0.8 * info["edge"]) / (0.8 * info["edge"] + bw / 2 - 0.01)
    need = max(math.ceil(need * 20) / 20, 0.05)
    run = simulate(build_tip({**b.params, "counterweight": f"{need * 1000:g} g"}).mech, frames=0)
    stays = float(np.max(run.free["stand_free"]["tilt"])) < 10
    out.append({"kind": "counterweight", "title": f"Add {fmt(need * 1000)} g of ballast low at the back edge",
                "summary": "That gives the weight half again the moment that tips it. Running it again with the "
                           f"ballast, it {'stays up' if stays else 'still tips, so widen the base as well'}.",
                "auto": False, "after": {"counterweight_kg": need, "stays": stays}})
    if m_o > 0:
        w = 2 * (1.5 * m_o / (mass * g) + hand["com_x_m"])
        out.append({"kind": "base", "title": f"Widen the base to {fmt(w * 1000)} mm front to back",
                    "summary": "With the same weight that keeps the moments about the front edge at a margin of 1.5.",
                    "auto": False, "after": {"base_mm": w * 1000}})
    return out
