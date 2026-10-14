"""Turn a mechanism run into an answer: the pendulum, and any spec or model file. The arm, the linkage, the gear
train, impacts and tipping each have their own module, registered in ANALYSES at the bottom."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .quantities import fmt_torque
from .sim import simulate
from .spec import Mechanism
from .study import (Answer, constraint_check, contact_check, energy_check, event, model_info, step_check, typical_size,
                    warnings_check)


def crossings(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Times where v changes sign, placed by straight line between the samples around each change."""
    s = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    return t[s] + (t[s + 1] - t[s]) * v[s] / (v[s] - v[s + 1])


def period(run) -> float:
    """A swing's period from the turning points, where the speed changes sign, two of them per period."""
    tc = crossings(run.t, run.v[:, 0])
    if len(tc) < 2:
        return float("nan")
    return float(2 * np.mean(np.diff(tc)))


def pendulum(b, img: Path, say) -> Answer:
    m, info = b.mech, b.info
    say("solve", "swinging the pendulum")
    run = simulate(m, samples=8000, frames=7)
    T1 = period(run)
    say("verify", "again at half the time step")
    run2 = simulate(m, timestep=m.timestep / 2, samples=8000, frames=0)
    T2 = period(run2)
    exact, small = info["period_s"], info["small_angle_s"]
    deg = math.degrees(info["angle_rad"])
    err = abs(T1 - exact) / exact
    checks = [
        Check("hand calculation", "pass" if err < 1e-3 else ("warn" if err < 1e-2 else "fail"),
              f"the elliptic integral 4 sqrt(L/g) K(sin(a/2)) gives {fmt(exact, 6)} s for a {fmt(info['length_m'])} m "
              f"pendulum released at {fmt(deg)} degrees, and MuJoCo swings it in {fmt(T1, 6)} s "
              f"({100 * err:.2g}% apart)", {"exact_s": exact, "mujoco_s": T1}),
        step_check(T1, T2, "period", " s"),
        energy_check(run, passive=True),
    ]
    checks = [c for c in checks + [warnings_check(run)] if c is not None]
    head = (f"The pendulum swings with a period of {fmt(T1, 4)} s from {fmt(deg)} degrees, "
            f"{100 * (T1 / small - 1):.3g}% longer than the small angle formula's {fmt(small, 4)} s.")
    img.mkdir(parents=True, exist_ok=True)
    images = {"swing": plots.series(img / "swing.png", run.t, {"angle": np.degrees(run.q[:, 0])}, "degrees"),
              "frames": plots.strobe(img / "frames.png", run, (("x", "z"),))}
    events = [event("release", 0.0, f"at rest, {fmt(deg)} degrees")]
    events += [event("turning point", t) for t in crossings(run.t, run.v[:, 0])[:4]]
    return Answer("ok", head, [("period", f"{fmt(T1, 4)} s"), ("small angle", f"{fmt(small, 4)} s"),
                               ("release", f"{fmt(deg)} deg"), ("length", f"{fmt(info['length_m'] * 1000)} mm")],
                  checks, [f"A point mass on a massless rod {fmt(info['length_m'] * 1000)} mm long, released from "
                           "rest, with no air drag or pivot friction."],
                  {"period_s": T1, "exact_period_s": exact, "small_angle_s": small},
                  events=events, figures=[["swing", "Angle over time"], ["frames", "The swing at several instants"]],
                  images=images, run=run, model=model_info(m, run))


def generic(m: Mechanism, img: Path, say) -> Answer:
    """A spec or model file: run it as written, find what each drive needs against its motor, and report contacts
    between parts, anything that tips and the energy account."""
    driven = [d for d in m.drives if d.motor is not None]
    say("solve", "running the mechanism")
    run = simulate(m, frames=9)
    ideal = simulate(m, ideal=True, frames=0) if driven else None
    say("verify", "again at half the time step")
    run2 = simulate(m, timestep=run.dt / 2, frames=0)
    checks, events, fixes = [], [], []
    passive = not m.drives and not m.loads and not any(j.damping or j.friction for j in m.joints)
    checks.append(energy_check(run, passive=passive))
    usage = {}
    for d in driven:
        c = ideal.col(d.joint)
        mot = d.at_joint
        u = mot.usage(ideal.tau_need[:, c], ideal.v[:, c])
        k = int(np.argmax(u))
        rms = float(np.sqrt(np.mean(ideal.tau_need[:, c] ** 2)))
        usage[d.joint] = {"peak_usage": float(u[k]), "at_s": float(ideal.t[k]),
                          "torque_Nm": float(ideal.tau_need[k, c]),
                          "speed_rpm": float(ideal.v[k, c] * 60 / (2 * math.pi)), "rms_Nm": rms, "rated_Nm": mot.rated,
                          "motor": mot.name}
        if u[k] > 1:
            events.append(event(f"{d.joint} stalls", ideal.t[k], f"needs {fmt_torque(abs(ideal.tau_need[k, c]))}"))
    hits = {k: v for k, v in run.contacts.items() if not v["support"]}
    for k, v in hits.items():
        events.append(event(f"{k} touch", v["first_s"], f"peak {fmt(v['peak_N'])} N"))
    tipped = {k: float(np.max(v["tilt"])) for k, v in run.free.items() if np.max(v["tilt"]) > 10}
    for k in tipped:
        i = int(np.argmax(run.free[k]["tilt"] > 10))
        events.append(event(f"{run.free[k]['body']} tips", run.t[i]))
    if driven:
        j = max(usage, key=lambda x: usage[x]["peak_usage"])
        c1, c2 = run.col(j), run2.col(j)
        checks.append(step_check(float(np.max(np.abs(run.tau[:, c1]))), float(np.max(np.abs(run2.tau[:, c2]))),
                                 f"peak {j} torque", " N m"))
    elif run.joints:
        a, b2 = float(np.max(np.abs(run.q))), float(np.max(np.abs(run2.q)))
        checks.append(step_check(a, b2, "largest joint excursion", " rad"))
    elif run.free:
        k = next(iter(run.free))
        checks.append(step_check(float(run.free[k]["z"][-1]) + 1.0, float(run2.free[k]["z"][-1]) + 1.0,
                                 "final height plus 1 m", " m"))
    size = typical_size(run)
    checks += [c for c in (contact_check(run, size), constraint_check(run, size, "mm"), warnings_check(run)) if c]
    stalls = [j for j, u in usage.items() if u["peak_usage"] > 1]
    hot = [j for j, u in usage.items() if u["rms_Nm"] > u["rated_Nm"]]
    worst = max((u["peak_usage"] for u in usage.values()), default=0.0)
    if stalls or hot or hits or tipped:
        verdict = "fails"
        parts = []
        if stalls:
            u = usage[stalls[0]]
            parts.append(f"The {stalls[0]} drive stalls because it needs {fmt_torque(abs(u['torque_Nm']))} at "
                         f"{fmt(abs(u['speed_rpm']))} rpm, {u['peak_usage']:.2g}x what its motor gives there.")
        if hot:
            u = usage[hot[0]]
            parts.append(f"The {hot[0]} drive runs hot, at {fmt(u['rms_Nm'])} N m RMS against about "
                         f"{fmt(u['rated_Nm'])} N m continuous.")
        if hits:
            k, v = next(iter(hits.items()))
            parts.append(f"{k.replace(' / ', ' and ').capitalize()} touch at {fmt(v['first_s'])} s with up to "
                         f"{fmt(v['peak_N'])} N.")
        if tipped:
            parts.append(f"{next(iter(run.free.values()))['body'].capitalize()} tips over.")
        head = " ".join(parts)
    elif worst > 0.8:
        verdict, head = "marginal", f"Every drive keeps up, the busiest at {100 * worst:.0f}% of what its motor gives."
    else:
        verdict = "ok"
        head = (f"Every drive keeps up, the busiest at {100 * worst:.0f}% of its motor." if usage else
                f"The mechanism runs {fmt(run.t[-1])} s with no parts touching and nothing tipping.")
    for j in stalls + [h for h in hot if h not in stalls]:
        from .motors import stronger

        u = usage[j]
        d = m.drive(j)
        pick = stronger(d.at_joint, abs(u["torque_Nm"]), abs(u["speed_rpm"]) * 2 * math.pi / 60, u["rms_Nm"])
        need = 1.25 * max(abs(u["torque_Nm"]), u["rms_Nm"] * 3)
        fixes.append({"kind": "motor", "title": f"Give {j} a motor with at least {fmt_torque(need)} of stall torque",
                      "summary": (f"The {pick.name} clears it. " if pick else "") +
                                 "A gear or belt reduction does the same while the joint stays slow enough.",
                      "auto": False})
    metrics = [(f"{j} use", f"{100 * u['peak_usage']:.0f}%") for j, u in list(usage.items())[:3]]
    metrics.append(("contacts", str(len(hits))))
    img.mkdir(parents=True, exist_ok=True)
    images = {"frames": plots.strobe(img / "frames.png", run, (("x", "z"), ("x", "y")))}
    figs = [["frames", "The mechanism at several instants"]]
    if driven:
        images["torque_speed"] = plots.torque_speed(img / "torque_speed.png", [
            {"name": d.joint, "motor": d.at_joint, "w": ideal.v[:, ideal.col(d.joint)],
             "tau": ideal.tau_need[:, ideal.col(d.joint)]} for d in driven])
        figs.insert(0, ["torque_speed", "What each drive needs against its motor's line"])
    assumptions = [f"Model {'from ' + Path(m.model_file).name if m.model_file else 'built from the spec'}, run for "
                   f"{fmt(run.t[-1])} s at a {fmt(run.dt * 1000)} ms step."]
    if m.drives:
        assumptions.append("Servos follow the move exactly while their motor allows, by computed torque.")
    return Answer(verdict, head, metrics, checks, assumptions, {"drives": usage, "contacts": hits},
                  fixes, events, figs, images, run, model=model_info(m, run))


def with_(m: Mechanism, **changes) -> Mechanism:
    """A copy of a mechanism with some fields changed, leaving the original alone."""
    return dataclasses.replace(m, **changes)


def _registry():
    from . import arm, gears, impacts, linkage

    return {"pendulum": pendulum, "arm": arm.answer, "fourbar": linkage.answer, "gears": gears.answer,
            "impact": impacts.impact, "tip": impacts.tip}


class _Lazy(dict):
    def __missing__(self, key):
        self.update(_registry())
        return dict.__getitem__(self, key)


ANALYSES = _Lazy()
