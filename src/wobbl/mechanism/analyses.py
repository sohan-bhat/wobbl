"""The pendulum template's answer and the registry of template answers. The arm, the linkage, the gear train,
impacts and tipping each have their own module; any model file or spec goes through general.py instead."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .sim import simulate
from .spec import Mechanism
from .study import Answer, energy_check, event, model_info, step_check, warnings_check


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


def with_(m: Mechanism, **changes) -> Mechanism:
    """A copy of a mechanism with some fields changed, leaving the original alone."""
    return dataclasses.replace(m, **changes)


def _registry():
    from . import arm, drive, gears, impacts, linkage

    return {"pendulum": pendulum, "arm": arm.answer, "fourbar": linkage.answer, "slider": drive.slider_answer,
            "gears": gears.answer,
            "impact": impacts.impact, "tip": impacts.tip}


class _Lazy(dict):
    def __missing__(self, key):
        self.update(_registry())
        return dict.__getitem__(self, key)


ANALYSES = _Lazy()
