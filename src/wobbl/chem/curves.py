"""A simulated motor's thrust curve for other tools: a CSV, a RASP .eng file and the motion domain's Motor.

RASP files are what thrustcurve.org serves and OpenRocket and Wobbl's motion domain read: a header line with the
designation, diameter and length in mm, delays, propellant and loaded mass in kg and the maker, then time and thrust
pairs that end at zero. The curve is thinned to the points that keep its shape, adding at each pass the point
farthest from the straight lines through those kept so far.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .ballistics import Burn, converge
from .spec import MotorSpec


def write_curves(spec: MotorSpec, b: Burn, out: Path) -> dict:
    """thrust.csv (time, thrust, pressure, mass flow, propellant left) and motor.eng, next to the report."""
    keep = np.unique(np.concatenate([np.linspace(0, len(b.t) - 1, min(len(b.t), 400)).astype(int),
                                     [int(np.argmax(b.P)), int(np.argmax(b.F))]]))
    burned = np.concatenate([[0.0], np.cumsum(0.5 * (b.mdot[1:] + b.mdot[:-1]) * np.diff(b.t))])
    left = np.clip(b.mass - burned * b.mass / max(burned[-1], 1e-12), 0.0, None)
    rows = ["time_s,thrust_N,pressure_Pa,mass_flow_kg_s,propellant_kg"]
    rows += [f"{b.t[i]:.5g},{b.F[i]:.6g},{b.P[i]:.6g},{b.mdot[i]:.5g},{left[i]:.5g}" for i in keep]
    (out / "thrust.csv").write_text("\n".join(rows) + "\n")
    (out / "motor.eng").write_text(rasp(spec, b))
    return {"thrust_csv": "thrust.csv", "eng": "motor.eng"}     # beside outcome.json, so results can move


def loaded_mass(spec: MotorSpec, b: Burn) -> float:
    """Propellant, casing tube and the closures and nozzle the spec names, kg."""
    c = spec.casing
    if c is None:
        return b.mass
    from ..materials import resolve_material
    tube = resolve_material(c.material).density * math.pi / 4 * (c.outer ** 2 - c.inner ** 2) * \
        (c.length or sum(g.L for g in spec.motor.grains))
    return b.mass + tube + c.hardware


def _length(spec: MotorSpec) -> float:
    c = spec.casing
    return c.length if c and c.length else sum(g.L for g in spec.motor.grains)


def curve(b: Burn, points: int) -> list[tuple[float, float]]:
    """The burn's thrust as at most ``points`` (s, N) pairs from ignition, closed with a zero."""
    on = np.nonzero(b.F > 0.001 * b.peak_thrust)[0]
    t, F = b.t[on[0]:on[-1] + 1], b.F[on[0]:on[-1] + 1]
    pts = [(float(t[i]), float(F[i])) for i in _simplify(t, F, points - 1) if t[i] > 0]
    pts.append((pts[-1][0] + max(b.dt, 1e-3), 0.0))
    return pts


def rasp(spec: MotorSpec, b: Burn) -> str:
    c = spec.casing
    dia = (c.outer if c else max(g.D for g in spec.motor.grains)) * 1000
    name = "".join(ch for ch in b.designation() if ch.isalnum()) or "motor"
    lines = [f"; {spec.name}, simulated by Wobbl from its grain, propellant and nozzle design",
             f"{name} {dia:.0f} {_length(spec) * 1000:.0f} P {b.mass:.4f} {loaded_mass(spec, b):.4f} Wobbl"]
    lines += [f"   {ti:.4f} {fi:.2f}" for ti, fi in curve(b, 60)]
    return "\n".join(lines) + "\n;\n"


def _simplify(t: np.ndarray, F: np.ndarray, n: int) -> list[int]:
    keep = {0, len(t) - 1, int(np.argmax(F))}
    scale = max(F.max(), 1e-12)
    while len(keep) < min(n, len(t)):
        ks = sorted(keep)
        err = np.abs(F - np.interp(t, t[ks], F[ks])) / scale
        j = int(np.argmax(err))
        if err[j] < 1e-4:
            break
        keep.add(j)
    return sorted(keep)


def motion_motor(path: str | Path, name: str = ""):
    """A motor design file as the motion domain's Motor: simulated here, handed over as its thrust curve, its
    propellant and loaded mass and its length."""
    from ..motion.spec import Motor as MotionMotor
    from .spec import read_spec

    spec = read_spec(str(path))
    b, _ = converge(spec.motor)
    mo = MotionMotor(name or f"{spec.name} ({b.designation()})", curve(b, 120), b.mass,
                     max(loaded_mass(spec, b), b.mass * 1.0001))
    mo.length = _length(spec)
    return mo
