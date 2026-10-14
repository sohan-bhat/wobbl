"""Mechanism validation cases. Each runs a whole mechanism study, MuJoCo, checks, report and all, and compares one
number with the exact answer for its setup, worked here independently of the run."""
from __future__ import annotations

import math
from pathlib import Path


G = 9.80665


def _study(template: str, params: dict, out: Path):
    from ..mechanism.study import run_mechanism

    return run_mechanism(template, params, out)


def _info(o, **extra) -> dict:
    return {"confidence": o.confidence, "time_s": round(o.t_total, 1), **extra}


def pendulum_period(out: Path):
    """A 1 m pendulum released from rest at 120 degrees: T = 4 sqrt(L / g) K(sin(60 degrees))."""
    from scipy.special import ellipk

    o = _study("pendulum", {"length": "1 m", "angle": "120 deg", "mass": "1 kg", "periods": 2}, out)
    exact = 4 * math.sqrt(1 / G) * float(ellipk(math.sin(math.radians(60)) ** 2))
    return o.values["period_s"], exact, _info(o, small_angle_s=round(2 * math.pi * math.sqrt(1 / G), 4))


def fourbar_sweep(out: Path):
    """A crank rocker of 120, 30, 125 and 45 mm turned once. The rocker turns back where the crank and coupler
    line up, stretched out (b + a from the crank pivot) and folded (b - a), so the law of cosines in the triangle
    of ground, rocker and that line gives its swing. The run also reports how far it strays from Freudenstein's
    equation at every crank angle."""
    o = _study("fourbar", {"ground": "120 mm", "crank": "30 mm", "coupler": "125 mm", "rocker": "45 mm",
                           "speed": "30 rpm", "turns": 1}, out)
    d, a, b, c = 0.120, 0.030, 0.125, 0.045

    def at_rocker(r):
        return math.acos((d * d + c * c - r * r) / (2 * d * c))

    exact = math.degrees(at_rocker(b + a) - at_rocker(b - a))
    return o.values["sweep_deg"], exact, _info(o, max_position_error_deg=round(o.values["freudenstein_error_deg"], 6))


def arm_statics(out: Path):
    """A two link arm holding 300 g at its pick point: the shoulder torque is the moment of every weight about the
    shoulder, with the links' weights at their middles and the elbow servo, gripper and payload at their ends."""
    from ..mechanism.motors import get_motor

    p = {"links": "150,120", "link_mass": "80,50", "servo": "xm430", "payload": "300 g", "gripper": "40 g",
         "pick": "200,0,20", "place": "0,200,20", "base_height": "60 mm"}
    o = _study("arm", p, out)
    l1, l2, m1, m2, mp, mg = 0.150, 0.120, 0.080, 0.050, 0.300, 0.040
    ms = get_motor("xm430").mass
    # elbow up inverse kinematics for a tip at 200 mm out and 40 mm below the shoulder
    x, z = 0.200, 0.020 - 0.060
    c2 = (x * x + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    t2 = -math.acos(c2)
    t1 = math.atan2(z, x) - math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
    tau1 = G * (math.cos(t1) * (m1 * l1 / 2 + (ms + m2 + mg + mp) * l1) +
                math.cos(t1 + t2) * (m2 * l2 / 2 + (mg + mp) * l2))
    return o.values["joints"]["shoulder"]["hold_Nm"], tau1, _info(o)


def gear_output(out: Path):
    """An FA-130 through 12:36 and 10:40 at 97% a mesh against 5 mN m: the motor line balanced through the
    ratio and the efficiency, w_out = w0 (1 - T / (N eta^2 Ts)) / N."""
    from ..mechanism.motors import get_motor

    o = _study("gears", {"teeth": "12:36,10:40", "efficiency": 0.97, "motor": "fa-130", "load": "0.005 N*m"}, out)
    m = get_motor("fa-130")
    n, eta = 3 * 4, 0.97 ** 2
    w_out = m.no_load_speed * (1 - 0.005 / (n * eta * m.stall_torque)) / n
    return o.values["output_rpm"], w_out * 60 / (2 * math.pi), _info(o, ratio=round(o.values["ratio_simulated"], 6))


def _collision(out: Path, e: float):
    o = _study("impact", {"mass": "1 kg", "target_mass": "3 kg", "speed": "1 m/s", "restitution": e,
                          "material": "mild-steel"}, out)
    # momentum m1 v1 = m1 u1 + m2 u2 and restitution u2 - u1 = e v1 give u2 = m1 (1 + e) v1 / (m1 + m2)
    return o.values["speeds_after_mps"][1], 1.0 * (1 + e) * 1.0 / 4.0, _info(o, peak_force_N=round(
        o.values["peak_force_N"], 1))


def collision_elastic(out: Path):
    """A 1 kg steel ball at 1 m/s hits a 3 kg one at rest, perfectly elastic."""
    return _collision(out, 1.0)


def collision_restitution(out: Path):
    """The same balls with a restitution of 0.5."""
    return _collision(out, 0.5)
