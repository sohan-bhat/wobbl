"""Closed form answers the mechanism checks compare MuJoCo against. All SI, angles in radians.

None of these call MuJoCo. Each is the textbook result for the case a run reduces to, so agreement between the two
says the model was built and solved right.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.80665


# --- pendulum -----------------------------------------------------------------------------------------------
def pendulum_period(length: float, amplitude: float, g: float = G) -> float:
    """Exact period of a point mass on a massless rod released from rest at ``amplitude``:
    4 sqrt(L/g) K(sin(amplitude/2)), with K the complete elliptic integral of the first kind."""
    from scipy.special import ellipk

    return 4 * math.sqrt(length / g) * float(ellipk(math.sin(amplitude / 2) ** 2))


# --- four bar linkage ---------------------------------------------------------------------------------------
def grashof(ground: float, crank: float, coupler: float, rocker: float) -> tuple[str, bool]:
    """Grashof class, and whether the crank turns all the way round."""
    s, *mid, l = sorted([ground, crank, coupler, rocker])
    if s + l > sum(mid) + 1e-12:
        return "non Grashof (no link turns fully)", False
    if abs(s + l - sum(mid)) < 1e-9 * l:
        return "change point (the links can line up and flip)", crank == s
    if crank == s:
        return "crank rocker", True
    if ground == s:
        return "double crank", True
    return "double rocker", False


def freudenstein(ground: float, crank: float, coupler: float, rocker: float, theta2, open_branch: bool = True):
    """Rocker and coupler angles from the crank angle, both measured from the ground line, by Freudenstein's
    equation K1 cos(t4) - K2 cos(t2) + K3 = cos(t2 - t4) solved as a quadratic in tan(t4/2). Returns (t4, t3) and
    nan where the loop cannot close."""
    d, a, b, c = ground, crank, coupler, rocker
    t2 = np.asarray(theta2, float)
    k1, k2, k3 = d / a, d / c, (a * a - b * b + c * c + d * d) / (2 * a * c)
    A = np.cos(t2) - k1 - k2 * np.cos(t2) + k3
    B = -2 * np.sin(t2)
    C = k1 - (k2 + 1) * np.cos(t2) + k3
    disc = B * B - 4 * A * C
    root = np.sqrt(np.where(disc >= 0, disc, np.nan))
    sign = -1.0 if open_branch else 1.0
    t4 = 2 * np.arctan2(-B + sign * root, 2 * A)
    # the coupler angle from the two pin positions
    ax, ay = a * np.cos(t2), a * np.sin(t2)
    bx, by = d + c * np.cos(t4), c * np.sin(t4)
    t3 = np.arctan2(by - ay, bx - ax)
    return t4, t3


def velocity_ratio(crank: float, rocker: float, t2, t3, t4):
    """Rocker speed over crank speed, a sin(t3 - t2) / (c sin(t3 - t4)), from the loop's velocity equations."""
    return crank * np.sin(np.asarray(t3) - t2) / (rocker * np.sin(np.asarray(t3) - t4))


def transmission_angle(t3, t4):
    """The angle between coupler and rocker, folded into 0 to 180 degrees (90 is ideal)."""
    mu = np.abs(np.mod(np.asarray(t4) - t3 + math.pi, 2 * math.pi) - math.pi)
    return mu


# --- slider crank -------------------------------------------------------------------------------------------
def slider_x(crank: float, rod: float, offset: float, theta):
    """Slider position along its line for a crank angle, with the line ``offset`` above the crank's pivot:
    r cos(t) + sqrt(l^2 - (r sin(t) - e)^2). Nan where the rod cannot reach."""
    t = np.asarray(theta, float)
    inside = rod * rod - (crank * np.sin(t) - offset) ** 2
    return crank * np.cos(t) + np.sqrt(np.where(inside >= 0, inside, np.nan))


def slider_ratio(crank: float, rod: float, offset: float, theta):
    """Slider speed over crank speed, dx/dt: -r sin(t) - r cos(t) (r sin(t) - e) / sqrt(l^2 - (r sin(t) - e)^2)."""
    t = np.asarray(theta, float)
    u = crank * np.sin(t) - offset
    root = np.sqrt(np.where(rod * rod - u * u >= 0, rod * rod - u * u, np.nan))
    return -crank * np.sin(t) - crank * np.cos(t) * u / root


# --- gear trains --------------------------------------------------------------------------------------------
def gear_train(stages: list[tuple[int, int]], efficiency: float) -> tuple[float, float]:
    """Overall ratio (input turns per output turn) and efficiency of meshes in series."""
    ratio = math.prod(b / a for a, b in stages)
    return ratio, efficiency ** len(stages)


def motor_steady(stall: float, no_load: float, ratio: float, eff: float, load: float) -> tuple[float, float]:
    """Motor speed and torque where a straight line motor driving through ``ratio`` at ``eff`` balances a load
    torque at the output. Motor torque is load / (ratio eff); speed follows the line."""
    tm = load / (ratio * eff)
    return no_load * (1 - tm / stall), tm


# --- impacts ------------------------------------------------------------------------------------------------
def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    return 1 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)


def hertz_impact(m_eff: float, v: float, E_eff: float, R_eff: float) -> dict:
    """Elastic Hertz impact of curved surfaces: largest compression, peak force and contact time."""
    k = 4 / 3 * E_eff * math.sqrt(R_eff)
    delta = (15 * m_eff * v * v / (16 * E_eff * math.sqrt(R_eff))) ** 0.4
    force = k * delta ** 1.5
    return {"delta_m": delta, "force_N": force, "time_s": 2.9432 * delta / v, "radius_m": math.sqrt(R_eff * delta)}


def dashpot_restitution(zeta: float) -> float:
    """Restitution of a linear spring and dashpot whose force can only push, so contact ends when the force
    reaches zero rather than when the bodies part (Schwager and Poschel, Granular Matter 9, 2007)."""
    if zeta <= 0:
        return 1.0
    if zeta < 1 / math.sqrt(2):
        w = math.sqrt(1 - zeta * zeta)
        return math.exp(-zeta / w * (math.pi - math.atan(2 * zeta * w / (1 - 2 * zeta * zeta))))
    if zeta < 1:
        w = math.sqrt(1 - zeta * zeta)
        return math.exp(-zeta / w * math.atan(2 * zeta * w / (2 * zeta * zeta - 1)))
    w = math.sqrt(zeta * zeta - 1)
    return math.exp(-zeta / w * math.log((zeta + w) / (zeta - w)))


def zeta_for_restitution(e: float) -> float:
    """Damping ratio of that contact for a target restitution, by bisection on ``dashpot_restitution``."""
    if e >= 1:
        return 0.0
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dashpot_restitution(mid) > e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def collision(m1: float, v1: float, m2: float, v2: float, e: float) -> tuple[float, float]:
    """Velocities after a head on collision, from momentum and the restitution coefficient."""
    p = m1 * v1 + m2 * v2
    return (p - m2 * e * (v1 - v2)) / (m1 + m2), (p + m1 * e * (v1 - v2)) / (m1 + m2)


def dashpot_peak(m_eff: float, v: float, w0: float, zeta: float) -> float:
    """Peak force of the linear spring and dashpot contact, from its exact motion sampled finely."""
    if zeta >= 1:
        zeta = 0.999
    wd = w0 * math.sqrt(1 - zeta * zeta)
    t = np.linspace(0, math.pi / wd, 4001)
    x = v / wd * np.exp(-zeta * w0 * t) * np.sin(wd * t)
    xd = v * np.exp(-zeta * w0 * t) * (np.cos(wd * t) - zeta * w0 / wd * np.sin(wd * t))
    f = m_eff * (w0 * w0 * x + 2 * zeta * w0 * xd)
    return float(np.max(f))


# --- tipping ------------------------------------------------------------------------------------------------
def tipping(masses: list[tuple[float, float, float]], edge: float, force: float = 0.0, force_height: float = 0.0,
            accel: float = 0.0, g: float = G) -> dict:
    """Moments about the edge a body tips over. ``masses`` are (kg, x, height) with x toward that edge.
    A push ``force`` toward the edge at ``force_height`` and a braking ``accel`` (the body decelerating, so its
    weight leans toward the edge) both tip it; weight inside the edge holds it."""
    m = sum(k for k, _, _ in masses)
    x = sum(k * xi for k, xi, _ in masses) / m
    h = sum(k * zi for k, _, zi in masses) / m
    restoring = m * g * (edge - x)
    overturning = force * force_height + m * accel * h
    crit_force = (restoring - m * accel * h) / force_height if force_height > 0 else math.inf
    return {"mass_kg": m, "com_x_m": x, "com_h_m": h, "restoring_Nm": restoring, "overturning_Nm": overturning,
            "margin": restoring / overturning if overturning > 0 else math.inf, "critical_force_N": crit_force}


# --- planar arms --------------------------------------------------------------------------------------------
def arm_statics(lengths: list[float], angles: list[float], masses: list[tuple[int, float, float]],
                g: float = G) -> list[float]:
    """Holding torque at each pitch joint of a planar arm, the moment of every weight beyond it about its axis.
    ``angles`` are relative joint angles (the first from horizontal, up positive) and ``masses`` are
    (link index, distance along that link from its joint, kg). Positive torque lifts."""
    n = len(lengths)
    absang = np.cumsum(angles)
    joints_x = np.concatenate([[0.0], np.cumsum([L * math.cos(a) for L, a in zip(lengths, absang)])])
    tau = []
    for i in range(n):
        tq = 0.0
        for link, s, m in masses:
            if link < i:
                continue
            x = joints_x[link] + s * math.cos(absang[link])
            tq += m * g * (x - joints_x[i])
        tau.append(tq)
    return tau


def arm_ik(l1: float, l2: float, x: float, z: float) -> tuple[float, float] | None:
    """Shoulder and elbow angles that put a two link arm's tip at (x, z), elbow up. None if out of reach."""
    c2 = (x * x + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(c2) > 1:
        return None
    t2 = -math.acos(c2)
    t1 = math.atan2(z, x) - math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
    return t1, t2
