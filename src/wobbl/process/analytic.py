"""Closed form answers the manufacturing studies check themselves against.

Each one is exact for its idealized setup, which is how the validation cases
use them, and an order of magnitude guide for a real part, which is how the
studies' hand calculation checks use them.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.special import erf, erfc


def neumann_lambda(k_s: float, k_l: float, a_s: float, a_l: float, c_s: float, latent: float, t_melt: float,
                   t_init: float, t_wall: float) -> float:
    """Root of Neumann's equation for a liquid at t_init > t_melt freezing from a face held at t_wall < t_melt.
    The front sits at X = 2 lambda sqrt(a_s t) (Carslaw and Jaeger, Conduction of Heat in Solids, section 11.2).

        exp(-l^2) / erf(l) - (k_l/k_s) sqrt(a_s/a_l) (Ti - Tm)/(Tm - Tw) exp(-l^2 a_s/a_l) / erfc(l sqrt(a_s/a_l))
            = l sqrt(pi) L / (c_s (Tm - Tw))
    """
    nu = math.sqrt(a_s / a_l)
    st = c_s * (t_melt - t_wall) / latent

    def f(lam):
        return (math.exp(-lam * lam) / erf(lam) - k_l / k_s * nu * (t_init - t_melt) / (t_melt - t_wall)
                * math.exp(-lam * lam * nu * nu) / erfc(lam * nu) - lam * math.sqrt(math.pi) / st)

    return brentq(f, 1e-6, 5.0)


def neumann_front(lam: float, a_s: float, t: float) -> float:
    """Front position, in the units of sqrt(a_s t)."""
    return 2.0 * lam * math.sqrt(a_s * t)


def power_law_gradient(m: float, n: float, mean_velocity: float, half_gap: float) -> float:
    """Pressure gradient of a power law fluid (eta = m gamma_dot^(n-1)) between plates 2h apart at mean velocity u:
    G = m ((2n + 1) u / n)^n / h^(n + 1). Pa/mm for m in Pa s^n, u in mm/s and h in mm."""
    return m * ((2 * n + 1) * mean_velocity / n) ** n / half_gap ** (n + 1)


def disk_gate_pressure(m: float, n: float, flow: float, half_gap: float, radius: float, gate_radius: float) -> float:
    """Gate pressure, Pa, to push a power law melt at flow Q (mm3/s) from a center gate of radius r_g out to R in a
    disk 2h thick: the mean velocity Q / (4 pi h r) integrated through the gradient above,
    p = m / h^(n+1) ((2n+1) Q / (4 pi n h))^n (R^(1-n) - r_g^(1-n)) / (1 - n)."""
    c = m / half_gap ** (n + 1) * ((2 * n + 1) * flow / (4 * math.pi * n * half_gap)) ** n
    if abs(1 - n) < 1e-12:
        return c * math.log(radius / gate_radius)
    return c * (radius ** (1 - n) - gate_radius ** (1 - n)) / (1 - n)


def strip_pressure(m: float, n: float, flow: float, half_gap: float, width: float, length: float) -> float:
    """Pressure, Pa, to push a power law melt at flow Q (mm3/s) the length of a strip of width W and gap 2h."""
    return power_law_gradient(m, n, flow / (2 * half_gap * width), half_gap) * length
