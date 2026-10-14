"""Handbook stress intensity factors and the crack turning rule they are checked against.

Stresses are in MPa and lengths in mm, and every K comes back in MPa sqrt(m), the unit crack growth data is
published in. These are the references the finite element answers are compared with: Tada's center and edge
cracked strips, the Newman and Raju surface crack, the Glinka and Shen weight function for an edge crack in any
stress field, and the maximum tangential stress rule of Erdogan and Sih for the angle a mixed mode crack turns.
"""
from __future__ import annotations

import math

import numpy as np

MM = 1e-3      # a crack length in mm times this is in m, so sigma sqrt(pi a) comes out in MPa sqrt(m)


def k_unit(stress: float, a: float) -> float:
    """sigma sqrt(pi a) in MPa sqrt(m), for a in mm."""
    return stress * math.sqrt(math.pi * a * MM)


def center_crack(stress: float, a: float, width: float) -> float:
    """Through crack of length 2a in the middle of a strip of the given width, under uniform tension far away.

    Tada, Paris and Irwin, The Stress Analysis of Cracks Handbook, 3rd ed. (2000), page 2.2, Feddersen's secant
    formula with Tada's correction: F = (1 - 0.025 s^2 + 0.06 s^4) sqrt(sec(pi s / 2)), s = 2a / width, within
    0.1% for any s.
    """
    s = 2.0 * a / width
    return k_unit(stress, a) * (1 - 0.025 * s ** 2 + 0.06 * s ** 4) * math.sqrt(1.0 / math.cos(math.pi * s / 2))


def edge_crack(stress: float, a: float, width: float) -> float:
    """Single edge crack of length a in a strip of the given width under uniform tension, free to bend.

    Tada, Paris and Irwin (2000), page 2.10: F = sqrt(2 tan(pi s / 2) / (pi s)) (0.752 + 2.02 s +
    0.37 (1 - sin(pi s / 2))^3) / cos(pi s / 2), s = a / width, within 0.5% for any s.
    """
    s = a / width
    h = math.pi * s / 2
    return k_unit(stress, a) * math.sqrt(math.tan(h) / h) * (0.752 + 2.02 * s + 0.37 * (1 - math.sin(h)) ** 3) \
        / math.cos(h)


def newman_raju(stress: float, a: float, c: float, t: float, width: float, phi: float = math.pi / 2) -> float:
    """Semi elliptical surface crack of depth a and half length c in a plate of thickness t and the given width,
    under uniform tension, at the parametric angle phi along the front (pi / 2 is the deepest point).

    Newman and Raju, Engineering Fracture Mechanics 15 (1981) 185, for a / c <= 1, within 5% of their finite
    element results for a / t < 0.8, computed with Poisson's ratio 0.3.
    """
    if a > c:
        raise ValueError("this Newman and Raju form is for a / c <= 1")
    q = 1 + 1.464 * (a / c) ** 1.65
    m1 = 1.13 - 0.09 * (a / c)
    m2 = -0.54 + 0.89 / (0.2 + a / c)
    m3 = 0.5 - 1.0 / (0.65 + a / c) + 14 * (1 - a / c) ** 24
    g = 1 + (0.1 + 0.35 * (a / t) ** 2) * (1 - math.sin(phi)) ** 2
    f_phi = ((a / c) ** 2 * math.cos(phi) ** 2 + math.sin(phi) ** 2) ** 0.25
    f_w = math.sqrt(1.0 / math.cos(math.pi * c / width * math.sqrt(a / t)))
    f = (m1 + m2 * (a / t) ** 2 + m3 * (a / t) ** 4) * g * f_phi * f_w
    return k_unit(stress, a) / math.sqrt(q) * f


# Glinka and Shen, Engineering Fracture Mechanics 40 (1991) 1135, edge crack in a semi infinite plate
_WF_EDGE = (0.0719768, 0.246984, 0.514465)


def edge_weight_function(x: np.ndarray, stress: np.ndarray, a: float) -> float:
    """K of an edge crack of length a from the uncracked stress acting across its line, sampled at depths x
    (mm from the free surface, 0 to a). Uniform stress gives 1.1215 sigma sqrt(pi a).

    K = integral of sigma(x) m(x, a) dx with m = 2 / sqrt(2 pi (a - x)) (1 + M1 u^0.5 + M2 u + M3 u^1.5), u = 1 - x / a.
    The square root at the tip is integrated exactly by writing the stress as a function of sqrt(a - x).
    """
    x, stress = np.asarray(x, float), np.asarray(stress, float)
    m1, m2, m3 = _WF_EDGE
    # substitute y = sqrt(a - x): dx / sqrt(a - x) = 2 dy, which removes the singularity
    y = np.linspace(0.0, math.sqrt(a), 201)
    u = y ** 2 / a
    sig = np.interp(a - y ** 2, x, stress)
    integrand = 2.0 * (1 + m1 * np.sqrt(u) + m2 * u + m3 * u ** 1.5) * sig
    k = 2.0 / math.sqrt(2 * math.pi) * float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y)))
    return k * math.sqrt(MM)


def inclined_crack(stress: float, a: float, beta: float) -> tuple[float, float]:
    """K_I and K_II of a through crack of length 2a at angle beta to the load in an infinite plate under uniaxial
    tension (beta = pi / 2 is a crack straight across the load)."""
    k = k_unit(stress, a)
    return k * math.sin(beta) ** 2, k * math.sin(beta) * math.cos(beta)


def mts_angle(k1, k2):
    """Angle the crack turns, radians, by the maximum tangential stress rule (Erdogan and Sih, 1963). Positive
    K_II turns it the negative way, toward the side the sliding pushes open."""
    k1, k2 = np.asarray(k1, float), np.asarray(k2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        th = 2.0 * np.arctan((k1 - np.sqrt(k1 ** 2 + 8.0 * k2 ** 2)) / (4.0 * k2))
    th = np.where(np.abs(k2) < 1e-12 * np.maximum(np.abs(k1), 1e-300), 0.0, th)
    return float(th) if th.ndim == 0 else th


def k_equivalent(k1, k2, k3, nu: float):
    """One K that releases the same energy as the three modes together, sqrt(K_I^2 + K_II^2 + K_III^2 / (1 - nu)).
    A crack that is closed under the load (negative K_I) counts only its sliding modes."""
    k1, k2, k3 = (np.asarray(v, float) for v in (k1, k2, k3))
    return np.sqrt(np.maximum(k1, 0.0) ** 2 + k2 ** 2 + k3 ** 2 / (1.0 - nu))
