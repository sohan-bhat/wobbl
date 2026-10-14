"""Published correlations every flow answer is checked against, and that answer the quick cases on their own.

Friction factors (Moody chart): 64/Re for laminar round pipes, Shah and London's fit for laminar rectangular ducts,
Colebrook for turbulent flow, and Churchill's (1977) all regime equation through transition, where no single
value is right. Drag: Clift, Grace and Weber's (1978) standard drag curve for spheres and Sucker and Brauer's
(1975) fit for cylinders in cross flow. Heat: Stephan's equation for laminar flow developing between parallel
plates, Teertstra's composite model for plate fin heat sinks and Gnielinski's for turbulent pipes.
"""
from __future__ import annotations

import math

LAMINAR_RE = 2300.0
TURBULENT_RE = 4000.0

# loss coefficients K of common fittings, turbulent flow, flanged or welded, valves fully open
# (White, Fluid Mechanics, 7th ed., Table 6.5 and Figure 6.19). Real fittings scatter about 30% around these.
FITTINGS = {
    "entrance": 0.5, "rounded-entrance": 0.05, "exit": 1.0, "elbow90": 0.3, "long-elbow90": 0.2, "elbow45": 0.2,
    "bend180": 0.2, "tee-run": 0.2, "tee-branch": 1.0, "gate-valve": 0.15, "ball-valve": 0.05,
    "globe-valve": 8.0, "check-valve": 2.0, "butterfly-valve": 0.3, "filter": 3.0,
}


def regime(re: float) -> str:
    return "laminar" if re < LAMINAR_RE else ("turbulent" if re > TURBULENT_RE else "transitional")


def colebrook(re: float, rel_rough: float = 0.0) -> float:
    """Darcy friction factor of fully turbulent pipe flow, solved from the Colebrook equation by fixed point
    iteration on 1/sqrt(f), started from Haaland's explicit form."""
    x = -1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / re)
    for _ in range(50):
        nx = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / re)
        if abs(nx - x) < 1e-12:
            break
        x = nx
    return 1.0 / x ** 2


def haaland(re: float, rel_rough: float = 0.0) -> float:
    return (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / re)) ** -2


def churchill(re: float, rel_rough: float = 0.0) -> float:
    """Churchill (1977), one equation for laminar, transitional and turbulent flow."""
    a = (2.457 * math.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * rel_rough))) ** 16
    b = (37530.0 / re) ** 16
    return 8.0 * ((8.0 / re) ** 12 + (a + b) ** -1.5) ** (1.0 / 12.0)


def rect_fRe(aspect: float) -> float:
    """f Re of fully developed laminar flow in a rectangular duct, Darcy f and the hydraulic diameter, for
    aspect = short side / long side (Shah and London 1978). 96 for parallel plates, 56.9 for a square."""
    a = min(aspect, 1.0 / aspect) if aspect > 0 else 0.0
    return 96.0 * (1 - 1.3553 * a + 1.9467 * a ** 2 - 1.7012 * a ** 3 + 0.9564 * a ** 4 - 0.2537 * a ** 5)


def friction_factor(re: float, rel_rough: float = 0.0, aspect: float | None = None) -> tuple[float, str]:
    """Darcy friction factor and the correlation behind it. aspect is None for a round pipe."""
    if re <= 0:
        return 0.0, "no flow"
    if re < LAMINAR_RE:
        if aspect is None:
            return 64.0 / re, "64/Re (Hagen Poiseuille)"
        return rect_fRe(aspect) / re, "Shah and London laminar rectangular duct"
    if re > TURBULENT_RE:
        return colebrook(re, rel_rough), "Colebrook"
    return churchill(re, rel_rough), "Churchill, transitional"


def darcy_dp(f: float, length: float, dh: float, rho: float, v: float, k_sum: float = 0.0) -> float:
    """Darcy Weisbach pressure drop, Pa, with minor losses."""
    return (f * length / dh + k_sum) * 0.5 * rho * v * v


# --- drag ------------------------------------------------------------------------------------------------------
def sphere_cd(re: float) -> float:
    """Standard drag curve of a smooth sphere (Clift, Grace and Weber 1978, Table 5.2), within a few percent of the
    data from creeping flow to Re 1e6, through the drag crisis near 3e5."""
    w = math.log10(re)
    if re < 0.01:
        return 3.0 / 16.0 + 24.0 / re
    if re <= 20:
        return 24.0 / re * (1 + 0.1315 * re ** (0.82 - 0.05 * w))
    if re <= 260:
        return 24.0 / re * (1 + 0.1935 * re ** 0.6305)
    if re <= 1500:
        return 10 ** (1.6435 - 1.1242 * w + 0.1558 * w * w)
    if re <= 1.2e4:
        return 10 ** (-2.4571 + 2.5558 * w - 0.9295 * w * w + 0.1049 * w ** 3)
    if re <= 4.4e4:
        return 10 ** (-1.9181 + 0.6370 * w - 0.0636 * w * w)
    if re <= 3.38e5:
        return 10 ** (-4.3390 + 1.5809 * w - 0.1546 * w * w)
    if re <= 4e5:
        return 29.78 - 5.3 * w
    if re <= 1e6:
        return 0.1 * w - 0.49
    return 0.19 - 8e4 / re


def cylinder_cd(re: float) -> float:
    """Drag of a long smooth cylinder in cross flow (Sucker and Brauer 1975), a fit to experiments from Re 1e-4 to
    2e5, about 10% scatter. Past 2e5 the drag crisis makes any single number unreliable."""
    return 1.18 + 6.8 / re ** 0.89 + 1.96 / re ** 0.5 - 0.0004 * re / (1 + 3.64e-7 * re * re)


# drag of bluff shapes past Re 1e4, where it barely depends on Re (Hoerner, Fluid Dynamic Drag, 1965)
BLUFF_CD = {"disk": 1.17, "square-plate": 1.18, "cube": 1.05, "hemisphere-open-front": 1.42}

# published references for validation (exact for the modeled setup or the best measurements)
CYLINDER_RE40_CD = 1.522        # Dennis and Chang (1970), steady 2D Navier Stokes
CYLINDER_RE100_CD = 1.33        # Park, Kwon and Choi (1998), mean over the shedding cycle
CYLINDER_RE3900_ST = 0.21       # Ong and Wallace (1996), hot wire in the near wake


def strouhal_williamson(re: float) -> float:
    """Shedding frequency of a cylinder in laminar parallel shedding, Re 49 to 180 (Williamson 1996 fit)."""
    return -3.3265 / re + 0.1816 + 1.6e-4 * re


def flat_plate_cf(re: float) -> float:
    """Mean skin friction of a smooth flat plate: Blasius when laminar, the ITTC 1957 line when turbulent."""
    if re < 5e5:
        return 1.328 / math.sqrt(re)
    return 0.075 / (math.log10(re) - 2.0) ** 2


# --- heat ------------------------------------------------------------------------------------------------------
def nu_plates_developing(re_dh: float, pr: float, length: float, dh: float) -> float:
    """Mean Nusselt number (on the hydraulic diameter, twice the gap) of laminar flow developing in velocity and
    temperature between parallel plates at a uniform wall temperature: Stephan (1959), as given by Shah and London
    (1978), for 0.1 < Pr < 1000. It falls to 7.54 when the channel is long."""
    x = length / (dh * re_dh * pr)
    return 7.55 + 0.024 * x ** -1.14 / (1 + 0.0358 * pr ** 0.17 * x ** -0.64)


def nu_teertstra(v_channel: float, gap: float, length: float, nu: float, pr: float) -> float:
    """Mean Nusselt number on the fin gap for a plate fin heat sink with the air ducted through it (Teertstra,
    Yovanovich and Culham 2000), a blend of fully developed and developing flow, within about 6% of their tests."""
    re_b = v_channel * gap / nu
    rs = re_b * gap / length
    return ((rs * pr / 2) ** -3 + (0.664 * math.sqrt(rs) * pr ** (1 / 3) * math.sqrt(1 + 3.65 / math.sqrt(rs)))
            ** -3) ** (-1 / 3)


def nu_gnielinski(re: float, pr: float, f: float | None = None) -> float:
    """Gnielinski (1976), turbulent pipe flow, 3000 < Re < 5e6."""
    f = f or (0.790 * math.log(re) - 1.64) ** -2
    return (f / 8) * (re - 1000) * pr / (1 + 12.7 * math.sqrt(f / 8) * (pr ** (2 / 3) - 1))


# --- walls -----------------------------------------------------------------------------------------------------
def first_cell(y_plus: float, u: float, nu: float, length: float, cf: float | None = None) -> float:
    """Height of the first cell, m, that puts its centre at y_plus, from a skin friction estimate. Without a cf the
    flat plate 0.026 / Re^(1/7) of Schlichting is used, which suits most attached flows."""
    re = max(u * length / nu, 1.0)
    cf = cf if cf is not None else 0.026 / re ** (1 / 7)
    u_tau = u * math.sqrt(cf / 2)
    return 2 * y_plus * nu / max(u_tau, 1e-12)
