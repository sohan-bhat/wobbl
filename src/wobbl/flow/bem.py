"""Propeller thrust, torque and power from blade element momentum theory.

Each blade is cut into stations. At each one the local airfoil makes lift from the flow it sees, and the momentum
the propeller puts into the air sets how much that flow is slowed and swirled. The two have to agree, which gives
one equation per station. Wobbl uses Drela's formulation from QPROP, which stays well posed at zero forward speed,
so a hovering rotor is no special case, with Prandtl's tip loss written for the helical wake and a Prandtl Glauert
correction near the tip.

The section aerodynamics are those QPROP's guide gives for thin cambered propeller sections: lift 0.5 + 5.8 alpha,
between -0.3 and 1.2, and drag 0.028 + 0.05 (cl - 0.5)^2 scaled by (Re / 70000)^-0.7, with stalled sections
taking extra drag. Propellers in the bundled UIUC set use their measured chord and twist, and their measured
thrust and power come back as a check. On the APC 10x4.7 Slow Flyer the method is within 5% at 3000 to 4000 rpm
and falls about 11% low on thrust and 17% low on power by 5000 rpm, because the generic section ignores how lift
improves with Reynolds number on small props. Across the seven bundled props static thrust lands within about 22%.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
from scipy.optimize import brentq

INCH = 0.0254


@dataclass
class Airfoil:
    cl0: float = 0.5
    cl_a: float = 5.8          # per radian
    cl_min: float = -0.3
    cl_max: float = 1.2
    cd0: float = 0.028
    cd2u: float = 0.050
    cd2l: float = 0.020
    cl_cd0: float = 0.5
    re_ref: float = 70000.0
    re_exp: float = -0.7

    def coeffs(self, alpha: float, re: float, mach: float = 0.0) -> tuple[float, float, bool]:
        """Lift, drag and whether the section is stalled. Past stall the lift holds at its limit and the drag grows
        to that of a flat plate, 2 sin^2 alpha (Hoerner), which is how thin sections behave at these Reynolds
        numbers."""
        cl = (self.cl0 + self.cl_a * alpha) / math.sqrt(max(1.0 - mach * mach, 0.1))
        stalled = not self.cl_min <= cl <= self.cl_max
        cl = min(max(cl, self.cl_min), self.cl_max)
        cd2 = self.cd2u if cl >= self.cl_cd0 else self.cd2l
        cd = (self.cd0 + cd2 * (cl - self.cl_cd0) ** 2) * (max(re, 1e3) / self.re_ref) ** self.re_exp
        if stalled:
            cd = max(cd, 2.0 * math.sin(alpha) ** 2)
        return cl, cd, stalled


@dataclass
class Propeller:
    name: str
    diameter: float            # m
    blades: int
    r_R: np.ndarray
    c_R: np.ndarray
    beta_deg: np.ndarray
    pitch: float = 0.0         # m, nominal
    key: str = ""
    data: dict = field(default_factory=dict)    # measured static and advance ratio data when it is a UIUC prop

    @property
    def radius(self) -> float:
        return self.diameter / 2


@dataclass
class BEMResult:
    rpm: float
    speed: float
    thrust: float              # N
    torque: float              # N m
    power: float               # W
    CT: float
    CP: float
    eta: float
    figure_of_merit: float
    tip_mach: float
    re75: float
    stall_fraction: float
    stations: dict


def _uiuc() -> dict:
    return json.loads(resources.files("wobbl.flow").joinpath("data/uiuc_props.json").read_text())


def prop_names() -> list[str]:
    return sorted(_uiuc()["props"])


def uiuc_source() -> str:
    return _uiuc()["source"]


def load_prop(spec: str) -> Propeller:
    """A UIUC prop by key ('apcsf_10x4.7'), or '10x4.7' / '10x4.7x3' (inches, blades), which takes the measured
    APC Slow Flyer chord shape and a constant geometric pitch."""
    db = _uiuc()["props"]
    key = spec.strip().lower()
    if key in db:
        p = db[key]
        return Propeller(p["name"], p["diameter_in"] * INCH, p["blades"], np.array(p["r_R"]), np.array(p["c_R"]),
                         np.array(p["beta_deg"]), p["pitch_in"] * INCH, key, p)
    for k, p in db.items():   # 'apc 10x4.7 sf' and similar
        if key.replace(" ", "") in (k.replace("_", ""), p["name"].lower().replace(" ", "")):
            return load_prop(k)
    parts = key.replace("in", "").replace('"', "").split("x")
    try:
        d, pitch = float(parts[0]) * INCH, float(parts[1]) * INCH
        blades = int(parts[2]) if len(parts) > 2 else 2
    except (ValueError, IndexError):
        raise ValueError(f"unknown propeller '{spec}'; use a diameter x pitch in inches like 10x4.7, or one of "
                         f"{', '.join(sorted(db))}") from None
    base = db["apcsf_10x4.7"]
    r_R = np.array(base["r_R"])
    beta = np.degrees(np.arctan(pitch / (2 * math.pi * r_R * d / 2)))
    return Propeller(f"{parts[0]}x{parts[1]} in, {blades} blades", d, blades, r_R, np.array(base["c_R"]), beta,
                     pitch, "", {})


def solve(prop: Propeller, rpm: float, speed: float = 0.0, rho: float = 1.225, mu: float = 1.81e-5,
          sound: float = 343.0, stations: int = 40, airfoil: Airfoil | None = None) -> BEMResult:
    af = airfoil or Airfoil()
    R, B = prop.radius, prop.blades
    omega = rpm * 2 * math.pi / 60
    edges = np.linspace(prop.r_R[0], 0.995, stations + 1) * R          # root of the measured blade to the tip
    rs = 0.5 * (edges[1:] + edges[:-1])
    dr = np.diff(edges)
    T = Q = 0.0
    out = {k: [] for k in ("r", "alpha_deg", "cl", "cd", "re", "dT", "stalled")}
    for r, d in zip(rs, dr):
        c = float(np.interp(r / R, prop.r_R, prop.c_R)) * R
        beta = math.radians(float(np.interp(r / R, prop.r_R, prop.beta_deg)))
        ua, ut = speed, omega * r
        u = math.hypot(ua, ut)

        def state(psi):
            wa = 0.5 * ua + 0.5 * u * math.sin(psi)
            wt = 0.5 * ut + 0.5 * u * math.cos(psi)
            vt = ut - wt
            w = math.hypot(wa, wt)
            alpha = beta - math.atan2(wa, wt)
            cl, cd, st = af.coeffs(alpha, rho * w * c / mu, w / sound)
            lam = (r / R) * wa / max(wt, 1e-9)
            f = 0.5 * B * (1 - r / R) / max(lam, 1e-9)
            F = 2 / math.pi * math.acos(min(math.exp(-f), 1.0))
            gam = vt * (4 * math.pi * r / B) * F * math.sqrt(1 + (4 * lam * R / (math.pi * B * r)) ** 2)
            return gam - 0.5 * w * c * cl, wa, wt, w, alpha, cl, cd, st

        psi = _root(lambda p: state(p)[0], ua, u)
        _, wa, wt, w, alpha, cl, cd, st = state(psi)
        dT = 0.5 * rho * B * c * w * (cl * wt - cd * wa) * d
        dQ = 0.5 * rho * B * c * w * (cl * wa + cd * wt) * r * d
        T += dT
        Q += dQ
        for k, v in zip(out, (r, math.degrees(alpha), cl, cd, rho * w * c / mu, dT, st)):
            out[k].append(v)
    n = rpm / 60
    D = prop.diameter
    P = Q * omega
    ct, cp = T / (rho * n * n * D ** 4), P / (rho * n ** 3 * D ** 5)
    area = math.pi * R * R
    fm = (max(T, 0.0) ** 1.5 / math.sqrt(2 * rho * area)) / P if P > 0 and speed == 0 else float("nan")
    eta = T * speed / P if P > 0 and speed > 0 else 0.0
    i75 = int(np.argmin(np.abs(np.array(out["r"]) / R - 0.75)))
    stall = float(np.sum(np.array(out["stalled"]) * dr) / dr.sum())
    return BEMResult(rpm, speed, T, Q, P, ct, cp, eta, fm, omega * R / sound, out["re"][i75], stall,
                     {k: np.array(v) for k, v in out.items()})


def _root(fn, ua: float, u: float) -> float:
    """The station's flow angle psi: the sign change of the residual nearest the unloaded state, by bisection."""
    lo = -0.5 * math.pi + 1e-4 if ua == 0 else -0.5 * math.pi
    grid = np.linspace(lo, 0.999 * math.pi, 181)
    vals = [fn(p) for p in grid]
    start = math.asin(ua / u) if u > 0 else 0.0          # psi of an unloaded blade, where the wake is undisturbed
    best = None
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0 or fa * fb < 0:
            if best is None or abs(0.5 * (a + b) - start) < abs(0.5 * sum(best) - start):
                best = (a, b)
    if best is None:
        return start
    return brentq(fn, best[0], best[1], xtol=1e-12)


def rpm_for_thrust(prop: Propeller, thrust: float, speed: float = 0.0, **kw) -> BEMResult:
    """The speed a propeller must turn to make a thrust, found by root finding on the BEM thrust."""
    def miss(rpm):
        return solve(prop, rpm, speed, **kw).thrust - thrust

    lo, hi = 500.0, 2000.0
    while miss(hi) < 0:
        lo, hi = hi, hi * 1.6
        if hi > 60000:
            raise ValueError(f"{prop.name} cannot make {thrust:.3g} N below 60000 rpm")
    return solve(prop, brentq(miss, lo, hi, xtol=0.5), speed, **kw)


def measured_static(prop: Propeller, rpm: float) -> tuple[float, float] | None:
    """Measured static CT and CP at this rpm, interpolated in the UIUC data, or None outside its range."""
    s = prop.data.get("static")
    if not s or not s["rpm"][0] * 0.9 <= rpm <= s["rpm"][-1] * 1.1:
        return None
    return float(np.interp(rpm, s["rpm"], s["CT"])), float(np.interp(rpm, s["rpm"], s["CP"]))
