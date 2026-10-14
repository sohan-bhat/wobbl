"""Calm water resistance of a displacement hull, by the method of Holtrop and Mennen.

J. Holtrop and G. G. J. Mennen, An approximate power prediction method, International Shipbuilding Progress 29
(1982) 166 to 170. It is a regression on hundreds of model tests and trials: friction from the ITTC 1957 line
times a form factor, plus appendages, waves, a bulbous bow near the surface, an immersed transom and a model to
ship correlation allowance. The paper's own worked example (a 205 m ship at 25 knots, 1793 kN) is one of the
validation cases. Outside the hull forms it was fitted to, mostly Froude numbers past 0.45 and planing craft, it
is an extrapolation and the result says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


KNOT = 0.5144     # m/s, the value the paper uses
NU_SEA = 1.19e-6  # m2/s, the paper's seawater at 15 C
RHO_SEA = 1025.0


@dataclass
class Hull:
    length: float                 # waterline length, m
    beam: float                   # m
    draft: float                  # mean draft, m
    displacement: float           # volume, m3
    cm: float = 0.0               # midship section coefficient, 0 to estimate
    cwp: float = 0.0              # waterplane coefficient, 0 to estimate
    lcb: float = 0.0              # centre of buoyancy forward of mid length, percent of length
    draft_fwd: float = 0.0        # 0 for the mean draft
    bulb_area: float = 0.0        # transverse area of the bulbous bow, m2
    bulb_height: float = 0.0      # height of its centre above the keel, m
    transom_area: float = 0.0     # immersed transom area at rest, m2
    appendage_area: float = 0.0   # wetted area of rudder, shafts, keels, m2
    appendage_factor: float = 1.5  # 1 + k2 of the appendages
    stern: float = 0.0            # Cstern: -25 pram with gondola, -10 V sections, 0 normal, 10 U sections
    wetted: float = 0.0           # wetted area, m2, 0 to estimate
    estimated: list[str] = field(default_factory=list)

    def fill(self) -> "Hull":
        """Estimate what was not given, and say so."""
        cb = self.displacement / (self.length * self.beam * self.draft)
        if not self.cm:
            self.cm = min(0.995, 1.006 - 0.0056 * cb ** -3.56)        # Kerlen (1970)
            self.estimated.append(f"midship coefficient {self.cm:.3f} from the block coefficient (Kerlen)")
        if not self.cwp:
            self.cwp = min(0.95, 0.18 + 0.86 * cb / self.cm)          # Schneekluth and Bertram
            self.estimated.append(f"waterplane coefficient {self.cwp:.3f} from the prismatic coefficient")
        if not self.draft_fwd:
            self.draft_fwd = self.draft
        return self


@dataclass
class Resistance:
    speed: float                  # m/s
    froude: float
    parts: dict                   # N each: friction (with form factor), appendages, waves, bulb, transom, allowance
    total: float                  # N
    power: float                  # W, effective (tow rope) power
    form_factor: float
    wetted: float
    cf: float
    cb: float
    cp: float
    out_of_range: list[str]


def resistance(h: Hull, speed: float, rho: float = RHO_SEA, nu: float = NU_SEA) -> Resistance:
    """Holtrop and Mennen (1982) at one speed, m/s."""
    L, B, T, V = h.length, h.beam, h.draft, h.displacement
    g = 9.81                                     # the paper's value, which its example reproduces
    cb = V / (L * B * T)
    cp = cb / h.cm
    fn = speed / math.sqrt(g * L)
    S = h.wetted or (L * (2 * T + B) * math.sqrt(h.cm) * (0.453 + 0.4425 * cb - 0.2862 * h.cm - 0.003467 * B / T
                                                           + 0.3696 * h.cwp) + 2.38 * h.bulb_area / cb)
    cf = 0.075 / (math.log10(speed * L / nu) - 2) ** 2
    q = 0.5 * rho * speed ** 2
    rf = q * S * cf
    lr = L * (1 - cp + 0.06 * cp * h.lcb / (4 * cp - 1))
    tl = T / L
    c12 = tl ** 0.2228446 if tl > 0.05 else (48.20 * (tl - 0.02) ** 2.078 + 0.479948 if tl > 0.02 else 0.479948)
    c13 = 1 + 0.003 * h.stern
    k1 = c13 * (0.93 + c12 * (B / lr) ** 0.92497 * (0.95 - cp) ** -0.521448 * (1 - cp + 0.0225 * h.lcb) ** 0.6906)
    rapp = q * h.appendage_area * h.appendage_factor * cf

    # waves
    bl = B / L
    c7 = 0.229577 * bl ** 0.33333 if bl < 0.11 else (bl if bl < 0.25 else 0.5 - 0.0625 / bl)
    ie = 1 + 89 * math.exp(-(L / B) ** 0.80856 * (1 - h.cwp) ** 0.30484 * (1 - cp - 0.0225 * h.lcb) ** 0.6367
                           * (lr / B) ** 0.34574 * (100 * V / L ** 3) ** 0.16302)
    c1 = 2223105 * c7 ** 3.78613 * (T / B) ** 1.07961 * (90 - ie) ** -1.37565
    tf = h.draft_fwd or T
    c3 = 0.56 * h.bulb_area ** 1.5 / (B * T * (0.31 * math.sqrt(h.bulb_area) + tf - h.bulb_height)) \
        if h.bulb_area else 0.0
    c2 = math.exp(-1.89 * math.sqrt(c3))
    c5 = 1 - 0.8 * h.transom_area / (B * T * h.cm)
    lam = 1.446 * cp - 0.03 * L / B if L / B < 12 else 1.446 * cp - 0.36
    c16 = 8.07981 * cp - 13.8673 * cp ** 2 + 6.984388 * cp ** 3 if cp < 0.8 else 1.73014 - 0.7067 * cp
    m1 = 0.0140407 * L / T - 1.75254 * V ** (1 / 3) / L - 4.79323 * B / L - c16
    lv = L ** 3 / V
    c15 = -1.69385 if lv < 512 else (0.0 if lv > 1727 else -1.69385 + (L / V ** (1 / 3) - 8.0) / 2.36)
    m2 = c15 * cp ** 2 * math.exp(-0.1 * fn ** -2)
    rw = c1 * c2 * c5 * V * rho * g * math.exp(m1 * fn ** -0.9 + m2 * math.cos(lam * fn ** -2))

    # bulb near the surface, transom, correlation allowance
    rb = 0.0
    if h.bulb_area:
        pb = 0.56 * math.sqrt(h.bulb_area) / (tf - 1.5 * h.bulb_height)
        fni = speed / math.sqrt(g * (tf - h.bulb_height - 0.25 * math.sqrt(h.bulb_area)) + 0.15 * speed ** 2)
        rb = 0.11 * math.exp(-3 * pb ** -2) * fni ** 3 * h.bulb_area ** 1.5 * rho * g / (1 + fni ** 2)
    rtr = 0.0
    if h.transom_area:
        fnt = speed / math.sqrt(2 * g * h.transom_area / (B + B * h.cwp))
        rtr = q * h.transom_area * (0.2 * (1 - 0.2 * fnt) if fnt < 5 else 0.0)
    c4 = min(tf / L, 0.04)
    ca = 0.006 * (L + 100) ** -0.16 - 0.00205 + 0.003 * math.sqrt(L / 7.5) * cb ** 4 * c2 * (0.04 - c4)
    ra = q * S * ca
    parts = {"friction": rf * k1, "appendages": rapp, "waves": rw, "bulb": rb, "transom": rtr, "allowance": ra}
    total = sum(parts.values())
    out = []
    if fn > 0.45:
        out.append(f"Froude number {fn:.2f} is past the 0.45 the method was fitted to")
    if not 0.55 <= cp <= 0.85:
        out.append(f"prismatic coefficient {cp:.2f} is outside the 0.55 to 0.85 of the fitted hulls")
    if not 3.9 <= L / B <= 9.5:
        out.append(f"length to beam {L / B:.1f} is outside the 3.9 to 9.5 of the fitted hulls")
    return Resistance(speed, fn, parts, total, total * speed, k1, S, cf, cb, cp, out)


EXAMPLE = Hull(205.0, 32.0, 10.0, 37500.0, cm=0.98, cwp=0.75, lcb=-0.75, draft_fwd=10.0, bulb_area=20.0,
               bulb_height=4.0, transom_area=16.0, appendage_area=50.0, appendage_factor=1.5, stern=10.0)
EXAMPLE_KN = 1793.26        # the paper's total at 25 knots


# --- the study ---------------------------------------------------------------------------------------------------
@dataclass
class HullStudy:
    hull: Hull
    speed: float                  # m/s
    power: float = 0.0            # installed power, W, 0 when not given
    efficiency: float = 0.55      # propulsive efficiency from installed to tow rope power
    fresh: bool = False
    question: str = ""


def run_hull(study: HullStudy, out_dir):
    import time
    from pathlib import Path

    import numpy as np

    from ..outcome_generic import GenericOutcome
    from ..units import fmt
    from ..verify.checks import Check, confidence
    from .render import chart

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = study.hull.fill()
    rho, nu = (999.1, 1.139e-6) if study.fresh else (RHO_SEA, NU_SEA)
    r = resistance(h, study.speed, rho, nu)
    kn = study.speed / KNOT
    checks = [Check("method range", "pass" if not r.out_of_range else "warn",
                    "this hull and speed sit inside the forms the method was fitted to" if not r.out_of_range else
                    ", ".join(r.out_of_range) + ", so the answer is an extrapolation")]
    # Denny and Mumford's wetted area, an independent check on the hull's numbers
    s_dm = 1.7 * h.length * h.draft + h.displacement / h.draft
    err = abs(r.wetted - s_dm) / s_dm
    checks.append(Check("hand calculation", "pass" if err < 0.1 else "warn",
                        f"Denny and Mumford's formula gives a wetted area of {fmt(s_dm)} m2 against the method's "
                        f"{fmt(r.wetted)} m2 ({100 * err:.2g}% apart)" + ("" if err < 0.1 else
                                                                         ", so check the hull's coefficients")))
    friction = r.parts["friction"] / r.total
    checks.append(Check("speed regime", "pass" if r.froude <= 0.4 else "warn",
                        f"Froude number {r.froude:.2f}, friction is {100 * friction:.0f}% of the resistance" +
                        ("" if r.froude <= 0.4 else ", and past about 0.4 the hull climbs its own bow wave, where "
                                                    "resistance rises steeply")))
    level, reasons = confidence(checks, [("medium", "Holtrop and Mennen is a regression on model tests, typically "
                                                    "within 5 to 10% of a towing tank test of the actual hull")])
    need = r.power / study.efficiency
    if study.power:
        verdict = "ok" if need <= 0.9 * study.power else ("marginal" if need <= study.power else "fails")
    else:
        verdict = "ok"
    head = (f"At {fmt(kn)} knots the hull takes {fmt(r.total / 1000)} kN, so {fmt(r.power / 1000)} kW at the tow "
            f"rope and about {fmt(need / 1000)} kW from the engine at a propulsive efficiency of "
            f"{study.efficiency:g}" + (f", against {fmt(study.power / 1000)} kW installed." if study.power else "."))
    speeds = np.linspace(0.3, 1.25, 30) * study.speed
    curves = [resistance(h, v, rho, nu) for v in speeds]
    fixes = []
    if verdict != "ok":
        top = max((v for v, c in zip(speeds, curves) if c.power / study.efficiency <= 0.9 * study.power),
                  default=None)
        if top:
            fixes.append({"kind": "speed", "title": f"Cruise at {fmt(top / KNOT)} knots",
                          "summary": f"The installed {fmt(study.power / 1000)} kW holds that speed with a 10% "
                                     "margin, since resistance grows faster than the square of speed.",
                          "auto": False})
        longer = Hull(**{**h.__dict__, "length": h.length * 1.15, "beam": h.beam / 1.15, "wetted": 0.0,
                         "estimated": []})
        rl = resistance(longer, study.speed, rho, nu)
        fixes.append({"kind": "hull", "title": f"Lengthen the hull to {fmt(longer.length)} m",
                      "summary": f"At the same displacement and draft, with the beam cut to {fmt(longer.beam)} m, "
                                 f"the resistance becomes {fmt(rl.total / 1000)} kN against {fmt(r.total / 1000)} kN, "
                                 "because a longer hull makes smaller waves at the same speed.", "auto": False})
        fixes.append({"kind": "power", "title": f"Install about {fmt(1.1 * need / 1000)} kW",
                      "summary": "That covers this speed with a 10% margin for a clean hull in calm water.",
                      "auto": False})
    imgs = {}
    try:
        imgs["resistance"] = chart(out / "images" / "resistance.png",
                                   {"total": (speeds / KNOT, [c.total / 1000 for c in curves]),
                                    "friction with form factor": (speeds / KNOT,
                                                                  [c.parts["friction"] / 1000 for c in curves]),
                                    "waves": (speeds / KNOT, [c.parts["waves"] / 1000 for c in curves])},
                                   "speed, knots", "resistance, kN", {"this speed": ([kn], [r.total / 1000])},
                                   "Calm water resistance (Holtrop and Mennen)")
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    metrics = [("resistance", f"{fmt(r.total / 1000)} kN"), ("tow rope power", f"{fmt(r.power / 1000)} kW"),
               ("engine power", f"{fmt(need / 1000)} kW"), ("Froude number", f"{r.froude:.2f}")]
    assumptions = [f"{'Fresh water' if study.fresh else 'Seawater'} at 15 C, {fmt(rho)} kg/m3.",
                   f"Waterline {fmt(h.length)} m, beam {fmt(h.beam)} m, draft {fmt(h.draft)} m, displacement "
                   f"{fmt(h.displacement)} m3, block coefficient {r.cb:.3f}.",
                   *[f"Estimated {e}." for e in h.estimated],
                   "Calm water and a clean hull. Waves, wind and fouling add the sea margin, often 15 to 30%.",
                   f"Propulsive efficiency {study.efficiency:g} from engine to tow rope, typical of a single screw.",
                   "No free surface CFD was run. A RANS solve of the waves around a hull needs millions of cells and "
                   "hours, which is beyond a quick check on a laptop."]
    values = {"resistance_N": r.total, "parts_N": r.parts, "tow_power_W": r.power, "engine_power_W": need,
              "froude": r.froude, "wetted_m2": r.wetted, "form_factor": r.form_factor,
              "_figures": [["resistance", "Resistance against speed, with this speed marked"]]}
    o = GenericOutcome("flow", "hull", study.question, verdict, head, metrics, level, reasons, checks, assumptions,
                       values, imgs, fixes, solver="Holtrop and Mennen (1982)", t_total=time.time() - t0)
    return o.write(out)
