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
