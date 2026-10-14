"""Fatigue crack initiation by the stress life method: S-N curve, Marin factors, notches, mean stress and Miner's rule.

The curve follows Shigley (Mechanical Engineering Design, 10th ed., section 6-7): a straight line on log-log axes
from f Sut at 1000 cycles to the endurance strength at the end of the material's data, where f comes from Fig. 6-18
and the endurance strength is the smooth specimen value times the Marin factors

    Se = ka kb kc kd ke S'e      surface, size, load, temperature and reliability

Steels and titanium have an endurance limit past that point. Aluminum, plastics and wood do not, so the line keeps
falling and a life past the data is marked as an extrapolation. A notch is counted through Peterson's notch
sensitivity, Kf = 1 + q (Kt - 1), and a mean stress through Goodman (the default, and the most conservative of the
usual curves for ductile metals), Gerber or Soderberg. A load spectrum adds damage block by block with Miner's
rule, D = sum n_i / N_i, and the part is predicted to crack at D = 1.

Fatigue lives scatter by a factor of three to ten at a given stress, so the safety factor on stress is the steadier
number and the life is an estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..materials import FatigueData, Material

KPSI = 6.894757
# Marin surface factor ka = a Sut^b, Sut in MPa (Shigley Table 6-2). An as cast surface is taken as hot rolled.
SURFACES = {"polished": None, "ground": (1.58, -0.085), "machined": (4.51, -0.265), "cold-drawn": (4.51, -0.265),
            "hot-rolled": (57.7, -0.718), "as-cast": (57.7, -0.718), "as-forged": (272.0, -0.995)}
LOADINGS = {"bending": 1.0, "axial": 0.85, "torsion": 0.59}
MEAN_MODELS = ("goodman", "gerber", "soderberg")


def shigley_f(sut: float) -> float:
    """Fraction of Sut a part survives for 1000 cycles, the fit to Shigley's Fig. 6-18 (Eq. 6-10 in kpsi)."""
    k = sut / KPSI
    if k < 70:
        return 0.9
    k = min(k, 200.0)
    return 1.06 - 2.8e-3 * k + 6.9e-6 * k * k


def surface_factor(surface: str, sut: float, metal: bool) -> tuple[float, str]:
    s = surface.lower().replace(" ", "-")
    if s not in SURFACES:
        raise ValueError(f"unknown surface '{surface}' (use {', '.join(SURFACES)})")
    if not metal or SURFACES[s] is None:
        return 1.0, ("polished, as tested" if metal else "not applied, since the data are for parts made the same way")
    a, b = SURFACES[s]
    return min(1.0, a * sut ** b), f"{s} surface (Shigley Table 6-2" + (", as cast taken as hot rolled)"
                                                                        if s == "as-cast" else ")")


def size_factor(d_mm: float | None, loading: str, metal: bool) -> tuple[float, str]:
    """Shigley Eq. 6-20 for bending and torsion; no size effect in axial loading."""
    if not metal or loading == "axial" or not d_mm:
        return 1.0, "not applied" if not metal else ("axial load" if loading == "axial" else "size not known")
    if d_mm < 2.79:
        return 1.0, f"equivalent diameter {d_mm:.3g} mm, below the range of the fit"
    k = 1.24 * d_mm ** -0.107 if d_mm <= 51 else 1.51 * min(d_mm, 254.0) ** -0.157
    return k, f"equivalent diameter {d_mm:.3g} mm"


def reliability_factor(reliability: float) -> tuple[float, str]:
    """ke = 1 - 0.08 z for an 8% scatter in endurance strength (Shigley Eq. 6-28 and Table 6-5)."""
    from scipy.stats import norm

    if not 0.5 <= reliability < 1.0:
        raise ValueError("reliability is a fraction from 0.5 to below 1, for example 0.9 or 0.99")
    z = float(norm.ppf(reliability))
    return 1.0 - 0.08 * z, f"{100 * reliability:g}% of parts survive"


def notch_sensitivity(data: FatigueData, sut: float, radius_mm: float | None) -> tuple[float, str]:
    """Peterson's q = 1 / (1 + a / r). Steels: a = 0.0254 (2079 / Sut)^1.8 mm; aluminum alloys a = 0.635 mm.
    Without data or a radius, q = 1: the full Kt counts, which is conservative."""
    if radius_mm is None or radius_mm <= 0:
        return 1.0, "notch radius not known, so the full stress concentration counts (q = 1)"
    if data.family == "steel":
        a = 0.0254 * (2079.0 / sut) ** 1.8
    elif data.family == "aluminum":
        a = 0.635
    else:
        return 1.0, f"no notch sensitivity data for {data.family}, so the full stress concentration counts (q = 1)"
    q = 1.0 / (1.0 + a / radius_mm)
    return q, f"Peterson q = 1 / (1 + {a:.3g} / {radius_mm:.3g}) = {q:.2f}"


@dataclass
class Curve:
    """The S-N line of one material in one part: f Sut at 1000 cycles down to Se at ne cycles."""
    su: float
    sy: float
    s1000: float
    se: float
    ne: float
    knee: bool
    factors: dict = field(default_factory=dict)   # name -> (value, why)

    @property
    def b(self) -> float:
        return math.log10(self.se / self.s1000) / math.log10(self.ne / 1e3)

    @property
    def a(self) -> float:
        return self.s1000 / 1e3 ** self.b

    def strength(self, n: float) -> float:
        """Fully reversed stress amplitude the part survives for n cycles."""
        if self.knee and n >= self.ne:
            return self.se
        return self.a * max(n, 1.0) ** self.b

    def life(self, s: float) -> float:
        """Cycles to crack initiation at a fully reversed amplitude s; infinite under an endurance limit."""
        if s <= 0 or (self.knee and s <= self.se):
            return math.inf
        if s >= self.su:
            return 0.5
        return (s / self.a) ** (1.0 / self.b)

    def describe(self) -> str:
        tail = "flat after that" if self.knee else "falling on past it"
        return (f"{self.s1000:.3g} MPa at 1e3 cycles down to {self.se:.3g} MPa at {self.ne:.0e} cycles, {tail}"
                .replace("e+0", "e").replace("e+", "e"))


def build_curve(mat: Material, data: FatigueData, surface: str = "polished", size_mm: float | None = None,
                loading: str = "bending", reliability: float = 0.5) -> Curve:
    metal = mat.category == "metal"
    su = mat.tensile_strength
    if loading not in LOADINGS:
        raise ValueError(f"unknown loading '{loading}' (use {', '.join(LOADINGS)})")
    ka = surface_factor(surface, su, metal)
    kb = size_factor(size_mm, loading, metal)
    kc = (LOADINGS[loading], f"{loading} load") if metal else (1.0, "not applied")
    ke = reliability_factor(reliability)
    factors = {"surface ka": ka, "size kb": kb, "load kc": kc, "temperature kd": (1.0, "room temperature"),
               "reliability ke": ke}
    k = ka[0] * kb[0] * kc[0] * ke[0]
    s1000 = shigley_f(su) * su
    se = min(k * data.strength, 0.999 * s1000)
    return Curve(su, mat.yield_strength, s1000, se, data.cycles, data.knee, factors)


# ------------------------------------------------------------------------------------------ mean stress
def equivalent_amplitude(sa: float, sm: float, curve: Curve, model: str = "goodman") -> float:
    """The fully reversed amplitude that does the same damage as amplitude sa about mean sm. A compressive mean
    gets no credit. Infinite when the mean alone reaches the limit of the model."""
    if sm <= 0 or model == "none":
        return sa
    if model == "goodman":
        d = 1.0 - sm / curve.su
    elif model == "gerber":
        d = 1.0 - (sm / curve.su) ** 2
    elif model == "soderberg":
        d = 1.0 - sm / curve.sy
    else:
        raise ValueError(f"unknown mean stress model '{model}' (use {', '.join(MEAN_MODELS)})")
    return sa / d if d > 0 else math.inf


def safety_factor(sa: float, sm: float, sn: float, curve: Curve, model: str = "goodman") -> float:
    """Factor on the whole load (amplitude and mean together) that reaches the fatigue strength sn
    (Shigley Table 6-7)."""
    if sa <= 0 and sm <= 0:
        return math.inf
    if sm <= 0 or model == "none":
        return sn / sa if sa > 0 else math.inf
    if model == "goodman":
        return 1.0 / (sa / sn + sm / curve.su)
    if model == "soderberg":
        return 1.0 / (sa / sn + sm / curve.sy)
    if model == "gerber":
        if sa <= 0:
            return curve.su / sm
        r = 2 * sm * sn / (curve.su * sa)
        return 0.5 * (curve.su / sm) ** 2 * (sa / sn) * (-1 + math.sqrt(1 + r * r))
    raise ValueError(f"unknown mean stress model '{model}' (use {', '.join(MEAN_MODELS)})")


def first_cycle_yield(sa: float, sm: float, sy: float) -> float:
    """Langer's factor against yielding on the first cycle: Sy / (sa + sm)."""
    peak = sa + abs(sm)
    return sy / peak if peak > 0 else math.inf


# -------------------------------------------------------------------------------------------- spectrum
@dataclass
class Block:
    amplitude: float      # MPa
    mean: float           # MPa
    cycles: float
    label: str = ""


def damage(blocks: list[Block], curve: Curve, model: str = "goodman", scale: float = 1.0) -> tuple[float, list]:
    """Miner's sum over the blocks with every stress times scale, and each block's life."""
    total, lives = 0.0, []
    for b in blocks:
        n = curve.life(equivalent_amplitude(scale * b.amplitude, scale * b.mean, curve, model))
        lives.append(n)
        total += b.cycles / n if n > 0 else math.inf
    return total, lives


def factor_to_failure(blocks: list[Block], curve: Curve, model: str = "goodman", repeats: float = 1.0) -> float:
    """The factor on every stress at which the spectrum, repeated, adds up to D = 1, found by bisection. A factor
    that takes the largest peak to the tensile strength always breaks the part, so it bounds the search."""
    peak = max((b.amplitude + max(b.mean, 0.0) for b in blocks if b.cycles > 0), default=0.0)
    if peak <= 0:
        return math.inf
    lo, hi = 1e-6 * curve.su / peak, 1.01 * curve.su / peak
    if repeats * damage(blocks, curve, model, hi)[0] < 1.0:
        return math.inf
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if repeats * damage(blocks, curve, model, mid)[0] >= 1.0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def steinberg_blocks(sigma_rms: float, rate: float, seconds: float) -> list[Block]:
    """Steinberg's three band method for a Gaussian random stress: 68.3% of the cycles at 1 sigma, 27.1% at
    2 sigma and 4.33% at 3 sigma, about a zero mean. The number of cycles is the zero crossing rate times the
    duration."""
    n = rate * seconds
    return [Block(sigma_rms, 0.0, 0.683 * n, "1 sigma"), Block(2 * sigma_rms, 0.0, 0.271 * n, "2 sigma"),
            Block(3 * sigma_rms, 0.0, 0.0433 * n, "3 sigma")]


def parse_block(text: str, base_amplitude: float, base_mean: float) -> Block:
    """'100% x 2e4', '0.6 x 1e6' or '60% x 1e6 R=-1': a fraction of the studied load and how many cycles.

    With R given, the block cycles between R times its peak and its peak; without it, it keeps the load ratio of the
    study.
    """
    import re

    t = text.strip().lower().replace("×", "x")
    m = re.match(r"^\s*([\d.]+(?:e[-+]?\d+)?)\s*(%?)\s*(?:x|\*|for)\s*([\d.]+(?:e[-+]?\d+)?)\s*(?:cycles?)?\s*"
                 r"(?:,?\s*r\s*=\s*([-+]?[\d.]+))?\s*$", t)
    if not m:
        raise ValueError(f"could not read the load block '{text}'. Write it like '100% x 2e4' or '0.6 x 1e6 R=-1' "
                         "(a fraction of the studied load, then the cycles)")
    frac = float(m.group(1)) / (100.0 if m.group(2) else 1.0)
    cycles = float(m.group(3))
    peak = base_amplitude + base_mean
    if m.group(4) is not None:
        r = float(m.group(4))
        amp, mean = frac * peak * (1 - r) / 2, frac * peak * (1 + r) / 2
    else:
        amp, mean = frac * base_amplitude, frac * base_mean
    return Block(amp, mean, cycles, text.strip())


def cycle_stresses(peak: float, ratio: float) -> tuple[float, float]:
    """Amplitude and mean of a stress that cycles between ratio x peak and peak (R = min / max)."""
    if ratio > 1:
        raise ValueError("the load ratio R = min / max is at most 1 (0 for on and off, -1 for fully reversed)")
    return peak * (1 - ratio) / 2, peak * (1 + ratio) / 2
