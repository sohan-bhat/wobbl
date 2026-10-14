"""Cycles from a crack's starting size to failure, for one part or for thousands of them.

The finite element crack path gives the stress intensity at each crack size along the real path, at the loads
given. Growth is linear in the load, so a scaled load scales every K, and a crack that starts smaller or larger
starts further back or further along the same path. Integrating the growth law over that table is a surrogate
of the full simulation that costs microseconds, which is what makes a Monte Carlo over the starting flaw, the
material constants and the load possible. The surrogate keeps the path's shape fixed and only changes how fast
the crack travels along it.

Failure is the first of K_max reaching the toughness and the average stress over the remaining ligament
reaching the flow stress, the mean of yield and tensile strength. A single part's failure cycle cannot be
predicted, only how failures spread over many parts, which is what the percentiles describe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .growth import GrowthLaw, Spectrum, wheeler_factors

Z90 = 1.6449   # two sided 90% normal quantile


@dataclass
class CrackPath:
    """The crack path reduced to what life needs."""
    s: np.ndarray                 # crack size: the starting size plus how far the leading front point moved, mm
    k: np.ndarray                 # governing equivalent K at the loads given, MPa sqrt(m)
    area: np.ndarray              # crack face area, mm2
    section_area: float = 0.0     # area of the uncracked section the crack runs across, mm2 (0 when unknown)
    section_force: float = 0.0    # force that section carries at the loads given, N
    small: Callable[[np.ndarray], np.ndarray] | None = None   # K below the smallest crack solved
    end: str = ""                 # why the path stops
    end_fails: bool = False       # the path stops because the part has failed there (the crack cut through it)
    solved: int = 0               # how many of the sizes come from solved cracks (the rest are extrapolated)

    def k_at(self, s) -> np.ndarray:
        """K at crack sizes s, a power law between solved cracks; below the first, ``small`` or constant beta.
        Beyond the last solved crack there is no answer, and the caller stops there."""
        s = np.asarray(s, float)
        ls, lk = np.log(self.s), np.log(np.maximum(self.k, 1e-12))
        out = np.exp(np.interp(np.log(np.maximum(s, 1e-12)), ls, lk))
        below = s < self.s[0]
        if below.any():
            if self.small is not None:
                out = np.where(below, self.small(np.minimum(s, self.s[0])), out)
            else:
                out = np.where(below, self.k[0] * np.sqrt(np.maximum(s, 1e-12) / self.s[0]), out)
        return out

    def net_stress(self, s) -> np.ndarray:
        """Average stress over the remaining ligament at the loads given, MPa (0 when the section is unknown)."""
        if self.section_area <= 0:
            return np.zeros_like(np.asarray(s, float))
        a = np.interp(np.asarray(s, float), self.s, self.area, left=0.0)
        return abs(self.section_force) / np.maximum(self.section_area - a, 1e-9)


@dataclass
class LifeInputs:
    law: GrowthLaw
    spectrum: Spectrum
    flow_stress: float            # MPa
    wheeler: float = 0.0          # Wheeler exponent, 0 for no retardation


def cycle_types(spectrum: Spectrum, wheeler: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The distinct cycles of a block (peak, valley, retardation, count), with factors rounded to 0.5% of the
    peak so a long measured block costs no more than a few dozen kinds of cycle."""
    phi = wheeler_factors(spectrum, wheeler)
    keep = spectrum.smax > 0
    step = 0.005 * spectrum.peak
    hi = np.round(spectrum.smax[keep] / step) * step
    lo = np.round(spectrum.smin[keep] / step) * step
    key = np.column_stack([hi, lo, np.round(phi[keep], 6)])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    count = np.bincount(inv.ravel(), weights=spectrum.counts[keep], minlength=len(uniq))
    return uniq[:, 0], uniq[:, 1], uniq[:, 2], count


def block_rate(inp: LifeInputs, k: np.ndarray, s: np.ndarray, scale, cmult, kc, types=None) -> np.ndarray:
    """Growth per block, mm, at crack sizes s for samples with load scale, C multiplier and toughness arrays
    shaped to broadcast against k (samples by sizes)."""
    hi, lo, phi, count = types if types is not None else cycle_types(inp.spectrum, inp.wheeler)
    out = np.zeros(np.broadcast_shapes(np.shape(k), np.shape(scale)))
    for h, l_, p, c in zip(hi, lo, phi, count):
        r = l_ / h
        out = out + c * p * inp.law.rate(scale * (h - l_) * k, r, s, kc)
    return out * cmult


def failure_size(path: CrackPath, inp: LifeInputs, grid: np.ndarray, kgrid: np.ndarray, scale, kc) -> np.ndarray:
    """For each sample, the crack size where it fails, from K_max at the block's peak and net section yield.
    Samples that survive the whole path get the path's last size."""
    peak = inp.spectrum.peak
    kmax = scale * peak * kgrid
    margin = np.full(np.broadcast_shapes(np.shape(kmax), np.shape(kc)), -np.inf)
    kc_b = np.broadcast_to(kc, margin.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(kc_b > 0, kmax / np.where(kc_b > 0, kc_b, 1.0) - 1.0, margin)
    if path.section_area > 0 and inp.flow_stress > 0:
        net = scale * peak * path.net_stress(grid) / inp.flow_stress - 1.0
        margin = np.maximum(margin, net)
    hit = margin >= 0
    first = np.where(hit.any(axis=-1), hit.argmax(axis=-1), grid.size - 1)
    i0 = np.maximum(first - 1, 0)
    m0 = np.take_along_axis(margin, i0[..., None], -1)[..., 0]
    m1 = np.take_along_axis(margin, first[..., None], -1)[..., 0]
    g0, g1 = grid[i0], grid[first]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((m1 > m0) & (first > 0) & np.isfinite(m0), -m0 / (m1 - m0), 1.0)
    return np.where(hit.any(axis=-1), g0 + np.clip(t, 0, 1) * (g1 - g0), grid[-1])


def cumulative(grid: np.ndarray, rate: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Cycles to grow from grid[0] to each grid size, by the trapezoid rule in log size, counting only sizes where
    the crack grows, and how many sizes up to each one it does not grow at (below threshold)."""
    with np.errstate(divide="ignore"):
        dn = np.where(rate > 0, block * grid / np.where(rate > 0, rate, 1.0), 0.0)    # cycles per unit ln s
    du = np.diff(np.log(grid))
    seg = 0.5 * (dn[..., 1:] + dn[..., :-1]) * du
    zero = np.zeros(seg.shape[:-1] + (1,))
    return np.concatenate([zero, np.cumsum(seg, axis=-1)], axis=-1), np.cumsum(rate <= 0, axis=-1)


def _stuck(stuck: np.ndarray, grid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Whether a crack growing from lo to hi meets a size where it stops (from the counts of cumulative)."""
    j0 = np.clip(np.searchsorted(grid, lo, side="left"), 0, grid.size - 1)
    j1 = np.clip(np.searchsorted(grid, hi, side="right") - 1, 0, grid.size - 1)
    rows = np.arange(stuck.shape[0])
    before = np.where(j0 > 0, stuck[rows, np.maximum(j0 - 1, 0)], 0)
    return stuck[rows, j1] - before > 0


def _interp_rows(x: np.ndarray, grid: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """rows[i] (on grid) read at x[i], linearly in log size."""
    lx, lg = np.log(np.clip(x, grid[0], grid[-1])), np.log(grid)
    j = np.clip(np.searchsorted(lg, lx) - 1, 0, grid.size - 2)
    t = (lx - lg[j]) / (lg[j + 1] - lg[j])
    a = np.take_along_axis(rows, j[:, None], 1)[:, 0]
    b = np.take_along_axis(rows, (j + 1)[:, None], 1)[:, 0]
    return a + t * (b - a)


def lives(path: CrackPath, inp: LifeInputs, a0, scale, cmult, kc, n_grid: int = 400,
          chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Cycles to failure for each sample, and whether that sample reached the end of the path first.
    Every input is a scalar or an array with one value per sample."""
    a0, scale, cmult, kc = (np.atleast_1d(np.asarray(v, float)) for v in (a0, scale, cmult, kc))
    n = max(a0.size, scale.size, cmult.size, kc.size)
    a0, scale, cmult, kc = (np.broadcast_to(v, (n,)) for v in (a0, scale, cmult, kc))
    lo = min(float(a0.min()), float(path.s[0]))
    grid = np.geomspace(lo, float(path.s[-1]), n_grid)
    kgrid = path.k_at(grid)
    types = cycle_types(inp.spectrum, inp.wheeler)
    out, ran_out = np.empty(n), np.zeros(n, bool)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        sc, cm, kk = scale[sl, None], cmult[sl, None], kc[sl, None]
        rate = block_rate(inp, kgrid[None, :], grid[None, :], sc, cm, kk, types)
        cum, stuck = cumulative(grid, rate, inp.spectrum.cycles)
        fail = failure_size(path, inp, grid, kgrid[None, :], sc, kk)
        start = np.minimum(a0[sl], fail)
        life = np.maximum(_interp_rows(fail, grid, cum) - _interp_rows(start, grid, cum), 0.0)
        # a crack that meets a size below threshold on its way stops there and never fails
        out[sl] = np.where(_stuck(stuck, grid, start, fail) & (fail > start), np.inf, life)
        ran_out[sl] = (fail >= grid[-1] * (1 - 1e-12)) & (not path.end_fails)
    return out, ran_out


def growth_curve(path: CrackPath, inp: LifeInputs, a0: float, n_grid: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Crack size against cycles for the nominal part, from a0 to failure or the end of the path."""
    grid = np.geomspace(min(a0, path.s[0]), path.s[-1], n_grid)
    kgrid = path.k_at(grid)
    rate = block_rate(inp, kgrid, grid, 1.0, 1.0, inp.law.kc)
    cum, _ = cumulative(grid, rate, inp.spectrum.cycles)
    fail = float(failure_size(path, inp, grid, kgrid, 1.0, inp.law.kc))
    stop = grid[(grid >= a0) & (rate <= 0)]
    if len(stop):     # below threshold: the crack grows up to there and stops
        fail = min(fail, float(stop[0]))
    keep = (grid >= a0) & (grid <= fail)
    n0 = float(np.interp(math.log(a0), np.log(grid), cum))
    return grid[keep], cum[keep] - n0


@dataclass
class Scatter:
    """How the inputs scatter from one part to the next, each as the standard deviation of its logarithm."""
    flaw: float = 0.3             # starting crack size
    growth: float = 0.25          # growth coefficient C, from one heat of the material to another
    load: float = 0.05            # the load each part sees
    toughness: float = 0.07       # fracture toughness
    samples: int = 20000
    seed: int = 1

    def draw(self, a_med: float, kc: float) -> dict:
        """Samples, the same for the same seed on every run."""
        rng = np.random.default_rng(self.seed)
        n = self.samples
        z = rng.standard_normal((4, n))
        return {"a0": a_med * np.exp(self.flaw * z[0]), "scale": np.exp(self.load * z[1]),
                "cmult": np.exp(self.growth * z[2]),
                "kc": kc * np.exp(self.toughness * z[3]) if kc > 0 else np.zeros(n)}


@dataclass
class LifeResult:
    nominal: float                # cycles from the starting crack for the nominal part
    lives: np.ndarray = field(repr=False, default=None)
    ran_out: int = 0              # samples that reached the end of the path before failing
    percentiles: dict = field(default_factory=dict)
    b1_band: tuple[float, float] = (0.0, 0.0)
    pof: float | None = None      # probability of failing before the target life
    pof_band: tuple[float, float] = (0.0, 0.0)
    half_b1: float = 0.0          # B1 from the first half of the samples

    def to_dict(self) -> dict:
        return {"nominal_cycles": self.nominal, "ran_out": self.ran_out, "percentiles": self.percentiles,
                "b1_band": list(self.b1_band), "pof": self.pof, "pof_band": list(self.pof_band),
                "b1_half_samples": self.half_b1, "samples": int(self.lives.size) if self.lives is not None else 0}


def monte_carlo(path: CrackPath, inp: LifeInputs, a_med: float, scatter: Scatter, target: float | None = None,
                draws: dict | None = None) -> LifeResult:
    """Lives of many parts, with B50, B10 and B1 (the lives by which half, a tenth and a hundredth have failed),
    sampling bands from order statistics and the probability of failing before a target life."""
    d = draws or scatter.draw(a_med, inp.law.kc)
    life, ran = lives(path, inp, d["a0"], d["scale"], d["cmult"], d["kc"])
    nominal = float(lives(path, inp, a_med, 1.0, 1.0, inp.law.kc)[0][0])
    srt = np.sort(life)
    n = srt.size
    pct = {name: float(np.quantile(life, q)) for name, q in (("B50", 0.5), ("B10", 0.1), ("B1", 0.01))}
    if n >= 10000:
        pct["B0.1"] = float(np.quantile(life, 0.001))
    half = np.sort(life[: n // 2])
    res = LifeResult(nominal, life, int(ran.sum()), pct, _quantile_band(srt, 0.01), half_b1=float(
        np.quantile(half, 0.01)))
    if target:
        k = int((life < target).sum())
        res.pof = k / n
        res.pof_band = _wilson(k, n)
    return res


def _quantile_band(srt: np.ndarray, p: float) -> tuple[float, float]:
    """90% band of a sample quantile from the binomial spread of the rank it sits at."""
    n = srt.size
    sd = math.sqrt(n * p * (1 - p))
    lo, hi = int(max(0, math.floor(n * p - Z90 * sd))), int(min(n - 1, math.ceil(n * p + Z90 * sd)))
    return float(srt[lo]), float(srt[hi])


def _wilson(k: int, n: int) -> tuple[float, float]:
    """90% Wilson band of a probability seen k times in n."""
    p = k / n
    den = 1 + Z90 ** 2 / n
    mid = (p + Z90 ** 2 / (2 * n)) / den
    half = Z90 * math.sqrt(p * (1 - p) / n + Z90 ** 2 / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def inspection(path: CrackPath, inp: LifeInputs, a_start: float, a_detect: float) -> dict:
    """Damage tolerance inspection times for the nominal part: the first inspection at half the life from the
    starting crack to failure, then every half of the life from the size an inspection reliably finds to failure."""
    full = float(lives(path, inp, a_start, 1.0, 1.0, inp.law.kc)[0][0])
    from_det = float(lives(path, inp, a_detect, 1.0, 1.0, inp.law.kc)[0][0])
    return {"first": 0.5 * full, "repeat": 0.5 * from_det, "detectable_mm": a_detect,
            "life_from_detectable": from_det}


def wheeler_reference(path: CrackPath, inp: LifeInputs, a0: float, sigma_y: float,
                      max_steps: int = 400000) -> float:
    """Life with Wheeler retardation followed cycle by cycle through the block, renewing and growing through
    each overload zone, to check the quasi steady factors the Monte Carlo uses.

    The crack size is held over a stretch of cycles short enough that the crack grows under a twentieth of the
    overload zone in it, and within the stretch each cycle's zone and retardation are worked out in order. When a
    whole block grows the crack by under a hundredth of the zone, several blocks are taken at once, holding the
    size to within 1%.
    """
    spec, law, m = inp.spectrum.expanded(), inp.law, inp.wheeler
    if spec is None:
        return math.nan                  # too long a block to follow one cycle at a time
    hi, lo = spec.smax, spec.smin
    r = np.where(hi > 0, lo / np.where(hi > 0, hi, 1.0), 0.0)
    a, edge, n = a0, a0, 0.0           # crack size, front edge of the overload zone (a_ol + r_ol), cycles
    i, size = 0, len(spec.smax)        # position in the block and cycles per stretch
    for _ in range(max_steps):
        k = float(path.k_at(a))
        if (law.kc and k * spec.peak >= law.kc) or \
                (path.section_area > 0 and spec.peak * path.net_stress(a) >= inp.flow_stress) or a >= path.s[-1]:
            return n
        j = min(i + size, len(spec.smax))
        ry = np.maximum(k * hi[i:j], 0.0) ** 2 / (2 * math.pi * sigma_y ** 2) * 1e3     # mm, K in MPa sqrt(m)
        reach = np.maximum.accumulate(np.maximum(a + ry, edge))
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(a + ry >= reach, 1.0, (ry / np.maximum(reach - a, 1e-12)) ** m)
        grow = float((phi * law.rate(k * (hi[i:j] - lo[i:j]), r[i:j], a)).sum())
        zone = max(float(reach[-1] - a), 1e-9)
        if grow > 0.05 * zone and j - i > 1:
            size = max(1, (j - i) // 2)
            continue
        if not np.isfinite(grow):
            return n
        reps = 1
        if i == 0 and j == len(spec.smax):
            if grow <= 0:
                return math.inf
            if grow < 0.01 * zone:
                reps = max(1, int(0.01 * a / grow))
        a += reps * grow
        n += reps * (j - i)
        # the zone renews with every block's peak, so after several blocks at once it has moved with the crack
        edge = float(reach[-1]) + (reps - 1) * grow
        i = 0 if j >= len(spec.smax) else j
        if grow < 0.01 * zone:
            size = min(len(spec.smax), 2 * size)
    return n
