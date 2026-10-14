"""Crack growth laws, load spectra and retardation.

Every rate is in mm per cycle for a stress intensity range in MPa sqrt(m), the units growth data is published in.

The NASGRO equation (Forman and Mettu, 1992, as used in NASGRO and AFGROW) is

    da/dN = C ((1 - f) / (1 - R) dK)^n (1 - dK_th / dK)^p / (1 - K_max / K_c)^q

where f is Newman's crack opening function (Newman, 1984), dK_th the threshold, which falls with R and for very
short cracks, and K_c the toughness. Paris is the simple option, da/dN = C dK^n at any R. A material's C and n are
the Paris fit at R = 0, so NASGRO's C is that coefficient divided by (1 - f(0))^n, which makes the two laws agree
at R = 0 away from threshold and fracture, and NASGRO adds the R ratio, the threshold and the toughness.

Wheeler's model (1972) slows growth after a large cycle: while the crack grows through the plastic zone the large
cycle left, each cycle's rate is multiplied by (r_y / (a_ol + r_ol - a))^m_w, with r_y = (K_max / sigma_y)^2 /
(2 pi) the plastic zone of that cycle (Irwin, plane stress). The exponent m_w is fitted to tests on the material
and the spectrum, so Wobbl only uses it when it is given.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

A0_NASGRO = 0.0381    # intrinsic crack length in the threshold term, mm (0.0015 in, NASGRO's default)


def closure_f(r, alpha: float = 2.0, smax_flow: float = 0.3):
    """Newman's crack opening function: the fraction of K_max at which the crack is open. alpha is the
    constraint (1 plane stress, 3 plane strain) and smax_flow the peak stress over the flow stress."""
    r = np.asarray(r, float)
    a0 = (0.825 - 0.34 * alpha + 0.05 * alpha ** 2) * math.cos(math.pi / 2 * smax_flow) ** (1.0 / alpha)
    a1 = (0.415 - 0.071 * alpha) * smax_flow
    a3 = 2 * a0 + a1 - 1
    a2 = 1 - a0 - a1 - a3
    pos = np.maximum(r, a0 + a1 * r + a2 * r ** 2 + a3 * r ** 3)
    neg = a0 + a1 * np.maximum(r, -2.0)
    return np.where(r >= 0, pos, neg)


@dataclass(frozen=True)
class GrowthLaw:
    kind: str                     # nasgro or paris
    C: float                      # mm/cycle; for NASGRO on the effective range (1 - f) / (1 - R) dK
    n: float
    dk1: float = 0.0              # threshold at R = 0 for a long crack, MPa sqrt(m), 0 for none
    kc: float = 0.0               # toughness, MPa sqrt(m), 0 for none
    p: float = 0.5
    q: float = 1.0
    alpha: float = 2.0
    smax_flow: float = 0.3
    c_th: float = 0.0             # how the threshold moves with R beyond the opening function
    source: str = ""
    estimated: bool = False       # constants estimated from the metal's stiffness, not measured on the alloy

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def a0(self) -> float:
        """Newman's opening function at R = 0, the fraction of the range that is closed there."""
        return float(closure_f(0.0, self.alpha, self.smax_flow))

    def threshold(self, r, a=None):
        """dK_th in MPa sqrt(m) at R, for a crack of length a in mm (None for a long crack)."""
        if self.dk1 <= 0:
            return np.zeros_like(np.asarray(r, float))
        r = np.minimum(np.asarray(r, float), 0.95)
        f = closure_f(r, self.alpha, self.smax_flow)
        ratio = (1 - f) / ((1 - self.a0) * (1 - r))
        short = 1.0 if a is None else np.sqrt(np.asarray(a, float) / (np.asarray(a, float) + A0_NASGRO))
        return self.dk1 * short / ratio ** (1 + self.c_th * r)

    def rate(self, dk, r, a=None, kc=None):
        """da/dN in mm/cycle for ranges dk (MPa sqrt(m)) at R ratios r. Zero below threshold, infinite once K_max
        reaches the toughness (kc overrides the law's own, per sample)."""
        # r and a are usually one value per kind of cycle and per crack size, so they stay small here
        r = np.minimum(np.asarray(r, float), 0.95)
        dk = np.maximum(np.asarray(dk, float), 0.0)
        if self.kind == "paris":
            out = self.C * dk ** self.n
        else:
            f = closure_f(r, self.alpha, self.smax_flow)
            out = self.C * ((1 - f) / (1 - r) * dk) ** self.n
            th = self.threshold(r, a)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out * np.where(dk > th, 1 - th / np.where(dk > 0, dk, 1.0), 0.0) ** self.p
        k_crit = np.asarray(self.kc if kc is None else kc, float)
        if np.any(k_crit > 0):
            frac = dk / (1 - r) / np.where(k_crit > 0, k_crit, np.inf)     # K_max over the toughness
            if self.kind == "nasgro":
                with np.errstate(divide="ignore"):
                    out = out / np.maximum(1 - frac, 0.0) ** self.q
            out = np.where(frac >= 1, np.inf, out)
        return out

    def scaled(self, factor: float) -> "GrowthLaw":
        """The same law growing factor times faster, as a sample from the material scatter does."""
        from dataclasses import replace
        return replace(self, C=self.C * factor)


def law_for(material, kind: str = "nasgro", custom: str = "", kc: float | None = None) -> GrowthLaw:
    """The growth law of a material (a library key or a Material), or of custom constants
    'C=2.7e-8,n=3.7,dkth=1.8,kic=29'. A metal without data of its own gets constants estimated from its stiffness."""
    from ..materials import crack_growth

    estimated = False
    if custom:
        vals = parse_constants(custom)
        c, n = vals["C"], vals["n"]
        dk1, k_ic = vals.get("dkth", 0.0), vals.get("kic", 0.0)
        p, q, source = vals.get("p", 0.5), vals.get("q", 1.0), "user supplied"
    else:
        key = material if isinstance(material, str) else material.key
        g = crack_growth(key, None if isinstance(material, str) else material)
        c, n, dk1, k_ic, p, q, source, estimated = g.C, g.n, g.dk_th, g.k_ic, g.p, g.q, g.source, g.estimated
    if kc is not None:
        k_ic = kc
    if kind == "paris":
        return GrowthLaw("paris", c, n, 0.0, k_ic, source=source, estimated=estimated)
    law = GrowthLaw("nasgro", c, n, dk1, k_ic, p, q, source=source, estimated=estimated)
    # match the Paris fit at R = 0: there NASGRO's effective range is (1 - f(0)) dK
    from dataclasses import replace
    return replace(law, C=c / (1 - law.a0) ** n)


_CONST_KEYS = {"c": "C", "n": "n", "m": "n", "dkth": "dkth", "dk_th": "dkth", "threshold": "dkth", "kic": "kic",
               "kc": "kic", "k_ic": "kic", "p": "p", "q": "q"}


def parse_constants(text: str) -> dict:
    """'C=2.7e-8,n=3.7,dkth=1.8,kic=29' with C in mm/cycle and K in MPa sqrt(m)."""
    out = {}
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"growth constants are key=value pairs, got '{part}' (for example "
                             "'C=2.7e-8,n=3.7,dkth=1.8,kic=29')")
        k, v = part.split("=", 1)
        key = _CONST_KEYS.get(k.strip().lower())
        if key is None:
            raise ValueError(f"unknown growth constant '{k}' (use C, n, dkth, kic, p, q)")
        out[key] = float(v)
    if "C" not in out or "n" not in out:
        raise ValueError("custom growth constants need at least C (mm/cycle) and n")
    return out


@dataclass
class Spectrum:
    """One repeating block of load cycles, as factors on the loads given (1 is the full load), in the order they
    happen, each row repeated ``count`` times (once when no counts are given)."""
    smax: np.ndarray
    smin: np.ndarray
    name: str = ""
    notes: list[str] = field(default_factory=list)
    count: np.ndarray | None = None

    @property
    def counts(self) -> np.ndarray:
        return np.ones(len(self.smax)) if self.count is None else np.asarray(self.count, float)

    @property
    def cycles(self) -> float:
        return float(self.counts.sum())

    def expanded(self, limit: int = 2_000_000) -> "Spectrum | None":
        """The block written out one cycle per row, or None when it is longer than ``limit`` cycles."""
        if self.count is None:
            return self
        if self.cycles > limit:
            return None
        reps = self.counts.astype(int)
        return Spectrum(np.repeat(self.smax, reps), np.repeat(self.smin, reps), self.name, self.notes)

    @property
    def r(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.smax > 0, self.smin / np.where(self.smax > 0, self.smax, 1.0), 0.0)

    @property
    def peak(self) -> float:
        return float(self.smax.max())

    @property
    def constant(self) -> bool:
        return len(self.smax) == 1

    def describe(self) -> str:
        if self.constant:
            return f"constant amplitude between {_g(self.smin[0])} and {_g(self.smax[0])} of the load (R = " \
                   f"{_g(self.r[0])})"
        return f"a block of {_count(self.cycles)} cycles{f' from {self.name}' if self.name else ''}, peaks up to " \
               f"{_g(self.peak)} of the load"


def _g(v: float) -> str:
    return f"{v:.3g}"


def _count(n: float) -> str:
    return f"{n:,.0f}" if n < 1e6 else f"{n:.3g}"


def parse_spectrum(spec: str = "", r_ratio: float = 0.0) -> Spectrum:
    """Constant amplitude at an R ratio, or a block of cycles from a file, one cycle per line as 'max,min' or
    'max,min,count' load factors, in the order they happen. Lines starting with # are comments."""
    if not spec:
        if not -5.0 <= r_ratio < 1.0:
            raise ValueError(f"R ratio {r_ratio:g} is outside -5 to 1")
        return Spectrum(np.array([1.0]), np.array([r_ratio]), f"R = {r_ratio:g}")
    path = Path(spec)
    if not path.exists():
        if _BLOCK.match(re.split(r"[;\n]+", spec.strip())[0] or ""):
            return parse_blocks(spec, r_ratio)
        raise ValueError(f"spectrum file not found: {spec} (or give load blocks such as '100% x 3650 R=0; "
                         "70% x 1e8 R=0.7')")
    smax, smin, counts = [], [], []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,;\s]+", line) if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            if not smax:
                continue   # a header row
            raise ValueError(f"{path.name} line {ln}: expected 'max,min' or 'max,min,count', got '{line}'") from None
        if len(vals) not in (2, 3):
            raise ValueError(f"{path.name} line {ln}: expected 'max,min' or 'max,min,count', got '{line}'")
        hi, lo = max(vals[0], vals[1]), min(vals[0], vals[1])
        count = vals[2] if len(vals) == 3 else 1.0
        if count < 1:
            raise ValueError(f"{path.name} line {ln}: the count must be at least 1")
        smax.append(hi)
        smin.append(lo)
        counts.append(round(count))
    if not smax:
        raise ValueError(f"{path.name} has no cycles")
    return _spectrum(smax, smin, counts, path.name)


def _spectrum(smax, smin, counts, name) -> Spectrum:
    if len(smax) > 2_000_000:
        raise ValueError(f"{name} has {len(smax):,} rows in one block; Wobbl reads up to two million")
    if max(smax) <= 0:
        raise ValueError(f"{name} never pulls the crack open (every peak is at or below zero)")
    c = np.asarray(counts, float)
    return Spectrum(np.asarray(smax, float), np.asarray(smin, float), name, count=None if np.all(c == 1) else c)


_BLOCK = re.compile(r"^\s*([-+]?[\d.]+(?:e[-+]?\d+)?)\s*(%?)\s*(?:,\s*([-+]?[\d.]+(?:e[-+]?\d+)?)\s*(%?))?"
                    r"\s*(?:x|\*)\s*([\d.]+(?:e[-+]?\d+)?)\s*(?:r\s*=\s*([-+]?[\d.]+))?\s*$", re.I)


def parse_blocks(blocks, r_ratio: float = 0.0) -> Spectrum:
    """A block from lines like '100% x 3650 R=0', '0.7 x 1e8 R=0.714' or '1,0.1 x 20': the peak as a fraction or
    percentage of the loads given (or peak and valley), how many cycles, and R when it differs from ``r_ratio``.
    The same lines a fatigue study takes with --block."""
    if isinstance(blocks, str):
        blocks = [b for b in re.split(r"[;\n]+", blocks) if b.strip()]
    smax, smin, counts = [], [], []
    for b in blocks:
        m = _BLOCK.match(str(b))
        if not m:
            raise ValueError(f"a load block reads like '100% x 3650 R=0' or '1,0.1 x 20', got '{b}'")
        hi = float(m.group(1)) / (100.0 if m.group(2) else 1.0)
        if m.group(3) is not None:
            lo = float(m.group(3)) / (100.0 if m.group(4) else 1.0)
        else:
            r = float(m.group(6)) if m.group(6) is not None else r_ratio
            if not -5.0 <= r < 1.0:
                raise ValueError(f"R ratio {r:g} in '{b}' is outside -5 to 1")
            lo = r * hi
        n = float(m.group(5))
        if n < 1:
            raise ValueError(f"'{b}' needs at least one cycle")
        smax.append(max(hi, lo))
        smin.append(min(hi, lo))
        counts.append(round(n))
    if not smax:
        raise ValueError("no load blocks given")
    return _spectrum(smax, smin, counts, "the load blocks")


def wheeler_factors(spectrum: Spectrum, exponent: float) -> np.ndarray:
    """Retardation of each cycle in a block when the block's largest cycle keeps renewing the overload zone and the
    crack grows only a little through it per block: (r_y / r_ol)^m_w = (S_max / S_peak)^(2 m_w)."""
    if exponent <= 0 or spectrum.constant:
        return np.ones(len(spectrum.smax))
    ratio = np.clip(spectrum.smax / spectrum.peak, 0.0, 1.0)
    return ratio ** (2.0 * exponent)
