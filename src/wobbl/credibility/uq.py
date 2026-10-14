"""Input scatter in, distributions out: sampling, surrogates, Sobol indices and the odds of failure.

A Wobbl run answers one set of inputs. Real loads, materials and parts scatter around the numbers in the study,
so this module carries that scatter through a model. Each uncertain input is a normal, lognormal or uniform
distribution. Samples come from a Latin hypercube or plain Monte Carlo, Sobol indices from Saltelli's first order
and Jansen's total estimators on a scrambled Sobol sequence, and the probability of failure from the share of
samples that fail, with an exact binomial interval.

When the model is expensive (a heat solve, a circuit run, a strength study on new geometry) it is replaced by a
polynomial chaos expansion fitted to a small design of real solves. Its leave one out error, what the expansion
gets wrong on a solve it was not fitted to, is reported next to every answer that leans on it.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri

from ..units import fmt

DISTS = ("normal", "lognormal", "uniform")
_DIST_WORDS = {"normal": "normal", "gaussian": "normal", "lognormal": "lognormal", "log-normal": "lognormal",
               "uniform": "uniform", "flat": "uniform"}
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


class VaryError(ValueError):
    pass


@dataclass
class Known:
    """Something a study lets Wobbl vary: its value in the run, the unit it is written in and what it does."""
    nominal: float
    unit: str
    role: str              # load, stiffness, strength, geometry, boundary, material, numerical
    label: str
    positive: bool = True


@dataclass
class Input:
    name: str
    dist: str
    mean: float
    sd: float
    low: float = float("nan")
    high: float = float("nan")
    unit: str = ""
    nominal: float = float("nan")
    role: str = ""
    label: str = ""
    text: str = ""

    @property
    def _ln(self) -> tuple[float, float]:
        s2 = math.log(1.0 + (self.sd / self.mean) ** 2)
        return math.log(self.mean) - 0.5 * s2, math.sqrt(s2)

    def ppf(self, u):
        u = np.clip(np.asarray(u, float), 1e-12, 1 - 1e-12)
        if self.dist == "normal":
            return self.mean + self.sd * ndtri(u)
        if self.dist == "lognormal":
            mu, s = self._ln
            return np.exp(mu + s * ndtri(u))
        return self.low + u * (self.high - self.low)

    def cdf(self, x):
        x = np.asarray(x, float)
        if self.dist == "normal":
            return ndtr((x - self.mean) / self.sd)
        if self.dist == "lognormal":
            mu, s = self._ln
            return ndtr((np.log(np.maximum(x, 1e-300)) - mu) / s)
        return np.clip((x - self.low) / (self.high - self.low), 0.0, 1.0)

    def to_z(self, x):
        """The standard normal variable this value maps to, so every input can share one prior N(0, 1)."""
        x = np.asarray(x, float)
        if self.dist == "normal":
            return (x - self.mean) / self.sd
        if self.dist == "lognormal":
            mu, s = self._ln
            return (np.log(np.maximum(x, 1e-300)) - mu) / s
        return ndtri(np.clip(self.cdf(x), 1e-12, 1 - 1e-12))

    def from_z(self, z):
        return self.ppf(ndtr(np.asarray(z, float)))

    @property
    def family(self) -> str:
        """Polynomial family orthogonal under this input: Hermite for normal and lognormal, Legendre for uniform."""
        return "legendre" if self.dist == "uniform" else "hermite"

    def to_xi(self, x):
        if self.dist == "uniform":
            return 2.0 * (np.asarray(x, float) - self.low) / (self.high - self.low) - 1.0
        return self.to_z(x)

    def describe(self) -> str:
        u = f" {self.unit}" if self.unit and self.unit != "x" else ""
        if self.dist == "uniform":
            return f"{self.name}: uniform from {fmt(self.low)} to {fmt(self.high)}{u}"
        cov = f" ({100 * self.sd / abs(self.mean):.3g}%)" if self.mean else ""
        return f"{self.name}: {self.dist}, mean {fmt(self.mean)}{u}, sd {fmt(self.sd)}{u}{cov}"

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Input":
        return cls(**{k: (float("nan") if v is None and k in ("low", "high", "nominal") else v)
                      for k, v in d.items() if k in cls.__dataclass_fields__})


def parse_vary(text: str, known: dict[str, Known], aliases: dict[str, str] | None = None) -> Input:
    """Read 'yield 8%', 'load lognormal 10%', 'E uniform 70000..75000', 'thickness tol 0.2' or
    'yield normal mean 180 sd 12' into an Input.

    A percentage is a coefficient of variation (half the width for a uniform input), sd is a standard deviation,
    tol is a tolerance band taken as three standard deviations (or the half width of a uniform input) and a..b is
    a range. Numbers are in the unit the input is written in, and the mean is the run's value unless given.
    """
    words = text.strip().split()
    if not words:
        raise VaryError("an empty --vary; write it like 'yield 8%' or 'load 10%'")
    name = words[0].lower()
    name = (aliases or {}).get(name, name)
    if name not in known:
        raise VaryError(f"cannot vary '{words[0]}' in this study; it can vary {', '.join(sorted(known))}")
    k = known[name]
    rest = " " + " ".join(words[1:]).lower() + " "
    dist = None
    for w, d in _DIST_WORDS.items():
        if re.search(rf"\s{re.escape(w)}\s", rest):
            dist, rest = d, re.sub(rf"\s{re.escape(w)}\s", " ", rest, count=1)
            break

    def grab(pattern):
        nonlocal rest
        m = re.search(pattern, rest)
        if not m:
            return None
        rest = rest[:m.start()] + " " + rest[m.end():]
        return [float(g) for g in m.groups()]

    mean = grab(rf"\bmean\s+({_NUM})")
    rng = grab(rf"({_NUM})\s*(?:\.\.|\bto\b)\s*({_NUM})")
    sd = grab(rf"\b(?:sd|std|sigma)\s+({_NUM})")
    tol = grab(rf"(?:\btol(?:erance)?\s+|±\s*|\+/-\s*)({_NUM})")
    pct = grab(rf"({_NUM})\s*%")
    left = rest.split()
    if left and "".join(left) != k.unit.lower().replace(" ", ""):
        raise VaryError(f"could not read '{' '.join(left)}' in '{text}'; write {name} values in "
                        f"{k.unit or 'its own units'}, for example '{name} 8%', '{name} sd 12' or "
                        f"'{name} uniform 10..12'")
    dist = dist or ("uniform" if rng else "normal")
    m = mean[0] if mean else (0.5 * (rng[0] + rng[1]) if rng else k.nominal)
    inp = Input(name, dist, m, 0.0, unit=k.unit, nominal=k.nominal, role=k.role, label=k.label, text=text.strip())
    if rng:
        lo, hi = sorted(rng)
        if dist == "uniform":
            inp.low, inp.high = lo, hi
        else:
            inp.sd = (hi - lo) / 6.0
    elif tol:
        if dist == "uniform":
            inp.low, inp.high = m - abs(tol[0]), m + abs(tol[0])
        else:
            inp.sd = abs(tol[0]) / 3.0
    elif sd:
        if dist == "uniform":
            inp.low, inp.high = m - abs(sd[0]) * math.sqrt(3), m + abs(sd[0]) * math.sqrt(3)
        else:
            inp.sd = abs(sd[0])
    elif pct:
        half = abs(pct[0]) / 100.0 * abs(m)
        if dist == "uniform":
            inp.low, inp.high = m - half, m + half
        else:
            inp.sd = half
    else:
        raise VaryError(f"say how much {name} scatters, for example '{name} 8%', '{name} sd 12' or "
                        f"'{name} uniform 10..12'")
    if dist == "uniform":
        inp.mean, inp.sd = 0.5 * (inp.low + inp.high), (inp.high - inp.low) / math.sqrt(12)
    if not inp.sd > 0:
        raise VaryError(f"'{text}' gives {name} no scatter")
    if k.positive and dist == "normal" and inp.mean - 4 * inp.sd <= 0:
        raise VaryError(f"{name} must stay positive, and a normal spread this wide reaches zero; use "
                        f"'{name} lognormal ...' instead")
    if k.positive and (dist == "lognormal" and inp.mean <= 0 or dist == "uniform" and inp.low <= 0):
        raise VaryError(f"{name} must stay positive in '{text}'")
    return inp


# --- sampling -----------------------------------------------------------------------
def unit_samples(n: int, d: int, method: str = "lhs", seed: int = 1) -> np.ndarray:
    """n points in the unit cube: a Latin hypercube (every input's range cut into n equal slices, one sample in each)
    or plain Monte Carlo."""
    rng = np.random.default_rng(seed)
    if method == "mc":
        return rng.random((n, d))
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def to_inputs(inputs: list[Input], u: np.ndarray) -> np.ndarray:
    return np.column_stack([inp.ppf(u[:, j]) for j, inp in enumerate(inputs)]) if inputs else np.zeros((len(u), 0))


def sample(inputs: list[Input], n: int, method: str = "lhs", seed: int = 1) -> np.ndarray:
    return to_inputs(inputs, unit_samples(n, len(inputs), method, seed))


def _qmc(n: int, d: int, seed: int) -> np.ndarray:
    """A scrambled Sobol sequence, which fills the cube far more evenly than random points, or Monte Carlo when
    this SciPy has none."""
    try:
        from scipy.stats import qmc
    except ImportError:   # pragma: no cover
        return unit_samples(n, d, "mc", seed)
    gen = np.random.default_rng(seed)
    try:
        s = qmc.Sobol(d, scramble=True, rng=gen)
    except TypeError:     # SciPy before 1.15 calls it seed
        s = qmc.Sobol(d, scramble=True, seed=gen)
    return s.random_base2(int(round(math.log2(n))))


def stats(y: np.ndarray) -> dict:
    y = np.asarray(y, float)
    q = np.percentile(y, [1, 5, 50, 95, 99])
    return {"mean": float(y.mean()), "sd": float(y.std(ddof=1)) if len(y) > 1 else 0.0, "p01": float(q[0]),
            "p05": float(q[1]), "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "min": float(y.min()),
            "max": float(y.max())}


def probability(fails: np.ndarray) -> dict:
    """Share of samples that fail, with its exact (Clopper Pearson) 95% interval and the reliability index."""
    from scipy.stats import beta

    fails = np.asarray(fails, bool)
    n, k = len(fails), int(fails.sum())
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    p = k / n
    rel = float(-ndtri(p)) if 0 < p < 1 else None
    return {"value": p, "ci95": [lo, hi], "failures": k, "samples": n, "reliability_index": rel}


def odds(pr: dict) -> str:
    """Plain words for a probability: 'about 1 in 40 (2.5%, 95% interval 2.4% to 2.6%)'."""
    p, (lo, hi), n = pr["value"], pr["ci95"], pr["samples"]
    if pr["failures"] == 0:
        return f"below 1 in {_round_odds(1 / hi)} (no failures in {n:,} samples, 95% upper bound {_pct(hi)})"
    if pr["failures"] == n:
        return f"all {n:,} samples fail"
    one_in = f"about 1 in {_round_odds(1 / p)}" if p < 0.5 else f"{_pct(p)}"
    return f"{one_in} ({_pct(p)}, 95% interval {_pct(lo)} to {_pct(hi)})"


def _round_odds(x: float) -> str:
    if x < 20:
        return f"{x:.2g}"
    mag = 10 ** (len(str(int(x))) - 2)
    return f"{int(round(x / mag) * mag):,}"


def _pct(p: float) -> str:
    if p == 0:
        return "0%"
    if p >= 0.01:
        return f"{100 * p:.3g}%"
    if p >= 1e-3:
        return f"{100 * p:.2g}%"
    return f"{p:.2g}"


# --- polynomial chaos -----------------------------------------------------------------
def _poly(family: str, xi: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal polynomials of one variable up to degree, shape (len(xi), degree + 1)."""
    out = np.ones((len(xi), degree + 1))
    if degree >= 1:
        out[:, 1] = xi
    for n in range(1, degree):
        if family == "hermite":   # probabilists' Hermite, orthogonal under N(0, 1)
            out[:, n + 1] = xi * out[:, n] - n * out[:, n - 1]
        else:                     # Legendre, orthogonal on [-1, 1]
            out[:, n + 1] = ((2 * n + 1) * xi * out[:, n] - n * out[:, n - 1]) / (n + 1)
    norm = np.array([math.sqrt(math.factorial(n)) if family == "hermite" else 1 / math.sqrt(2 * n + 1)
                     for n in range(degree + 1)])
    return out / norm


def multi_indices(d: int, degree: int) -> np.ndarray:
    idx = [a for a in itertools.product(range(degree + 1), repeat=d) if sum(a) <= degree]
    return np.array(sorted(idx, key=lambda a: (sum(a), [-x for x in a])), int).reshape(-1, d)


def n_terms(d: int, degree: int) -> int:
    return math.comb(d + degree, degree)


def design_size(d: int, degree: int = 2) -> int:
    """Solves a design needs: twice the terms of a degree 2 expansion, so its leave one out error means something."""
    return max(2 * n_terms(d, degree), d + 3)


@dataclass
class Surrogate:
    inputs: list[Input]
    degree: int
    alphas: np.ndarray
    coef: np.ndarray            # (terms, outputs)
    loo: np.ndarray             # relative leave one out error, a fraction of each output's variance
    rms_loo: np.ndarray         # the same error in the output's own units
    n_train: int
    names: list[str] = field(default_factory=list)

    def basis(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        polys = [_poly(inp.family, inp.to_xi(X[:, j]), self.degree) for j, inp in enumerate(self.inputs)]
        out = np.ones((len(X), len(self.alphas)))
        for j, P in enumerate(polys):
            out *= P[:, self.alphas[:, j]]
        return out

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.basis(X) @ self.coef

    def summary(self) -> dict:
        return {"kind": "polynomial chaos", "degree": self.degree, "terms": len(self.alphas), "solves": self.n_train,
                "loo_error": {n: float(e) for n, e in zip(self.names, self.loo)},
                "rms_loo": {n: float(e) for n, e in zip(self.names, self.rms_loo)}}


def fit_pce(inputs: list[Input], X: np.ndarray, Y: np.ndarray, names: list[str] | None = None,
            max_degree: int = 3) -> Surrogate:
    """Least squares polynomial chaos of the lowest leave one out error, among the degrees the design supports
    (at least 1.5 solves per term). The leave one out residuals come from the hat matrix, so no refit is needed."""
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.asarray(Y, float).reshape(len(X), -1)
    best = None
    for p in range(1, max_degree + 1):
        alphas = multi_indices(len(inputs), p)
        if len(X) < 1.5 * len(alphas) and p > 1:
            break
        s = Surrogate(inputs, p, alphas, np.zeros((len(alphas), Y.shape[1])), np.zeros(Y.shape[1]),
                      np.zeros(Y.shape[1]), len(X), list(names or []))
        Psi = s.basis(X)
        coef, *_ = np.linalg.lstsq(Psi, Y, rcond=None)
        q, _ = np.linalg.qr(Psi)
        h = np.clip((q ** 2).sum(1), 0.0, 1.0 - 1e-9)
        res = (Y - Psi @ coef) / (1.0 - h)[:, None]
        mse = (res ** 2).mean(0)
        var = Y.var(0)
        s.coef = coef
        s.rms_loo = np.sqrt(mse)
        s.loo = np.where(var > 1e-30 * np.maximum(Y.mean(0) ** 2, 1e-300), mse / np.maximum(var, 1e-300), 0.0)
        if best is None or s.loo.mean() < best.loo.mean() * 0.98:
            best = s
    return best


# --- sensitivity ----------------------------------------------------------------------
def sobol_indices(f, inputs: list[Input], n: int = 8192, seed: int = 2, boot: int = 100) -> list[dict]:
    """First order and total Sobol indices of a scalar model, with bootstrap 95% intervals.

    The first order index is the share of the output's variance one input explains alone, the total index the
    share it has a hand in, interactions included. Inputs are ranked by the total index. f maps an (n, d) array
    of inputs to n outputs and should be cheap: an exact scaling or a surrogate, never a fresh solve per point.
    """
    d = len(inputs)
    U = _qmc(n, 2 * d, seed)
    A, B = to_inputs(inputs, U[:, :d]), to_inputs(inputs, U[:, d:])
    fA, fB = np.asarray(f(A), float), np.asarray(f(B), float)
    fAB = np.empty((d, len(A)))
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB[i] = f(AB)

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        v = np.var(np.concatenate([a, b]))
        if v <= 0:
            return np.zeros(d), np.zeros(d)
        return np.mean(b * (ab - a), axis=1) / v, 0.5 * np.mean((a - ab) ** 2, axis=1) / v

    s1, st = est(slice(None))
    rng = np.random.default_rng(seed + 1)
    draws = [est(rng.integers(0, len(A), len(A))) for _ in range(boot)]
    b1 = np.array([x[0] for x in draws])
    bt = np.array([x[1] for x in draws])
    out = []
    for i, inp in enumerate(inputs):
        out.append({"input": inp.name, "label": inp.label, "role": inp.role, "first_order": float(s1[i]),
                    "total": float(st[i]), "first_order_ci": [float(np.percentile(b1[:, i], 2.5)),
                                                              float(np.percentile(b1[:, i], 97.5))],
                    "total_ci": [float(np.percentile(bt[:, i], 2.5)), float(np.percentile(bt[:, i], 97.5))]})
    return sorted(out, key=lambda r: -r["total"])


def ishigami(X: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """The Ishigami function, a standard test of sensitivity estimators with known indices."""
    X = np.atleast_2d(X)
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_indices(a: float = 7.0, b: float = 0.1) -> dict:
    """Analytic first order and total indices of the Ishigami function with inputs uniform on [-pi, pi]."""
    pi = math.pi
    v1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    v2 = a ** 2 / 8
    v13 = b ** 2 * pi ** 8 * (1 / 18 - 1 / 50)
    v = v1 + v2 + v13
    return {"x1": (v1 / v, (v1 + v13) / v), "x2": (v2 / v, v2 / v), "x3": (0.0, v13 / v)}
