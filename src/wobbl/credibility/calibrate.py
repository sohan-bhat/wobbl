"""Bayesian calibration: which input values make the model agree with the test readings, and how sure that is.

Each uncertain input maps to a standard normal variable through its own distribution, so the prior is N(0, I) in
those variables for any mix of normal, lognormal and uniform inputs. Each reading is the model's prediction plus
Gaussian noise whose standard deviation combines the reading's own uncertainty with the model's numerical
uncertainty. The posterior is sampled by an adaptive Metropolis chain that starts at the most probable point and
learns its step from the curvature there, so a well posed problem mixes from the first step. The chain uses a
fixed seed, so the same data gives the same intervals on every run.
"""
from __future__ import annotations

import numpy as np

from .uq import Input


def calibrate(predict, inputs: list[Input], data: np.ndarray, sigma: np.ndarray, steps: int = 20000,
              burn: int = 2000, seed: int = 3) -> dict:
    """Posterior of the inputs given readings.

    predict maps an (m, d) array of input values to (m, k) predictions at the k readings; data and sigma are the
    k readings and their standard uncertainties. Returns each input's prior and posterior mean, standard deviation
    and 95% interval, the posterior correlation, the chain's acceptance rate and effective sample size, the
    residuals before and after, and a thinned set of posterior draws.
    """
    from scipy.optimize import minimize

    data = np.asarray(data, float)
    sigma = np.asarray(sigma, float)
    d = len(inputs)

    def to_x(z):
        z = np.atleast_2d(z)
        return np.column_stack([inp.from_z(z[:, j]) for j, inp in enumerate(inputs)])

    def logpost(z):
        r = (predict(to_x(z))[0] - data) / sigma
        return float(-0.5 * (z @ z) - 0.5 * (r @ r))

    best = minimize(lambda z: -logpost(z), np.zeros(d), method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000 * d})
    zmap = best.x
    cov = _laplace(logpost, zmap)
    rng = np.random.default_rng(seed)
    z, lp = zmap.copy(), logpost(zmap)
    chain = np.empty((steps, d))
    accepted = 0
    scale = 2.38 ** 2 / d
    prop = np.linalg.cholesky(scale * cov + 1e-12 * np.eye(d))
    for k in range(steps):
        cand = z + prop @ rng.standard_normal(d)
        lc = logpost(cand)
        if np.log(rng.random()) < lc - lp:
            z, lp = cand, lc
            accepted += k >= burn
        chain[k] = z
        if k >= burn and k % 1000 == 999:   # learn the shape of the posterior from the chain so far (Haario)
            c = np.cov(chain[burn:k + 1].T).reshape(d, d)
            if np.all(np.isfinite(c)) and np.linalg.eigvalsh(c).min() > 0:
                prop = np.linalg.cholesky(scale * c + 1e-12 * np.eye(d))
    post = chain[burn:]
    xs = to_x(post)
    out = []
    for j, inp in enumerate(inputs):
        x = xs[:, j]
        lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
        out.append({"input": inp.name, "label": inp.label, "unit": inp.unit, "prior_mean": inp.mean,
                    "prior_sd": inp.sd, "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(med),
                    "ci95": [float(lo), float(hi)], "map": float(to_x(zmap)[0, j]),
                    "narrowed": float(inp.sd / max(x.std(ddof=1), 1e-300))})
    prior_pred = predict(to_x(np.zeros(d)))[0]
    post_pred = predict(np.array([[r["mean"] for r in out]]))[0]
    thin = xs[:: max(1, len(xs) // 2000)]
    return {"inputs": out, "correlation": np.corrcoef(xs.T).reshape(d, d).tolist() if d > 1 else [[1.0]],
            "acceptance": accepted / max(len(post), 1), "effective_samples": float(min(_ess(post[:, j])
                                                                                     for j in range(d))),
            "steps": steps, "burn_in": burn, "seed": seed,
            "residual_before": _residual(prior_pred, data, sigma), "residual_after": _residual(post_pred, data, sigma),
            "draws": thin.tolist()}


def _laplace(logpost, z0: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Posterior covariance from the curvature at the most probable point (finite differences), or the prior's
    when the curvature is not usable."""
    d = len(z0)
    H = np.zeros((d, d))
    f0 = logpost(z0)
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
            if i == j:
                H[i, i] = (logpost(z0 + ei) - 2 * f0 + logpost(z0 - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (logpost(z0 + ei + ej) - logpost(z0 + ei - ej) - logpost(z0 - ei + ej)
                                     + logpost(z0 - ei - ej)) / (4 * h ** 2)
    try:
        cov = np.linalg.inv(-H)
        if np.linalg.eigvalsh(cov).min() > 0:
            return cov
    except np.linalg.LinAlgError:
        pass
    return np.eye(d)


def _ess(x: np.ndarray) -> float:
    """Effective sample size of one chain from its autocorrelation, summed until it first turns negative."""
    x = x - x.mean()
    n = len(x)
    if n < 10 or x.var() == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (x.var() * n)
    s = 0.0
    for k in range(1, n):
        if ac[k] < 0:
            break
        s += ac[k]
    return float(n / (1 + 2 * s))


def _residual(pred: np.ndarray, data: np.ndarray, sigma: np.ndarray) -> dict:
    r = (pred - data) / sigma
    return {"rms_in_sigma": float(np.sqrt(np.mean(r ** 2))), "max_in_sigma": float(np.max(np.abs(r))),
            "mean_relative": float(np.mean((pred - data) / np.where(data != 0, np.abs(data), 1.0)))}
