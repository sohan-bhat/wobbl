"""wobbl uncertainty: carry input scatter through a finished run to distributions, failure odds and rankings.

The run's answer is one point. Here the inputs the user names (load, material properties, geometry, boundary
conditions) scatter as stated, the discretization error the checks measured comes along as one more input, and
the model of model.py turns a large Latin hypercube sample into distributions of the answer, the probability that
the part fails against its limit and Sobol indices that rank what matters. The result is saved as
uncertainty.json next to the run and shown in its report.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from .. import __version__
from ..units import fmt
from . import uq
from .models import for_result

SOBOL_N = 2 ** 13


def run_uncertainty(result_dir, vary: list[str], samples: int = 200000, method: str = "lhs", seed: int = 1,
                    solves: int | None = None, numerical: bool = True, progress=None) -> dict:
    d = Path(result_dir)
    model = for_result(d)
    inputs = model.parse(list(vary))
    if not inputs:
        raise uq.VaryError("say what scatters, for example --vary 'load 10%' --vary 'yield 8%'")
    num = model.numerical() if numerical else None
    allin = inputs + ([num] if num else [])
    f, info = model.evaluator(allin, progress, solves)
    # summing a field per load node by node costs more per sample than a scale factor, so fewer samples are drawn
    samples = min(samples, info.pop("samples_cap", samples))
    if progress:
        progress("sample", f"{samples:,} {'Latin hypercube' if method == 'lhs' else 'Monte Carlo'} samples")
    X = uq.sample(allin, samples, method, seed)
    Y = f(X)
    outputs = {name: {**uq.stats(Y[:, j]), "unit": model.units.get(name, "")} for j, name in enumerate(model.outputs)}
    probs = {k: {**uq.probability(v), "against": why} for k, (v, why) in model.fails(X, Y, allin).items()}
    dec = model.outputs.index(model.decision)
    if progress:
        progress("sensitivity", "Sobol indices")
    sens = uq.sobol_indices(lambda Z: f(Z)[:, dec], allin, n=SOBOL_N, seed=seed + 1)
    alts = model.alternative_results(Y)
    y = Y[:, dec]
    lo, hi = np.percentile(y, [0.05, 99.95])
    counts, edges = np.histogram(np.clip(y, lo, hi), bins=40, range=(lo, hi))
    res = {"kind": "uncertainty", "domain": model.domain, "wobbl": __version__,
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "inputs": [i.to_dict() for i in allin],
           "model": info, "samples": samples, "sampling": "Latin hypercube" if method == "lhs" else "Monte Carlo",
           "seed": seed, "decision": model.decision, "outputs": outputs, "probability": probs, "alternatives": alts,
           "sensitivity": {"output": model.decision, "base_samples": SOBOL_N, "indices": sens},
           "histogram": {"edges": edges.tolist(), "counts": counts.tolist()}}
    res.update(model.extras())
    res["headline"] = headline(res)
    from .report import refresh

    (d / "uncertainty.json").write_text(json.dumps(_clean(res), indent=1))
    refresh(d)
    return res


def headline(r: dict) -> str:
    """One plain paragraph: the failure odds, the spread of the answer and what drives it."""
    out = r["outputs"][r["decision"]]
    unit = f" {out['unit']}" if out["unit"] else ""
    name = {"safety_factor": "The safety factor", "t_max_C": "The hottest point"}.get(
        r["decision"], f"The {r['decision'].replace('_', ' ')}")
    s = (f"{name} averages {fmt(out['mean'])}{unit} with a standard deviation of {fmt(out['sd'])}{unit}, and 90% "
         f"of samples fall between {fmt(out['p05'])} and {fmt(out['p95'])}{unit}.")
    pf = r["probability"].get("failure")
    if pf:
        s = f"The chance of failing against {pf['against']} is {uq.odds(pf)}. " + s
    top = [x for x in r["sensitivity"]["indices"] if x["total"] >= 0.05]
    if top:
        words = [f"{x['input']} ({x['total']:.2f})" for x in top[:3]]
        lead = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        s += f" By total Sobol index the spread comes from {lead}."
    for a in r.get("alternatives", []):
        s += (f" With {a['label']} instead of {a['assumed']}, the chance of failing is "
              f"{uq.odds(a['probability'])}, so that choice bounds the answer too.")
    return s


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return float(f"{v:.8g}") if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load(result_dir, name: str = "uncertainty.json") -> dict | None:
    p = Path(result_dir) / name
    return json.loads(p.read_text()) if p.exists() else None
