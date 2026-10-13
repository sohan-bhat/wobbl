"""Checks every structural answer runs before Wobbl will stand behind it.

Each check returns a status (pass, warn, fail, info, skip) and one plain
sentence. ``confidence`` turns the set of checks into high, medium or low,
with the reasons. A result that failed a check is never reported as high
confidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..units import fmt


@dataclass
class Check:
    name: str
    status: str            # pass, warn, fail, info, skip
    summary: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detail"] = _jsonable(d["detail"])
        return d


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(round(float(x), 6))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def convergence_check(values: list[float], tol: float, label: str, unit: str, stopped_by: str = "",
                      disp: list[float] | None = None) -> Check:
    if len(values) < 2:
        return Check("mesh convergence", "fail", f"only one mesh was solved, so the {label} is unverified",
                     {"values": values})
    a, b = values[-2], values[-1]
    ch = abs(b - a) / max(abs(b), 1e-300)
    trend = " -> ".join(fmt(v) for v in values)
    detail = {"values": values, "last_change": ch, "tolerance": tol, "stopped_by": stopped_by}
    if disp:
        detail["max_displacement"] = disp
    if ch <= tol:
        return Check("mesh convergence", "pass",
                     f"{label} settled: {trend} {unit} over {len(values)} meshes, last change {100 * ch:.1f}%", detail)
    if ch <= 0.10:
        return Check("mesh convergence", "warn",
                     f"{label} still moving: {trend} {unit}, last change {100 * ch:.1f}% "
                     f"(stopped by {stopped_by or 'limit'}); expect about that much error", detail)
    return Check("mesh convergence", "fail",
                 f"{label} not converged: {trend} {unit}, last change {100 * ch:.1f}%. The number is not "
                 "reliable yet; run with --quality fine or a longer --budget", detail)


def balance_check(applied: np.ndarray, reaction: np.ndarray, applied_m: np.ndarray, reaction_m: np.ndarray,
                  size: float, partial: bool = False) -> Check:
    fa = float(np.linalg.norm(applied))
    if fa < 1e-12 and np.linalg.norm(applied_m) < 1e-12:
        return Check("force balance", "skip", "no net load to balance")
    err = float(np.linalg.norm(applied + reaction)) / max(fa, 1e-12)
    mref = max(float(np.linalg.norm(applied_m)), fa * size, 1e-12)
    merr = float(np.linalg.norm(applied_m + reaction_m)) / mref
    detail = {"applied_N": applied, "reaction_N": reaction, "force_error": err, "moment_error": merr}
    if partial:
        detail["note"] = "sliding supports carry part of the load through constraint equations"
        return Check("force balance", "info",
                     f"supports carry {fmt(float(np.linalg.norm(reaction)))} N of {fmt(fa)} N directly; the rest "
                     "goes through sliding supports", detail)
    worst = max(err, merr)
    s = f"reactions balance the loads to {100 * err:.2g}% (forces) and {100 * merr:.2g}% (moments)"
    if worst <= 0.005:
        return Check("force balance", "pass", s, detail)
    if worst <= 0.02:
        return Check("force balance", "warn", s, detail)
    return Check("force balance", "fail", s + "; the solution is not in equilibrium", detail)


def confidence(checks: list[Check], extra_reasons: list[tuple[str, str]] | None = None) -> tuple[str, list[str]]:
    """high unless something argues otherwise; any failed check means low."""
    level = 2
    reasons = []
    for c in checks:
        if c.status == "fail":
            level = 0
            reasons.append(f"{c.name} failed: {c.summary}")
        elif c.status == "warn":
            level = min(level, 1)
            reasons.append(f"{c.name}: {c.summary}")
    for lvl, why in extra_reasons or []:
        level = min(level, {"low": 0, "medium": 1}.get(lvl, 2))
        reasons.append(why)
    if level == 2:
        passed = [c.name for c in checks if c.status == "pass"]
        reasons.insert(0, "passed " + ", ".join(passed) if passed else "no warnings")
    return ["low", "medium", "high"][level], reasons
