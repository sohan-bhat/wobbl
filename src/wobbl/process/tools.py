"""The manufacturing studies behind the CLI and the MCP server, built from plain text inputs."""
from __future__ import annotations

import uuid
from pathlib import Path

from ..units import parse_quantity


def compact(o) -> dict:
    """What an agent reads first: verdict, numbers, checks, fixes and file paths."""
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.values.items() if not k.startswith("_") and k != "levels"},
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions,
            "fixes": [{"kind": f["kind"], "title": f["title"], "summary": f["summary"]} for f in o.fixes],
            "images": o.images, "report": o.report, "result_dir": o.workdir, "time_s": round(o.t_total, 1)}


def _err(e: Exception) -> dict:
    hint = getattr(e, "hint", "")
    return {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}


def _temp(text, default=None):
    if text in (None, ""):
        return default
    return parse_quantity(str(text), "temperature").value


def casting_study(path: str, alloy: str = "a356", risers=(), chills=(), mold: str = "sand", pour: str = "",
                  mold_temp: str = "", question: str = "", quality: str = "normal", params: dict | None = None,
                  mesh_size: float | None = None, budget: float = 600.0):
    """risers: riser texts, 'auto' (the default when there are none) to place them over the hot spots, or 'none'
    to freeze the bare casting."""
    from .casting import CastingStudy
    from .data import get_alloy, get_mold
    from .regions import parse_chill, parse_riser

    get_alloy(alloy), get_mold(mold)
    words = [str(r).strip().lower() for r in risers or []]
    given = [r for r, w in zip(risers or [], words) if w not in ("auto", "none")]
    return CastingStudy(str(Path(path).expanduser().resolve()), alloy, mold, _temp(pour), _temp(mold_temp, 25.0),
                        risers=[parse_riser(r) for r in given], chills=[parse_chill(c) for c in chills or []],
                        question=question, quality=quality, params=dict(params or {}),
                        auto_risers=not given and "none" not in words, mesh_size=mesh_size, budget=budget)


def solve_casting_tool(path: str, alloy: str = "a356", risers: list[str] | None = None,
                       chills: list[str] | None = None, mold: str = "sand", pour: str = "", mold_temp: str = "",
                       quality: str = "normal") -> dict:
    from ..solvers.locate import wobbl_home
    from .casting import run_casting

    try:
        st = casting_study(path, alloy, risers, chills, mold, pour, mold_temp, quality=quality)
        out = wobbl_home() / "results" / f"{Path(path).stem}-casting-{uuid.uuid4().hex[:6]}"
        o = run_casting(st, out)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return compact(o)
