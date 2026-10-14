"""Build laminate studies from flags and MCP calls, and the compact result agents read."""
from __future__ import annotations

import uuid
from pathlib import Path

from .study import LaminateStudy, run_laminate


def study_from(path: str, layup: str, ply: str = "t300-5208", loads=(), fixtures=(), ply_thickness: float | None = None,
               target_sf: float | None = None, reference: str = "", quality: str = "normal", question: str = "",
               params: dict | None = None, keep: bool = False) -> LaminateStudy:
    return LaminateStudy(str(Path(path).expanduser().resolve()), layup, ply, ply_thickness, list(loads),
                         list(fixtures), target_sf, reference, quality, question, params=dict(params or {}), keep=keep)


def compact(o) -> dict:
    v = o.values
    return {"verdict": o.verdict, "headline": o.headline, "first_ply_failure_load_factor": round(v["strength_ratio"], 4),
            "criterion": v["criterion"], "tsai_wu": round(v["tsai_wu"], 4), "hashin": round(v["hashin"], 4),
            "hashin_mode": v["hashin_mode"], "governing_ply": v["governing_ply"],
            "governing_angle": v["governing_angle"], "location_mm": [round(x, 3) for x in v["governing_xyz"]],
            "max_deflection_mm": round(v["max_deflection_mm"], 5), "layup": v["layup"],
            "laminate_constants_MPa": {k: round(x, 2) for k, x in v["clt"].items()},
            "plies": v["plies"], "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir}


def solve_laminate_tool(path: str, layup: str, loads: list[str], fixtures: list[str], ply: str = "t300-5208",
                        ply_thickness: float | None = None, target_sf: float | None = None, reference: str = "",
                        quality: str = "normal", question: str = "") -> dict:
    from ..solvers.locate import wobbl_home

    try:
        st = study_from(path, layup, ply, loads, fixtures, ply_thickness, target_sf, reference, quality, question)
        out = wobbl_home() / "results" / f"{Path(path).stem}-laminate-{uuid.uuid4().hex[:6]}"
        o = run_laminate(st, out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}
    return compact(o)


__all__ = ["study_from", "compact", "solve_laminate_tool"]
