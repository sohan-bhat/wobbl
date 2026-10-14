"""The plastic study behind the MCP server, as a plain function returning a compact dict."""
from __future__ import annotations

import uuid
from pathlib import Path


def compact(o) -> dict:
    """What an agent reads first from any of these studies: verdict, numbers, checks, fixes and file paths."""
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.values.items() if not k.startswith("_") and k not in ("levels", "load_path")},
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions,
            "fixes": [{"kind": f["kind"], "title": f["title"], "summary": f["summary"]} for f in o.fixes],
            "images": o.images, "report": o.report, "result_dir": o.workdir, "time_s": round(o.t_total, 1)}


def solve_plastic_tool(path: str, material: str, loads: list[str], fixtures: list[str], target: float | None = None,
                       quality: str = "normal", curve: str = "auto") -> dict:
    from ..solvers.locate import wobbl_home
    from .study import PlasticStudy, run_plastic

    try:
        st = PlasticStudy(str(Path(path).expanduser().resolve()), material, list(loads or []), list(fixtures or []),
                          target, curve, quality)
        out = wobbl_home() / "results" / f"{Path(path).stem}-plastic-{uuid.uuid4().hex[:6]}"
        o = run_plastic(st, out)
    except Exception as e:  # noqa: BLE001
        hint = getattr(e, "hint", "")
        return {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}
    return compact(o)
