"""The crack growth study as an agent tool: plain arguments in, a compact dict out."""
from __future__ import annotations

import math
import uuid
from pathlib import Path


def _r(x, digits: int = 0):
    """A rounded number, or None for no failure (an infinite life)."""
    return round(float(x), digits) if x is not None and math.isfinite(float(x)) else None


def grow_crack_tool(path: str, material: str, loads: list[str], fixtures: list[str], crack: str,
                    r_ratio: float = 0.1, spectrum: str = "", law: str = "nasgro", growth: str = "",
                    kc: float | None = None, wheeler: float = 0.0, target_life: float | None = None,
                    max_pof: float = 0.01, detectable: str = "1.27 mm", samples: int = 20000, steps: int = 12,
                    budget: float = 1200.0, quality: str = "normal", question: str = "") -> dict:
    from ..cli.crack import study_from_options
    from ..solvers.locate import wobbl_home
    from .study import run_fracture

    try:
        st = study_from_options(path, material, loads or [], fixtures or [], crack, detectable=detectable,
                                r_ratio=r_ratio, spectrum=spectrum, law=law, growth=growth, kc=kc, wheeler=wheeler,
                                target_life=target_life, max_pof=max_pof, samples=samples, steps=steps, budget=budget,
                                quality=quality, question=question)
        out = wobbl_home() / "results" / f"{Path(path).stem}-crack-{uuid.uuid4().hex[:6]}"
        o = run_fracture(st, out)
    except Exception as e:  # noqa: BLE001
        hint = getattr(e, "hint", "")
        return {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}
    v = o.values
    return {"verdict": o.verdict, "headline": o.headline, "where": v["where"],
            "nominal_cycles": _r(v["nominal_cycles"]),
            "percentiles": {k: _r(x) for k, x in v["percentiles"].items()},
            "b1_band": [_r(x) for x in v["b1_band"]], "probability_of_failure_before_target": v["pof"],
            "inspection": {k: _r(x, 3) for k, x in v["inspection"].items()}, "path_end": v["path_end"],
            "solved_steps": v["steps"], "k_table": v["k_table"], "confidence": o.confidence,
            "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir}
