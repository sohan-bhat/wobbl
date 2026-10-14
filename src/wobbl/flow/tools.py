"""Flow studies from MCP calls. One tool, solve_flow, takes the kind of study and its values with units."""
from __future__ import annotations

import uuid

from .fluids import qty

KINDS = ("pipe", "drag", "prop", "hull")


def _num(params: dict, key: str, kind: str, default: float = 0.0) -> float:
    """A value from the call: plain numbers stay as they are, anything else is read with its unit."""
    v = params.get(key)
    if v in (None, ""):
        return default
    return float(v) if kind == "plain" else qty(str(v), kind)


def build(kind: str, params: dict):
    """The study for a kind and its parameters, and the function that runs it."""
    p = {k.lower(): v for k, v in (params or {}).items()}
    if kind == "drag":
        from .drag import DragStudy, run_drag
        return DragStudy(str(p.get("shape") or "sphere").lower(), _num(p, "size", "length"),
                         _num(p, "speed", "velocity"), _num(p, "length", "length"), str(p.get("fluid") or "air"),
                         _num(p, "temp", "temperature", 20.0), _num(p, "cd", "plain"), _num(p, "max_drag", "force"),
                         str(p.get("question", ""))), run_drag
    if kind == "prop":
        from .propeller import PropStudy, run_prop
        return PropStudy(str(p.get("prop", "apcsf_10x4.7")), rpm=_num(p, "rpm", "plain"),
                         thrust=_num(p, "thrust", "force"), mass=_num(p, "mass", "mass"),
                         rotors=int(_num(p, "rotors", "plain", 1)), speed=_num(p, "speed", "velocity"),
                         air_temp=_num(p, "air_temp", "temperature", 20.0), altitude=_num(p, "altitude", "length"),
                         max_rpm=_num(p, "max_rpm", "plain"), question=str(p.get("question", ""))), run_prop
    if kind == "hull":
        from .hull import RHO_SEA, Hull, HullStudy, run_hull
        fresh = bool(p.get("fresh", False))
        disp = str(p.get("displacement", ""))
        if disp.replace(" ", "").lower().endswith(("m3", "m^3")):
            vol = float(disp.lower().replace("m^3", "").replace("m3", ""))
        else:
            vol = qty(disp, "mass") / (999.1 if fresh else RHO_SEA)
        h = Hull(_num(p, "length", "length"), _num(p, "beam", "length"), _num(p, "draft", "length"), vol,
                 cm=_num(p, "cm", "plain"), cwp=_num(p, "cwp", "plain"), lcb=_num(p, "lcb", "plain"),
                 transom_area=_num(p, "transom_area", "plain"))
        return HullStudy(h, _num(p, "speed", "velocity"), _num(p, "power", "power"),
                         _num(p, "efficiency", "plain", 0.55), fresh, str(p.get("question", ""))), run_hull
    if kind == "pipe":
        from .pipe import PipeStudy, parse_fittings, run_pipe
        if not p.get("diameter") and not (p.get("width") and p.get("height")):
            raise ValueError("give the pipe's diameter, or a duct's width and height")
        return PipeStudy(_num(p, "length", "length"), _num(p, "flow", "flowrate"), _num(p, "diameter", "length"),
                         _num(p, "width", "length"), _num(p, "height", "length"), str(p.get("fluid") or "water"),
                         _num(p, "temp", "temperature", 20.0), str(p.get("material") or "pvc").lower(),
                         _num(p, "roughness", "length") if p.get("roughness") not in (None, "") else None,
                         parse_fittings(str(p.get("fittings") or "")), _num(p, "rise", "length"),
                         _num(p, "max_dp", "pressure"), str(p.get("question", ""))), run_pipe
    raise ValueError(f"unknown flow study '{kind}'; use one of {', '.join(KINDS)}")


def solve_flow_tool(kind: str, params: dict | None = None) -> dict:
    from ..solvers.locate import wobbl_home

    out = wobbl_home() / "results" / f"flow-{kind}-{uuid.uuid4().hex[:6]}"
    try:
        study, run = build(kind, params or {})
        o = run(study, out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.to_dict()["values"].items() if not k.startswith("_")},
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir}
