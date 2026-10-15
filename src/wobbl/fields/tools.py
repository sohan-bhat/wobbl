"""Field studies from flags and MCP calls: one runner per domain, the same compact result for agents."""
from __future__ import annotations

import uuid
from importlib import import_module

TEMPLATES = {
    "magnetic": ("solenoid", "electromagnet", "magnet"),
    "electric": ("plates", "coax", "sphere"),
    "board": ("microstrip", "stripline", "coplanar", "coupled", "coax"),
    "antenna": ("dipole", "monopole", "patch", "loop", "invf", "link"),
    "sound": ("room", "enclosure", "muffler", "panel"),
}


def runner(domain: str):
    if domain not in TEMPLATES:
        raise ValueError(f"unknown field domain '{domain}'")
    return getattr(import_module(f"wobbl.fields.{domain}"), f"run_{domain}")


def check_params(domain: str, template: str, params: dict) -> None:
    """Read a template's values without solving, so a bad value fails before any mesh is built."""
    if template not in TEMPLATES.get(domain, ()):
        raise ValueError(f"unknown {domain} template '{template}' (use {', '.join(TEMPLATES.get(domain, ()))})")
    mod = import_module(f"wobbl.fields.{domain}")
    p = dict(params)
    if domain == "magnetic":
        mod.TEMPLATES[template][0](p)
    elif domain == "electric":
        mod.TEMPLATES[template][0](p)
    elif domain == "board":
        mod.spec(template, p)
    elif domain == "antenna":
        if template in ("dipole", "monopole", "loop", "invf"):
            from .radiate import route
            if route(template, p).full_wave:
                from .antenna3d import KEYS
                mod.check_known(p, KEYS[template], template)
            else:
                mod.wire_spec(template, p)
        elif template == "link":
            mod.check_known(p, mod.LINK_KEYS, "link")
            mod.link_budget(p)
        else:
            from .antenna3d import KEYS
            mod.check_known(p, mod.PATCH_KEYS | KEYS["patch"], "patch")
    else:
        keys = {"room": mod.ROOM_KEYS, "enclosure": mod.ROOM_KEYS, "muffler": mod.MUFFLER_KEYS,
                "panel": mod.PANEL_KEYS}[template]
        mod.check_known(p, keys, template)


def solve_field_tool(domain: str, template: str, params: dict | None = None, question: str = "") -> dict:
    from ..solvers.locate import wobbl_home

    out = wobbl_home() / "results" / f"{domain}-{template}-{uuid.uuid4().hex[:6]}"
    try:
        o = runner(domain)(template, dict(params or {}), out, question)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}
    values = {k: v for k, v in o.values.items() if not k.startswith("_") and k != "params"}
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics), "values": values,
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir, "time_s": round(o.t_total, 1)}
