"""Buckling, vibration and fatigue for agents: plain functions returning compact dicts with file paths."""
from __future__ import annotations

import uuid
from pathlib import Path


def _out(path: str, what: str) -> Path:
    from ..solvers.locate import wobbl_home

    stem = Path(path).stem if path else what
    return wobbl_home() / "results" / f"{stem}-{what}-{uuid.uuid4().hex[:6]}"


def _common(o) -> dict:
    return {"verdict": o.verdict, "headline": o.headline, "confidence": o.confidence,
            "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": [{"kind": f.get("kind", ""), "title": f["title"],
                                                      "summary": f["summary"]} for f in o.fixes],
            "images": o.images, "report": o.report, "result_dir": o.workdir, "time_s": round(o.t_total, 1)}


def _r(v, n=4):
    return None if v is None or v != v or v in (float("inf"), float("-inf")) else round(float(v), n)


def solve_buckling_tool(path: str, material: str, loads: list[str], fixtures: list[str],
                        target_sf: float | None = None, shape: str = "auto", quality: str = "normal") -> dict:
    from . import api

    try:
        o = api.buckling(path, material, fixtures, loads, target_sf=target_sf, shape=shape, quality=quality,
                         out=_out(path, "buckling"))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    v = o.values
    return {**_common(o), "load_factor": _r(v.get("load_factor")), "critical_load_N": _r(v.get("critical_load_N"), 2),
            "shape": v.get("shape"), "knockdown": _r(v.get("knockdown"), 3), "design_factor": _r(v.get("design_factor")),
            "yield_factor": _r(v.get("yield_factor")), "target_safety_factor": v["target_sf"],
            "all_load_factors": [round(f, 4) for f in v["load_factors"]]}


def solve_vibration_tool(path: str, material: str, fixtures: list[str], masses: list[str] | None = None,
                         rpm: list[str] | None = None, blades: int = 2, frequencies: list[str] | None = None,
                         sine: str = "", psd: str = "", axis: str = "z", duration: str = "",
                         damping: float | None = None, modes: int = 15, target_sf: float | None = None,
                         quality: str = "normal", loads: list[str] | None = None) -> dict:
    from . import api

    try:
        o = api.vibration(path, material, fixtures, loads=loads or [], masses=masses or [], rpm=rpm or [],
                          blades=blades, frequencies=frequencies or [], sine=sine, psd=psd, axis=axis,
                          duration=duration, damping=damping, modes=modes, target_sf=target_sf, quality=quality,
                          out=_out(path, "vibration"))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    v = o.values
    out = {**_common(o), "frequencies_Hz": [round(f, 3) for f in v["frequencies"]],
           "mass_fraction_xyz": v["mass_fraction"], "damping": v["damping"]}
    for k in ("resonance", "harmonic", "random"):
        if v.get(k):
            out[k] = v[k]
    if v.get("sf") is not None:
        out["safety_factor"] = round(v["sf"], 3)
    return out


def fatigue_life_tool(result_dir: str = "", material: str = "", amplitude: str = "", mean: str = "",
                      kt: float | None = None, radius: str = "", ratio: float = 0.0, cycles: str = "1e7",
                      blocks: list[str] | None = None, repeats: float = 1.0, duration: str = "", surface: str = "auto",
                      reliability: float = 0.9, mean_stress: str = "goodman", target_sf: float | None = None,
                      history: str = "", path: str = "", fixtures: list[str] | None = None,
                      loads: list[str] | None = None, quality: str = "normal") -> dict:
    from . import api

    kw = dict(material=material, amplitude=amplitude, mean=mean, kt=kt, radius=radius, ratio=ratio, cycles=cycles,
              blocks=blocks or [], history=history, repeats=repeats, duration=duration, surface=surface,
              reliability=reliability, mean_stress=mean_stress, target_sf=target_sf)
    try:
        if path:
            o = api.fatigue(part=path, fixtures=fixtures or [], loads=loads or [], quality=quality,
                            out=_out(path, "fatigue"), **kw)
        else:
            o = api.fatigue(result_dir, out=_out(result_dir.rstrip("/") if result_dir else "", "fatigue"), **kw)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    v = o.values
    life = v["life"]
    return {**_common(o), "life": round(life, 1) if life != float("inf") else "infinite",
            "required": v["required"], "safety_factor": round(v["sf"], 3), "damage": v["damage"],
            "mean_stress_model": v["mean_model"], "kt": v["kt"], "kf": v["kf"], "sn_source": v["data_source"],
            "blocks": v["blocks"][:50]}


def footprint() -> dict:
    """What buckling, vibration and fatigue add to an install: they run on the CalculiX that strength already uses,
    and fatigue on numpy, so they download nothing and leave nothing to prune."""
    code = sum(p.stat().st_size for p in Path(__file__).parent.glob("*.py"))
    return {"download_bytes": 0, "code_bytes": code,
            "line": f"buckling, vibration and fatigue use the CalculiX strength already installed, so they download "
                    f"nothing and their own code is {code / 1000:.0f} kB"}


__all__ = ["solve_buckling_tool", "solve_vibration_tool", "fatigue_life_tool", "footprint"]
