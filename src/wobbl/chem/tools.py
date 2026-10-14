"""Motor, flame and propellant studies from MCP calls, returned as compact dicts."""
from __future__ import annotations

import uuid
from pathlib import Path

from ..solvers.locate import wobbl_home

SKIP = {"time_step_runs", "stations", "products", "composition", "condensed"}


def _compact(o, extra: dict | None = None) -> dict:
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.to_dict()["values"].items() if not k.startswith("_") and k not in SKIP},
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir, **(extra or {})}


def _err(e: Exception) -> dict:
    hint = getattr(e, "hint", "")
    return {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}


def solve_motor_tool(spec: str = "", propellant: str = "", grains: list[str] | None = None, throat: str = "",
                     exit: str = "", casing: str = "", efficiency: float | None = None, question: str = "") -> dict:
    from .motor import run_motor
    from .spec import read_spec, spec_from_flags

    try:
        ms = read_spec(spec or spec_from_flags(propellant, list(grains or []), throat, exit, casing, efficiency),
                       question)
        out = wobbl_home() / "results" / f"{ms.name}-motor-{uuid.uuid4().hex[:6]}"
        o = run_motor(ms, out)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    files = {k: str(Path(o.workdir) / v) for k, v in o.values["files"].items()}
    return _compact(o, files)


def solve_flame_tool(fuel: str, oxidizer: str = "air", phi: float = 1.0, pressure: str = "1 atm",
                     temperature: str = "25 C", limit: str = "", question: str = "") -> dict:
    from ..units import parse_quantity
    from .combustion import run_flame

    try:
        P = parse_quantity(pressure, "stress").value * 1e6
        T = parse_quantity(temperature, "temperature").value + 273.15
        lim = parse_quantity(limit, "temperature").value + 273.15 if limit else None
        out = wobbl_home() / "results" / f"{fuel.replace(' ', '-')}-flame-{uuid.uuid4().hex[:6]}"
        o = run_flame(fuel, oxidizer, float(phi), T, P, out, question, lim)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _compact(o, {"products": o.values["products"]})


def solve_propellant_tool(propellant: str, chamber_pressure: str = "6.895 MPa", area_ratio: float | None = None,
                          exit_pressure: str = "", ambient: str = "1 atm", question: str = "") -> dict:
    from ..units import parse_quantity
    from .combustion import run_propellant

    try:
        pc = parse_quantity(chamber_pressure, "stress").value * 1e6
        pa = parse_quantity(ambient, "stress").value * 1e6
        pe = parse_quantity(exit_pressure, "stress").value * 1e6 if exit_pressure else None
        name = "".join(ch if ch.isalnum() else "-" for ch in propellant)[:40]
        out = wobbl_home() / "results" / f"{name}-propellant-{uuid.uuid4().hex[:6]}"
        o = run_propellant(propellant, pc, area_ratio, pe, out, question, pa)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _compact(o, {"stations": o.values["stations"]})


def footprint() -> dict:
    """What chemistry costs on disk: nothing until the first study, then the pruned Cantera wheel."""
    from . import install

    d = install.lib_dir()
    size = install.tree_size(d) if d.exists() else 0
    if size:
        line = (f"chemistry and motors use Cantera {install.CANTERA_VERSION}, {size / 1e6:.1f} MB in {d} after "
                "pruning")
    elif install.available():
        line = "chemistry and motors use the Cantera already in this Python"
    else:
        line = ("chemistry and motors fetch Cantera on their first run, a 6 MB download that takes about 16 MB "
                "once pruned, or now with wobbl setup --only cantera")
    return {"installed": install.available(), "path": str(d), "installed_bytes": size,
            "version": install.CANTERA_VERSION, "line": line}
