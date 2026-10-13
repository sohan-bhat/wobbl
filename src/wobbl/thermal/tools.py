"""Build thermal studies from flags, plain questions and MCP calls."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from ..units import parse_quantity
from .study import H_TYPICAL, Film, ThermalStudy, parse_film, run_thermal


def study_from(path: str, material: str, power: str = "", heat_faces: str = "", films=(), ambient: str = "25 C",
               fixed=(), duration: str = "", limit: str = "", emissivity: float = 0.0, question: str = "",
               quality: str = "normal") -> ThermalStudy:
    amb = parse_quantity(ambient or "25 C", "temperature").value
    st = ThermalStudy(str(Path(path).expanduser().resolve()), material, ambient=amb, question=question,
                      emissivity=emissivity, quality=quality, heat_faces=heat_faces)
    if power:
        st.power = parse_quantity(power, "power").value
    if isinstance(films, str):
        films = [films] if films else []
    st.films = [parse_film(f, amb) for f in films]
    for fx in fixed or []:
        t, sel = fx.rsplit("@", 1)
        st.fixed.append((parse_quantity(t.strip(), "temperature").value, sel.strip()))
    if duration:
        st.duration = parse_quantity(duration, "time").value
    if limit:
        st.limit = parse_quantity(limit, "temperature").value
    return st


def read_thermal_question(q: str) -> dict:
    """Pull power, ambient, cooling and limit out of a plain question; return guesses too."""
    low = q.lower()
    out, guesses = {}, []
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kw|mw|w|watts?)\b", low)
    if m:
        out["power"] = f"{m.group(1)} {m.group(2).rstrip('s').replace('watt', 'w')}"
    m = re.search(r"(?:at|in|ambient(?: of)?|air(?: at)?)\s+(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(c|f)\b", low)
    if m:
        out["ambient"] = f"{m.group(1)} {m.group(2).upper()}"
    m = re.search(r"(?:above|over|exceed|past|beyond|limit(?: of)?)\s+(\d+(?:\.\d+)?)\s*(?:°\s*)?(c|f)\b", low)
    if m:
        out["limit"] = f"{m.group(1)} {m.group(2).upper()}"
    m = re.search(r"(?:for|after|in)\s+(\d+(?:\.\d+)?)\s*(min|minutes|h|hours?|s|seconds)\b", low)
    if m:
        out["duration"] = f"{m.group(1)} {m.group(2)}"
    if re.search(r"sealed|enclosed|closed box|no airflow|inside a box", low):
        h, why = H_TYPICAL["still air in a sealed box"], "still air in a sealed box"
    elif re.search(r"\bfan|forced air|blower", low):
        h, why = H_TYPICAL["a small fan"], "a small fan"
    elif re.search(r"water|liquid cool", low):
        h, why = H_TYPICAL["water cooling"], "water cooling"
    else:
        h, why = H_TYPICAL["natural convection in open air"], "natural convection in open air"
    out["film"] = f"{h} W/m2K @ all"
    guesses.append(f"cooling: {why}, about {h:g} W/(m2 K) on every face (override with --film)")
    if "ambient" not in out:
        guesses.append("air temperature not given: assumed 25 C (override with --ambient)")
    out["guesses"] = guesses
    return out


def solve_thermal_tool(path, material, power="", heat_faces="", film="", ambient="25 C", fixed_temp="",
                       duration="", limit="", radiation=0.0) -> dict:
    from ..solvers.locate import wobbl_home

    try:
        st = study_from(path, material, power, heat_faces, [film] if film else [], ambient,
                        [fixed_temp] if fixed_temp else [], duration, limit, radiation)
        out = wobbl_home() / "results" / f"{Path(path).stem}-heat-{uuid.uuid4().hex[:6]}"
        o = run_thermal(st, out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"verdict": o.verdict, "headline": o.headline, "max_temperature_C": round(o.values["t_max_C"], 3),
            "hottest_point_mm": [round(v, 3) for v in o.values["hot_xyz"]], "limit_C": o.values["limit_C"],
            "time_over_limit_s": o.values.get("time_over_limit_s"),
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir}


__all__ = ["study_from", "read_thermal_question", "solve_thermal_tool", "Film"]
