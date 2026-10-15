"""Mechanism studies from plain questions and MCP calls."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from .motors import CATALOG, _ALIASES

_MASS = r"(\d+(?:\.\d+)?)\s*(kg|g|grams?|lbs?|oz)\b"
_LEN = r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch(?:es)?)\b"


def _servo_names(low: str) -> list[str]:
    names = sorted(set(CATALOG) | set(_ALIASES), key=len, reverse=True)
    found = []
    for n in names:
        if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", low):
            key = _ALIASES.get(n, n)
            if key not in found:
                found.append(key)
            low = low.replace(n, " ")
    return found


def _unit(m) -> str:
    u = m.group(2)
    u = {"grams": "g", "gram": "g", "lbs": "lb", "inch": "in", "inches": "in"}.get(u, u)
    return f"{m.group(1)} {u}"


KINDS = (("pendulum", r"\bpendulum\b"),
         ("slider", r"slider.?crank|crank.?slider|\bpiston\b|connecting rod|scotch yoke"), ("fourbar", r"four.?bar|linkage|wiper|\bcrank\b|\brocker\b"),
         ("gears", r"\bgears?\b|gearbox|gear train|\bteeth\b|gear ratio"),
         ("tip", r"tip(?:s|ping)? over|topple|fall over|knock(?:ed)? over|\bstable\b"),
         ("impact", r"\bimpact\b|collid|collision|\bhits?\b|dropped onto|drop onto|peak force"),
         ("arm", r"\bservos?\b|robot arm|\barm\b|pick and place|\blift\b"))


def template_of(q: str) -> str | None:
    """Which ready made mechanism a description or question is about, if any."""
    low = (q or "").lower()
    return next((k for k, rx in KINDS if re.search(rx, low)), None)


def read_mechanism_question(q: str) -> tuple[str, dict, list[str]]:
    """Which template a question is about, the values it gives and the guesses for the ones it leaves out."""
    kind = template_of(q)
    if kind is None:
        raise ValueError("could not tell which mechanism this is, so give a URDF, MJCF or spec file, or say which "
                         "it is: an arm, a linkage, gears, an impact, something that tips or a pendulum")
    p, guesses = read_for(kind, q)
    return kind, p, guesses


def read_for(kind: str, q: str) -> tuple[dict, list[str]]:
    """The values a description gives for one template, and the guesses for the ones it leaves out."""
    low = (q or "").lower()
    p, guesses = {}, []
    mass = re.search(_MASS, low)
    if kind == "pendulum":
        m = re.search(_LEN, low)
        if m:
            p["length"] = _unit(m)
        a = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degrees?|°)", low)
        if a:
            p["angle"] = f"{a.group(1)} deg"
    elif kind == "fourbar":
        m = re.search(r"(\d+(?:\.\d+)?)\s*rpm", low)
        if m:
            p["speed"] = f"{m.group(1)} rpm"
        lens = re.findall(r"(\d+(?:\.\d+)?)\s*mm", low)
        if len(lens) >= 4:
            p.update(dict(zip(("ground", "crank", "coupler", "rocker"), (f"{x} mm" for x in lens[:4]))))
        else:
            guesses.append("link lengths not all given, so Wobbl assumed a 120 mm ground, 30 mm crank, 125 mm coupler "
                           "and 45 mm rocker (override with --set ground=... crank=... coupler=... rocker=...)")
        t = re.search(r"(\d+(?:\.\d+)?)\s*n\s*\*?\s*m\b", low)
        if t:
            p["load"] = f"{t.group(1)} N*m"
    elif kind == "slider":
        m = re.search(r"(\d+(?:\.\d+)?)\s*rpm", low)
        if m:
            p["speed"] = f"{m.group(1)} rpm"
        lens = re.findall(r"(\d+(?:\.\d+)?)\s*mm", low)
        if len(lens) >= 2:
            p.update(dict(zip(("crank", "rod"), (f"{x} mm" for x in lens[:2]))))
        else:
            guesses.append("crank and rod lengths not both given, so Wobbl assumed a 40 mm crank and a 150 mm rod "
                           "(override with --set crank=... rod=...)")
        f = re.search(r"(\d+(?:\.\d+)?)\s*n\b", low)
        if f:
            p["load"] = f"{f.group(1)} N"
    elif kind == "gears":
        stages = re.findall(r"(\d+)\s*[:/]\s*(\d+)", low)
        if stages:
            p["teeth"] = ",".join(f"{a}:{b}" for a, b in stages)
        else:
            guesses.append("teeth not given, so Wobbl assumed two stages of 12:36 and 10:40 (override with "
                           "--set teeth=...)")
        t = re.search(r"(\d+(?:\.\d+)?)\s*(n\s*\*?\s*m|mn\s*\*?\s*m|kg\s*\*?\s*cm)\b", low)
        if t:
            unit = {"nm": "N*m", "mnm": "mN*m", "kgcm": "kg*cm"}[t.group(2).replace(' ', '').replace('*', '')]
            p["load"] = f"{t.group(1)} {unit}"
        else:
            guesses.append("output load not given, so Wobbl assumed 0.005 N m (override with --set load=...)")
        names = _servo_names(low)
        if names:
            p["motor"] = names[0]
    elif kind == "tip":
        if mass:
            p["payload"] = _unit(mass)
        m = re.search(rf"{_LEN}\s*(?:out|reach|from the (?:center|centre|base))", low)
        if m:
            p["reach"] = _unit(m)
        f = re.search(r"(\d+(?:\.\d+)?)\s*n\b", low)
        if f:
            p["force"] = f"{f.group(1)} N"
        guesses.append("base not given, so Wobbl assumed 160 x 160 mm weighing 1200 g (override with --set base=... "
                       "base_mass=...)")
    elif kind == "impact":
        if mass:
            p["mass"] = _unit(mass)
        h = re.search(rf"(?:from|drop(?:ped)?)\s+{_LEN}", low)
        if h:
            p["drop"] = f"{h.group(1)} {h.group(2)}"
        v = re.search(r"(\d+(?:\.\d+)?)\s*(m/s|km/h|mph)", low)
        if v and not h:
            p["speed"] = f"{v.group(1)} {v.group(2)}"
        if not h and not v:
            guesses.append("speed not given, so Wobbl assumed 2 m/s (override with --set speed=... or drop=...)")
        guesses.append("contact assumed round with a 10 mm radius, steel on steel unless materials are given")
    elif kind == "arm":
        names = _servo_names(low)
        if names:
            p["servo"] = names[0]
        else:
            guesses.append("servo not named, so Wobbl assumed a TowerPro MG996R at every joint (override with "
                           "--set servo=...)")
        if mass:
            p["payload"] = _unit(mass)
        else:
            guesses.append("payload not given, so Wobbl assumed 500 g (override with --set payload=...)")
        m = re.search(rf"{_LEN}\s*(?:away|out|reach|from the base)", low) or re.search(rf"reach(?:es)? (?:of )?{_LEN}",
                                                                                        low)
        if m:
            from . import quantities as Q
            r = Q.length(_unit(m)) * 1000
            p["pick"], p["place"] = f"{r:g},0,40", f"0,{r:g},40"
        guesses.append("the move is assumed to be a pick and place that lifts 80 mm, swings a quarter turn and "
                       "lowers, 1.2 s a step (override with --set pick=... place=... move_time=...)")
    return p, guesses


def solve_mechanism_tool(template: str = "", spec: str = "", params: dict | None = None, question: str = "") -> dict:
    from ..solvers.locate import wobbl_home
    from .study import run_mechanism

    params = {str(k): v for k, v in (params or {}).items()}
    guesses = []
    out = wobbl_home() / "results" / f"mechanism-{template or 'spec'}-{uuid.uuid4().hex[:6]}"
    try:
        target = template
        if spec:
            path = Path(spec)
            if len(spec) < 1000 and "\n" not in spec and path.exists():
                target = str(path)
            else:   # the spec itself, as TOML or JSON text
                out.mkdir(parents=True, exist_ok=True)
                target = str(out / ("spec.json" if spec.lstrip().startswith("{") else "spec.toml"))
                Path(target).write_text(spec, encoding="utf-8")
        if not target:
            if not question:
                return {"error": "pass a template (arm, fourbar, gears, impact, tip, pendulum), a spec or a question"}
            target, guessed, guesses = read_mechanism_question(question)
            params = {**guessed, **params}
        o = run_mechanism(target, params, out, question)
        if guesses:
            o.assumptions = [g[0].upper() + g[1:] + "." for g in guesses] + o.assumptions
            o.write(out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.values.items() if not k.startswith("_")}, "events": o.events,
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "trajectory": str(Path(o.workdir) / "trajectory.csv"), "result_dir": o.workdir}
