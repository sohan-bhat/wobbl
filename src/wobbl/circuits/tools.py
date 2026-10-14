"""Circuit checks from flags, plain questions and MCP calls."""
from __future__ import annotations

import re
import uuid

from .templates import LEDS


def read_circuit_question(q: str) -> tuple[str, dict, list[str]]:
    low = q.lower()
    p, guesses = {}, []

    def grab(pattern, key):
        m = re.search(pattern, low)
        if m:
            p[key] = m.group(1).replace(" ", "")
        return m

    if re.search(r"\bled\b", low):
        grab(r"(\d+(?:\.\d+)?\s*v)\b", "supply")
        grab(r"(\d+(?:\.\d+)?\s*(?:k|m)?)\s*(?:ohm|Ω|r\b|resistor)", "resistor")
        for color in LEDS:
            if re.search(rf"\b{re.escape(color)}\b", low):
                p["led"] = color
        if "supply" not in p:
            p["supply"] = "5"
            guesses.append("supply voltage not given: assumed 5 V")
        if "resistor" not in p:
            guesses.append("no resistor given: assumed 150 ohm")
        if "led" not in p:
            guesses.append("LED color not given: assumed red (2.0 V)")
        return "led", p, guesses
    if re.search(r"ripple|buck|regulator|converter|step.?down", low):
        volts = re.findall(r"(\d+(?:\.\d+)?)\s*v\b", low)
        if len(volts) >= 2:
            p["vin"], p["vout"] = volts[0], volts[1]
        grab(r"(\d+(?:\.\d+)?)\s*a\b", "current")
        grab(r"(\d+(?:\.\d+)?\s*(?:k|m)?)hz", "frequency")
        grab(r"(\d+(?:\.\d+)?\s*[un]?)h\b", "inductor")
        grab(r"(\d+(?:\.\d+)?\s*[unp]?)f\b", "capacitor")
        for k, d in (("vin", "12"), ("vout", "5"), ("current", "1"), ("frequency", "500k"), ("inductor", "10u"),
                     ("capacitor", "22u")):
            if k not in p:
                p[k] = d
                guesses.append(f"{k} not given: assumed {d}")
        return "buck", p, guesses
    if re.search(r"mosfet|fet\b|transistor", low):
        grab(r"(\d+(?:\.\d+)?)\s*a\b", "current")
        grab(r"(\d+(?:\.\d+)?\s*m)\s*(?:ohm|Ω)", "rds_on")
        grab(r"(\d+(?:\.\d+)?\s*k?)hz", "frequency")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:c/w|°c/w)", low)
        if m:
            p["theta_ja"] = m.group(1)
        for k, d in (("current", "10"), ("rds_on", "10m")):
            if k not in p:
                p[k] = d
                guesses.append(f"{k} not given: assumed {d}")
        if "theta_ja" not in p:
            guesses.append("thermal resistance not given: assumed 62 C/W (TO-220 with no heatsink)")
        return "mosfet", p, guesses
    if re.search(r"\brc\b|time constant|charge", low):
        grab(r"(\d+(?:\.\d+)?\s*(?:k|m)?)\s*(?:ohm|Ω)", "r")
        grab(r"(\d+(?:\.\d+)?\s*[unp]?)f\b", "c")
        return "rc", p, guesses
    raise ValueError("could not tell which circuit this is; use 'wobbl circuit led|buck|mosfet|rc' or a netlist")


def solve_circuit_tool(template: str = "", netlist: str = "", params: dict | None = None,
                       question: str = "") -> dict:
    """Any circuit (netlist text or path, KiCad schematic, words) or a template, as the MCP tool runs it."""
    from ..solvers.locate import wobbl_home
    from .study import simulate_circuit

    params = dict(params or {})
    probes = list(params.pop("vectors", []) or params.pop("probes", []) or [])
    out = wobbl_home() / "results" / f"circuit-{template or 'any'}-{uuid.uuid4().hex[:6]}"
    said: list[str] = []
    try:
        o = simulate_circuit(netlist or template or "led", question, params, out, probes,
                             assume=lambda t, ov="": said.append(t + (f" (override with {ov})" if ov else "")))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}
    parts = o.values.get("parts", {})
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "parts": {k: {"status": v.get("status"), "use": v.get("use"), "power_W": v.get("p"), "tj_C": v.get("tj")}
                      for k, v in parts.items()},
            "values": {k: v for k, v in o.values.items() if not k.startswith("_") and k != "parts"},
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "guesses": said, "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "result_dir": o.workdir}
