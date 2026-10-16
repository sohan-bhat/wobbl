"""Circuits for wobbl simulate: any netlist, KiCad schematic or description, and the templates as shortcuts.

The circuit domain takes a SPICE netlist (.cir, .sp, .spice, .ckt, or .net when it is SPICE), a KiCad
schematic (.kicad_sch) or KiCad netlist export (.net that starts with '(export'), or words. Words that fit a
template exactly (an LED and its resistor, a buck converter's ripple, a MOSFET's heating, an RC time
constant) run that template, and any other description is built into a netlist with every guess recorded.

The physics are the analyses of one circuit study: ratings (every part against its limits, the default),
operating-point, transient, ac (frequency response and loop stability) and tolerance (supply range,
Monte Carlo and worst case corners). Each one still judges every part, so any of them answers whether
anything burns.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..simulate import Analysis, Domain, Option, register

TEMPLATE_KINDS = ("led", "buck", "mosfet", "rc")
MORE_PARTS = r"\b555\b|timer|op.?amp|transistor|\bnpn\b|\bpnp\b|regulator|ldo|78\d\d|1117|lm317|zener|relay|motor|" \
             r"h.?bridge|array|parallel|series|\b\d+\s+leds\b|leds\b|charger|tp4056|esp32|arduino|atmega|" \
             r"microcontroller|divider|filter|bridge|rectifier|optocoupler|comparator|battery"


def sniff(path: Path) -> str | None:
    """A .net file is a KiCad netlist export when it starts with '(export', and SPICE otherwise."""
    try:
        head = path.read_text(errors="replace")[:200].lstrip()
    except OSError:
        return None
    return "kicad" if head.startswith("(export") or head.startswith("(kicad_sch") else "netlist"


WORDS = (r"\bleds?\b", r"\bresistors?\b", r"\bripple\b", r"\bmosfets?\b", r"\bregulators?\b", r"\bbuck\b",
         r"\bconverter\b", r"\bcapacitors?\b", r"\bcircuits?\b", r"\b\d+(?:\.\d+)?\s*v\b", r"\bvolts?\b", r"\bamps?\b",
         r"\bohms?\b|Ω", r"\bnetlist\b", r"\btransistors?\b", r"\bdiodes?\b", r"\bop.?amps?\b", r"\btime constant\b",
         r"\brc\b", r"\bfry\b", r"\b555\b", r"\bkicad\b", r"\bschematic\b", r"\bzener\b", r"\brelays?\b",
         r"\bh.?bridge\b", r"\b(?:lm|l)?78\d\d\b", r"\bams1117\b", r"\blm317\b", r"\besp32\b", r"\barduino\b",
         r"\bbreadboard\b", r"\bpcb\b", r"\bbattery\b.*\b(?:led|resistor|circuit)",
         r"\b(?:lm358|lm324|tl07\d|tl08\d|ne5532|lm741|ua741|mcp600\d|lm393|lm339)\b", r"\bamplifier\b",
         r"non.?inverting|\binverting\b", r"\bcomparators?\b", r"(?:low|high|band).?pass", r"\brectifier\b",
         r"\boptocoupler\b|\bpc817\b", r"\bgpio\b", r"microcontroller|\bmcu\b", r"\bcharger\b|\btp4056\b",
         r"pull.?up|pull.?down", r"\bne555\b|\btimer ic\b", r"\bnpn\b|\bpnp\b|2n2222|2n3904|irlz44|irf540")

register(Domain(
    "circuit", "An electronic circuit: a SPICE netlist, a KiCad schematic, or a description in words",
    extensions={".cir": "netlist", ".sp": "netlist", ".spice": "netlist", ".ckt": "netlist", ".net": "",
                ".kicad_sch": "kicad"},
    sniff=sniff, words=WORDS,
    kinds={"led": r"\bleds?\b", "buck": r"ripple|\bbuck\b|regulator|converter|step.?down",
           "mosfet": r"mosfet|\bfet\b|transistor", "rc": r"\brc\b|time constant"},
    options=(Option("set", "A part value or rating (R1=470, R1.power=1W, C3.voltage=25V, U1.PB5=high) or a template "
                           "value (supply=5V)", multiple=True),
             Option("probe", "A signal to report, e.g. 'v(out)'", multiple=True),
             Option("ambient", "Air temperature around the board", "25 C"),
             Option("parasitics", "Board parasitics: a JSON file or a board result folder")),
    results=("circuit",), priority=1,
    example='wobbl simulate board.kicad_sch "will anything overheat at 12 V?"'))


def template_fit(kind: str, text: str) -> bool:
    """Words that fit a template exactly go to it; anything with more parts in it is built as a circuit."""
    t = text.lower()
    if kind not in TEMPLATE_KINDS or re.search(MORE_PARTS, t) and kind != "buck":
        return False
    if kind == "buck":
        return bool(re.search(r"\bbuck\b|step.?down|switching|ripple", t)) and not re.search(
            r"78\d\d|1117|lm317|\bldo\b|linear|7805", t)
    if kind == "mosfet":
        return bool(re.search(r"\d+(?:\.\d+)?\s*a\b", t)) and bool(re.search(r"m\s*(?:ohm|Ω)|rds", t))
    if kind == "rc":
        return bool(re.search(r"\d\s*(?:k|m)?\s*(?:ohm|Ω)?\b", t)) and bool(re.search(r"\d\s*[unp]?f\b", t))
    return True


def _params(job) -> dict:
    p = {}
    for s in job.values("set"):
        if "=" not in s:
            raise ValueError(f"--set needs name=value, got '{s}'")
        k, v = s.split("=", 1)
        p[k.strip()] = v.strip()
    amb = job.value("ambient")
    if amb:
        p["ambient"] = amb
    par = job.value("parasitics")
    if par:
        p["parasitics"] = par
    return p


def _template(job, kind: str, text: str, params: dict):
    from .run import run_circuit
    from .tools import read_circuit_question

    tpl, p, guesses = read_circuit_question(text)
    for g in guesses:
        key = g.split(" not given")[0].split()[0].lower()
        val = re.search(r"assumed (\S+)", g)
        job.assume(g, f"--set {key}={val.group(1)}" if val else "")
    p.update({k: v for k, v in params.items() if k not in ("ambient", "parasitics")})
    if "ambient" in params:              # the air around the part, which the MOSFET template heats into
        from .study import _ambient
        p["ambient"] = _ambient(params)
    if p:
        job.use("set", [f"{a}={b}" for a, b in p.items()])
    job.ready()
    o = run_circuit(tpl, p, job.out, job.question)
    job.situate(objects=["breadboard"], environment={"ambient": f"{p.get('ambient', 25)} C"},
                circuit={"template": tpl})
    return o


def _run(job, focus: str):
    from .study import simulate_circuit

    params = _params(job)
    text = job.thing if job.path is None and job.thing and job.thing != "-" else ""
    words = " ".join(x for x in (text, job.focus) if x)
    if job.path is None and job.kind in TEMPLATE_KINDS and template_fit(job.kind, words):
        return _template(job, job.kind, words, params)
    source = str(job.path) if job.path is not None else (text or job.focus)
    m = re.search(r"\b(?:at|from|on|with|off)\s+(\d+(?:\.\d+)?)\s*v\b", job.focus.lower())
    if job.path is not None and m and "supply" not in params:
        params["supply"] = m.group(1)       # "will it overheat at 12 V" sets an input the board leaves open
        job.use("set", [*job.values("set"), f"supply={m.group(1)}"])
    if "ambient" not in params:
        params["ambient"] = job.value("ambient", "25 C", "air temperature around the board not given")
    o = simulate_circuit(source, job.focus, params, job.out, job.values("probe"), focus, quick=job.quick,
                         log=job.log, assume=lambda t, ov="": job.assume(t, ov), ready=job.ready)
    board = "pcb" if job.kind == "kicad" or source.endswith(".kicad_sch") else "breadboard"
    parts = o.values.get("parts", {})
    hot = sorted(parts.values(), key=lambda d: -(d.get("use") or 0))[:3]
    job.situate(objects=[board], environment={"ambient": f"{params['ambient']}"},
                circuit={"playback": "playback.json", "parts": len(parts), "verdict": o.verdict,
                         "hottest": [{"ref": d["ref"], "use": d.get("use"), "tj": d.get("tj")} for d in hot]})
    for ref, d in parts.items():
        for it in d.get("items", []):
            if it.get("limit") and it.get("use") is not None and it.get("status") != "info":
                job.criterion(f"{ref} {it['what']}", f"{ref}_{it['what'].replace(' ', '_')}", it["limit"], "max")
    return o


def ratings(job):
    return _run(job, "ratings")


def operating_point(job):
    return _run(job, "operating-point")


def transient(job):
    return _run(job, "transient")


def ac(job):
    return _run(job, "ac")


def tolerance(job):
    return _run(job, "tolerance")


KINDS = ("netlist", "kicad", "words", *TEMPLATE_KINDS)
register(Analysis(
    "circuit", "ratings", "Every part against its voltage, current, power and temperature limits, and whether the "
                          "circuit works", ratings, kinds=KINDS, default=True, legacy="wobbl circuit", dry=True,
    keywords=(r"\bfry\b|\bburn|blow (?:up|out)|smoke|overheat|\bhot\b|too hot|\bsafe\b|ratings?\b|survive|"
              r"stress|exceed|over ?current|too much current|damage|anything|warm|temperature|will it work|"
              r"does it work|work\?|dissipat|power rating",),
    aliases=("stress", "check", "parts"), covers=("operating-point",),
    example='wobbl simulate board.kicad_sch "will anything overheat at 12 V?"'))
register(Analysis(
    "circuit", "operating-point", "The steady DC voltages at every node and currents through every part",
    operating_point, kinds=KINDS, dry=True,
    keywords=(r"what is v\(|voltage (?:at|on|across)|current (?:through|in|into)|\bbias\b|operating point|"
              r"quiescent|what voltage|how much current|how many (?:milli)?amps|\bdc\b|node voltage|output voltage|"
              r"what is the output|divider",),
    aliases=("op", "dc", "bias")))
register(Analysis(
    "circuit", "transient", "The circuit over time from power on: ripple, inrush, timing, oscillation and switch off",
    transient, kinds=KINDS, dry=True,
    keywords=(r"ripple|start.?up|power.?on|inrush|transient|waveform|blink|flash|oscillat|frequency of|period|"
              r"timing|how long|charge time|rise time|time constant|over time|turn.?off|switch(?:es|ing)? off|spike|"
              r"how fast",),
    aliases=("tran", "time", "startup")))
register(Analysis(
    "circuit", "ac", "Frequency response and, where there is feedback, loop gain, phase margin and gain margin", ac,
    kinds=("netlist", "kicad", "words"), dry=True,
    keywords=(r"frequency response|bode|cut.?off|corner frequency|bandwidth|\bfilter|phase margin|gain margin|"
              r"stab(?:le|ility)|\bgain\b|-3 ?db|resonan|\bac\b",),
    aliases=("frequency", "stability", "bode", "loop")))
register(Analysis(
    "circuit", "tolerance", "Every part across the supply range and part tolerances: Monte Carlo builds and worst "
                            "case corners", tolerance, kinds=("netlist", "kicad", "words"), dry=True,
    keywords=(r"toleranc|worst.?case|monte.?carlo|spread|corners?\b|every board|batch|production|"
              r"battery (?:drains|runs down|range|gets low)|as the battery|supply range|over the range",),
    aliases=("monte-carlo", "worst-case", "corners", "sweep")))
