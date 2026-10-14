"""Flow numbers read from a plain question, like '30 L/min of water through 40 m of 25 mm pipe'.

Each reader returns the quantity as the user wrote it (number and unit), so it can be recorded and passed on to
the studies, which turn it into SI. Lengths are told apart by the word that follows them: a length before 'pipe',
'duct', 'bore' or 'sphere' is a size, one before 'of', 'long' or 'run' is a length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM = r"(\d+(?:[.,]\d+)?)"
FLOW = r"(?:l/min|lpm|l/s|l/h|ml/min|gpm|gal/min|gph|cfm|ft3/min|m3/h|m³/h|m3/s|m³/s)"
SPEED = r"(?:m/s|km/h|kph|kmh|mph|knots?|kn|kts?|ft/s|fps|ft/min|fpm|lfm)"
LENGTH = r"(?:mm|cm|m|in|inch|inches|ft|feet|foot)"
PRESSURE = r"(?:pa|kpa|mpa|bar|mbar|psi|inh2o|in\s*h2o|mmh2o|mm\s*h2o)"
POWER = r"(?:kw|w|watts?)"
SIZE_WORDS = r"(?:pipes?|ducts?|hoses?|tubes?|tubing|bore|diameter|dia\b|sphere|ball|cylinder|disk|disc|plate|cube)"


@dataclass
class Read:
    flow: str = ""
    speed: str = ""
    size: str = ""            # a diameter or side
    length: str = ""
    fluid: str = ""
    temp: str = ""
    pressure: str = ""        # a pressure limit or what a pump gives
    power: str = ""
    mass: str = ""
    rotors: str = ""
    fan: str = ""             # 'Q, P' when a fan's free flow and shut off pressure are both given
    mode: str = ""            # internal or external, from 'through' and 'around'
    found: list = field(default_factory=list)


FLUID_WORDS = [("seawater", r"sea\s*water|salt\s*water|\bocean\b|\bsea\b"),
               ("iso-vg-46", r"hydraulic\s+oil|iso\s*vg\s*46"), ("iso-vg-32", r"iso\s*vg\s*32"),
               ("iso-vg-68", r"iso\s*vg\s*68"), ("engine-oil", r"engine\s+oil|motor\s+oil|\boil\b"),
               ("ethylene-glycol", r"glycol|antifreeze"), ("glycerin", r"glycerin|glycerol"),
               ("water", r"\bwater\b|coolant"), ("air", r"\bair\b|\bwind\b|airflow|\bfan\b|\bcfm\b")]


def read(text: str) -> Read:
    t = (text or "").lower().replace("³", "3")
    r = Read()

    def grab(rx, name):
        m = re.search(rx, t)
        if m:
            setattr(r, name, re.sub(r"\s+", " ", m.group(0).strip()).replace(",", "."))
            r.found.append(name)
        return m

    grab(rf"{NUM}\s*{FLOW}", "flow")
    grab(rf"{NUM}\s*{SPEED}(?![a-z])", "speed")
    grab(rf"{NUM}\s*{POWER}(?![a-z/])", "power")
    grab(rf"{NUM}\s*{PRESSURE}(?![a-z])", "pressure")
    grab(rf"{NUM}\s*kg\b", "mass")
    m = re.search(rf"{NUM}\s*(?:rotors?|motors?|props?\b)", t)
    if m:
        r.rotors = m.group(1)
    m = re.search(rf"(?:at|of|in)\s+(-?{NUM})\s*(?:°\s*)?c\b", t)
    if m:
        r.temp = f"{m.group(1)} C"
    for m in re.finditer(rf"{NUM}\s*({LENGTH})\b\s*(?:-|\s)?\s*(\w+)?", t):
        val = f"{m.group(1)} {m.group(2)}"
        after = t[m.end(2):m.end(2) + 24]
        before = t[max(0, m.start() - 14):m.start()]
        if re.match(rf"\s*(?:-|\s)?\s*(?:inner\s+|inside\s+|internal\s+)?{SIZE_WORDS}", after) or \
                re.search(r"diameter\s*(?:of)?\s*$|\bd\s*=\s*$|bore\s*(?:of)?\s*$", before):
            r.size = r.size or val
        elif re.match(r"\s*(?:of|long|run|length)\b", after) or re.search(r"length\s*(?:of)?\s*$", before):
            r.length = r.length or val
    if r.flow and r.pressure and re.search(r"\bfan\b|\bpump\b|blower", t):
        r.fan = f"{r.flow}, {r.pressure}"
    for key, rx in FLUID_WORDS:
        if re.search(rx, t):
            r.fluid = key
            break
    if re.search(r"\bthrough\b|\binside\b|pressure drop|\bpump\b|flow rate|coolant|channel|manifold|passage", t):
        r.mode = "internal"
    if re.search(r"\baround\b|\bdrag\b|\bwind\b|\blift\b|\bstream\b|\bpast\b|\bacross\b", t):
        r.mode = r.mode or "external"
    return r
