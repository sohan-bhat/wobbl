"""Flights for wobbl simulate: a dropped, thrown or launched body, a rocket spec, or a solid rocket motor."""
from __future__ import annotations

import re

from ..simulate import Analysis, Option, register

# regulation balls: mass, diameter and a typical drag coefficient at the speeds they fly
BALLS = {"baseball": ("145 g", "7.4 cm", 0.35), "golf ball": ("45.9 g", "4.27 cm", 0.25),
         "tennis ball": ("58 g", "6.7 cm", 0.55), "soccer ball": ("430 g", "22 cm", 0.25),
         "basketball": ("620 g", "24 cm", 0.5), "ping pong ball": ("2.7 g", "4 cm", 0.5),
         "bowling ball": ("7 kg", "21.8 cm", 0.47)}
DENSITY = {"rock": 2700, "stone": 2700, "granite": 2700, "steel": 7850, "iron": 7870, "lead": 11340,
           "aluminum": 2700, "aluminium": 2700, "wood": 600, "rubber": 1100, "glass": 2500, "ice": 917,
           "concrete": 2400, "brass": 8500, "copper": 8960}
_N = r"(\d+(?:\.\d+)?)"


def read_flight(text: str) -> dict:
    """The body and its launch from a sentence: size, mass or what it is made of, height, speed and angle."""
    low = text.lower()
    out = {}
    for name, (mass, dia, cd) in BALLS.items():
        if name in low:
            out.update(mass=mass, diameter=dia, cd=str(cd), name=name.replace(" ", "-"))
    m = re.search(rf"{_N}\s*(mm|cm|m|in|inch(?:es)?)\s*(?:across|diameter|wide|round)|diameter (?:of )?{_N}\s*"
                  rf"(mm|cm|m|in)|\ba {_N}\s*(mm|cm|in)\b", low)
    if m:
        g = [x for x in m.groups() if x]
        out["diameter"] = f"{g[0]} {g[1]}"
    m = re.search(rf"{_N}\s*(kg|g|grams?|lbs?|oz)\b", low)
    if m:
        out["mass"] = f"{m.group(1)} {m.group(2).rstrip('s') if m.group(2) != 'lbs' else 'lb'}"
    m = re.search(rf"(?:from|off|height of|above the ground|up)\s+(?:a height of\s+)?{_N}\s*(km|m|ft|feet|meters?)\b",
                  low)
    if m:
        unit = {"feet": "ft"}.get(m.group(2), "m" if m.group(2).startswith("meter") else m.group(2))
        out["height"] = f"{m.group(1)} {unit}"
    m = re.search(rf"{_N}\s*(m/s|km/h|kph|mph|ft/s)", low)
    if m:
        out["speed"] = f"{m.group(1)} {m.group(2).replace('kph', 'km/h')}"
    m = re.search(rf"{_N}\s*(?:deg|degrees?|°)", low)
    if m:
        out["angle"] = f"{m.group(1)} deg"
    for word, rho in DENSITY.items():
        if re.search(rf"\b{word}\b", low):
            out.setdefault("density", f"{rho} kg/m3")
            out.setdefault("name", word)
            break
    return out


def trajectory(job):
    from .study import run_motion
    from .tools import study_from

    wind = job.value("wind") or ""
    if job.path is not None:
        st = study_from(name=job.value("name") or "", wind=wind, question=job.question, spec=str(job.path))
    elif job.kind == "rocket":
        raise ValueError("a rocket needs its spec, with its motor, nose, fins, balance point, rail and parachutes, "
                         "as in examples/rocket-launch/rocket.toml")
    else:
        r = read_flight(" ".join(x for x in (job.thing, job.question) if x))
        for k in ("name", "mass", "density", "diameter", "cd", "height", "speed", "angle"):
            if k in r and not job.options.get(k):
                job.use(k, r[k])
        get = lambda k: job.value(k) or r.get(k, "")  # noqa: E731
        diameter = get("diameter") or job.value("diameter", "10 cm", "size not given")
        mass, density = get("mass"), get("density")
        if not mass and not density:
            density = job.value("density", "2700 kg/m3", "mass and material not given, so a solid rock")
        if mass and density and not job.options.get("density"):
            density = ""
        cd = get("cd")
        st = study_from(get("name") or "body", mass, density, diameter, float(cd) if cd else None, get("height"),
                        get("speed"), get("angle"), wind, job.question)
    job.situate(objects=["ground"], environment={"gravity": "1 g", "air": "1976 standard atmosphere",
                                                 **({"wind": wind} if wind else {})})
    return run_motion(st, job.out, job.log)


def motor(job):
    from ..chem.motor import run_motor
    from ..chem.spec import read_spec, spec_from_flags

    if job.path is not None:
        source = str(job.path)
    else:
        source = spec_from_flags(job.value("propellant") or "", job.values("grain"), job.value("throat") or "",
                                 job.value("exit") or "", job.value("casing") or "", job.number("efficiency"))
    ms = read_spec(source, job.question)
    job.situate(objects=["test stand"], environment={"ambient": "1 atm"})
    return run_motor(ms, job.out, job.log, render=not job.quick)


FLIGHT_OPTS = (Option("wind", "Wind blowing back toward the start, e.g. '5 m/s'"),
               Option("name", "Name for the report and the output folder"))

register(Analysis(
    "flight", "trajectory", "Drops, throws and rocket flights under gravity, air drag, wind, thrust and parachutes",
    trajectory, kinds=("body", "rocket"), default=True, legacy="wobbl motion",
    keywords=(r"\bfall", r"\bdrop", r"\bthrow|\bthrown", r"\blaunch", r"apogee", r"\bland", r"how (?:far|high|fast)",
              r"\brange\b", r"impact speed", r"trajector", r"stability", r"parachute", r"\bflight\b|\bfly\b"),
    aliases=("motion", "flight", "ballistics"),
    options=FLIGHT_OPTS + (
        Option("mass", "Mass of the body, e.g. '145 g'"), Option("density", "Density of a solid round body"),
        Option("diameter", "Diameter that meets the air, e.g. '9 cm'", "10 cm"),
        Option("cd", "Drag coefficient on the frontal area", "0.47"),
        Option("height", "Starting height above the ground, e.g. '300 m'"),
        Option("speed", "Launch speed, e.g. '30 m/s'"), Option("angle", "Launch angle above level", "45 deg")),
    example='wobbl simulate "a 9 cm rock dropped from 300 m"'))

register(Analysis(
    "flight", "motor", "Solid rocket motor ballistics: thrust curve, chamber pressure against the casing, Kn and "
    "erosive burning", motor, kinds=("motor",), default=True, legacy="wobbl motor",
    keywords=(r"thrust curve", r"chamber pressure", r"\bcasing\b", r"total impulse", r"motor class", r"\bburn\b",
              r"\bkn\b", r"erosive"),
    aliases=("ballistics-internal", "solid-motor"),
    options=(Option("propellant", "knsb, kndx, knsu, ocean-water or cherry-limeade"),
             Option("grain", "A grain, e.g. 'bates d=48 core=18 L=80 x4' (mm)", multiple=True),
             Option("throat", "Nozzle throat diameter, e.g. '13 mm'"), Option("exit", "Nozzle exit diameter"),
             Option("efficiency", "Nozzle efficiency", "0.90"),
             Option("casing", "Casing, e.g. '6061-t6 od=54 wall=1.6 L=400 sf=2'")),
    example="wobbl simulate motor.toml"))
