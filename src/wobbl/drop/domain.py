"""Drop tests of a CAD part, for wobbl simulate."""
from __future__ import annotations

import re

from ..simulate import Analysis, Option, register
from ..simulate.intent import phrases
from ..structural.domain import PART_KINDS, material

LENGTH = r"(\d+(?:\.\d+)?)\s*(mm|cm|m|meters?|metres?|ft|foot|feet|in|inch(?:es)?)\b"


def height_in(question: str) -> str | None:
    """'a 1.2 m drop', 'dropped from 4 feet', 'falls 80 cm' as a height with its unit."""
    q = question.lower()
    for m in re.finditer(LENGTH, q):
        near = q[max(0, m.start() - 40):m.end() + 40]
        if re.search(r"drop|fall|fell|height|high|off|from", near):
            return f"{m.group(1)} {m.group(2)}"
    return None


def floor_in(question: str) -> str | None:
    from .floors import ALIASES, FLOORS

    q = question.lower()
    for word in sorted(list(FLOORS) + list(ALIASES), key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", q):
            return word
    return None


def landing_in(question: str) -> str | None:
    q = question.lower()
    if re.search(r"\bcorners?\b", q):
        return "corners"
    if re.search(r"\bedges?\b", q):
        return "edges"
    if re.search(r"\bflat\b|\bface down\b|\bon its (?:back|face|side)\b", q):
        return "flat"
    return None


def drop(job):
    from .floors import get_floor
    from .study import DropStudy, parse_height, run_drop

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    q = job.focus
    found = height_in(q)
    if found and not job.value("height"):
        job.use("height", found)
    height = job.value("height") or found or job.value("height", "1 m", "drop height not given")
    fl = job.value("floor") or floor_in(q)
    if fl:
        job.use("floor", fl)
    else:
        fl = job.value("floor", "concrete", "floor not given")
    land = job.value("orientation") or landing_in(q)
    if land:
        job.use("orientation", land)
    else:
        land = job.value("orientation", "worst", "landing not given, so every face and corner is tried")
    mat = material(job, "abs")
    num = job.number
    st = DropStudy(str(job.path), mat, height, land, fl, job.values("mass"), job.value("watch") or "",
                   num("target-sf"), "fast" if job.quick else (job.value("quality") or "normal"), q,
                   num("scale", 1.0) or 1.0, dict(p.split("=", 1) for p in job.values("param") if "=" in p),
                   num("mesh-size"), job.number("budget", 600.0) or 600.0)
    floor = get_floor(fl)
    job.situate(environment={"floor": floor.key, "floor_name": floor.name, "height_m": parse_height(height) / 1000},
                loads=[f"dropped from {height}"], objects=job.values("mass"))
    job.ready()
    o = run_drop(st, job.out, job.log, render=not job.quick)
    v = o.values
    job.situate(environment={"down": v["down"], "landing": v["orientation"],
                             "impact_speed_m_s": v["impact_speed_mm_s"] / 1000})
    job.criterion("stress", "governing_MPa", v["allowable_MPa"], "max")
    return o


register(Analysis(
    "part", "drop", "Drop onto a floor: peak stress, whether it cracks and the g it feels, worst landing found",
    drop, dry=True, kinds=PART_KINDS, default=False, covers=("stress",),
    keywords=(r"(?<!pressure )(?<!voltage )\bdrop(?:ped|s|ping)?\b", r"\bfalls? (?:off|from|onto|on)\b",
              r"\bfell\b", r"\bimpact\b", r"\bknocked off\b", *phrases("drop test")),
    aliases=("impact test", "drop test", "impact", "fall"),
    options=(Option("height", "Drop height; a bare number is meters", "1 m"),
             Option("floor", "concrete, steel, tile, stone, wood, vinyl, carpet, foam, grass, or "
                             "'layer:E=5,t=10' (MPa, mm)", "concrete"),
             Option("orientation", "worst (every face and corner, edges too at fine quality), all, faces, edges, "
                                   "corners, flat, or a direction pointing down such as -z or 1,1,-1", "worst"),
             Option("mass", "A point mass on the part, like a battery: '45 g @ 0,0,5' or '45 g @ <faces>'",
                    multiple=True),
             Option("watch", "Where the accelerometer sits: 'x,y,z' or faces (default the first mass, else the "
                             "center of mass)"),
             Option("budget", "Run time in seconds the drop may be predicted to take before its mesh is coarsened",
                    "600")),
    example='wobbl simulate phone.step "will it survive a 1.2 m drop onto tile?"'))
