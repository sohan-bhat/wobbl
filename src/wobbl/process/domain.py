"""Manufacturing analyses of a CAD part for wobbl simulate: casting, mold filling and 3D print warping."""
from __future__ import annotations

import re

from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS

TEMP = r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(c|f|k)\b"


def _named(table: dict, words: str) -> str | None:
    """The longest key or alias of a data table that the words name."""
    low = words.lower()
    hits = []
    for key, item in table.items():
        for name in (key, *getattr(item, "aliases", ())):
            if len(name) >= 2 and re.search(r"(?<![\w-])" + re.escape(name.replace("-", " ")).replace(r"\ ", r"[\s-]")
                                            + r"(?![\w-])", low):
                hits.append((len(name), key))
    return max(hits)[1] if hits else None


def _temp_after(words: str, pattern: str) -> str | None:
    m = re.search(pattern + r"\D{0,12}?" + TEMP, words.lower())
    return f"{m.group(1)} {m.group(2).upper()}" if m else None


def casting(job):
    from .casting import run_casting
    from .data import ALLOYS, MOLDS
    from .tools import casting_study

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    alloy = job.value("alloy") or _named(ALLOYS, job.value("material") or "") or _named(ALLOYS, job.focus)
    if alloy:
        job.use("alloy", alloy)
    else:
        alloy = job.value("alloy", "a356", "casting alloy not mentioned")
    mold = job.value("mold") or _named(MOLDS, job.focus) or "sand"
    job.use("mold", mold)
    pour = job.value("pour") or _temp_after(job.focus, r"pour(?:ed|ing)?")
    if pour:
        job.use("pour", pour)
    risers = job.values("riser")
    if not risers:
        job.assume("no riser given, so Wobbl places risers over the hot spots by the modulus rule and checks them",
                   "--riser 'd=40 h=60 at X,Y,Z' or --riser none")
    chills = job.values("chill")
    quality = "fast" if job.quick else (job.value("quality") or "normal")
    st = casting_study(str(job.path), alloy, risers, chills, mold, pour or "", job.value("mold-temp") or "",
                       job.question, quality, dict(p.split("=", 1) for p in job.values("param") if "=" in p),
                       job.number("mesh-size"))
    job.situate(environment={"process": "sand casting", "alloy": alloy, "pour": pour or "the alloy's usual"},
                objects=["sand mold"] + [f"riser {r}" for r in risers if r.lower() not in ("none", "auto")] +
                [f"chill {c}" for c in chills])
    job.ready()
    o = run_casting(st, job.out, job.log, keep=job.flag("keep"))
    if o.values.get("porosity_max_pct") is not None:
        job.criterion("porosity", "porosity_max_pct", 1.0, "max")
    return o


register(Analysis(
    "part", "casting", "Sand casting: where the metal freezes last, riser feeding and shrinkage porosity",
    casting, dry=True, kinds=PART_KINDS, legacy="wobbl cast", covers=("heat",),
    keywords=(r"\bcast(?:ing|s|ed)?\b", r"\briser", r"\bshrink", r"\bporosit", r"\bsand\b", r"\bfoundry",
              r"\bsolidif", r"\bpour", r"\bhot spot", r"\bniyama", r"\bchill"),
    aliases=("cast", "solidification", "foundry"),
    options=(Option("alloy", "Casting alloy: a356 or pure-al", "a356"),
             Option("riser", "Riser, e.g. 'd=40 h=60 at 50,0,30' (mm, its foot on the casting), or none",
                    multiple=True),
             Option("chill", "Chill block against the casting, e.g. 'steel 30x30x15 at 50,0,-10'", multiple=True),
             Option("mold", "Mold material", "sand"),
             Option("pour", "Pouring temperature, e.g. '720 C' (default the alloy's usual)"),
             Option("mold-temp", "Starting mold temperature", "25 C"),
             Option("keep", "Keep the meshes and scratch files", flag=True)),
    example='wobbl simulate bracket.step "will this cast in A356 without shrinkage porosity?"'))


def molding(job):
    from .data import RESINS
    from .molding import MoldingStudy, parse_gate, run_molding

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    resin = job.value("resin") or _named(RESINS, job.value("material") or "") or _named(RESINS, job.focus)
    if resin:
        job.use("resin", resin)
    else:
        resin = job.value("resin", "gpps", "resin not mentioned")
    gates = job.values("gate")
    if not gates:
        job.assume("no gate given, so Wobbl puts one where the melt has the shortest way to the farthest corner",
                   "--gate 'at X,Y,Z'")
    melt = job.value("melt") or _temp_after(job.focus, r"melt")
    mold = job.value("mold-temp") or _temp_after(job.focus, r"mold")
    fill = job.value("fill-time")
    m = None if fill else re.search(r"fill(?:s|ed|ing)?\s+(?:in|within)\s+(\d+(?:\.\d+)?)\s*s\b", job.focus.lower())
    if m:
        fill = m.group(1)
    if fill:
        job.use("fill-time", fill)
    p_max = job.number("max-pressure") or 150.0
    if not job.value("max-pressure"):
        job.assume("machine pressure limit not given, so Wobbl assumed 150 MPa", "--max-pressure 180")
    from ..units import parse_quantity

    def temp(t):
        return parse_quantity(t, "temperature").value if t else None

    st = MoldingStudy(str(job.path), resin, temp(melt), temp(mold), float(fill) if fill else None,
                      [parse_gate(g) for g in gates], float(p_max), job.question,
                      "fast" if job.quick else (job.value("quality") or "normal"),
                      params=dict(p.split("=", 1) for p in job.values("param") if "=" in p),
                      mesh_size=job.number("mesh-size"))
    job.situate(environment={"process": "injection molding", "resin": resin, "machine_MPa": p_max},
                objects=["mold", "gate"])
    job.ready()
    o = run_molding(st, job.out, job.log, keep=job.flag("keep"))
    job.criterion("pressure", "pressure_hot_MPa", p_max, "max")
    return o


register(Analysis(
    "part", "molding", "Injection molding fill: fill time, pressure against the machine, weld lines, short shot",
    molding, dry=True, kinds=PART_KINDS, covers=("heat",),
    keywords=(r"\bmou?ld(?:ed|ing|s)?\b", r"\binjection", r"\bshort shot", r"\bweld line", r"\bgate\b",
              r"\bfill(?:s|ed|ing)?\b", r"\bflow front", r"\bresin\b"),
    aliases=("mold", "fill", "injection"),
    options=(Option("resin", "Resin: gpps, pc-oq1028 or custom:n=..,tau=..,d1=..,d2=..,a1=..", "gpps"),
             Option("gate", "Gate, e.g. 'at 40,0,2 d=1.5' (mm, on the surface)", multiple=True),
             Option("melt", "Melt temperature (default the resin's usual)"),
             Option("mold-temp", "Mold temperature (default the resin's usual)"),
             Option("fill-time", "Fill time in seconds (default a 100 mm/s flow front)"),
             Option("max-pressure", "The machine's injection pressure limit, MPa", "150"),
             Option("keep", "Keep the meshes and scratch files", flag=True)),
    example='wobbl simulate lid.step "will this lid short shot in polystyrene?"'))
