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
                       job.number("mesh-size"), job.number("budget", 600.0) or 600.0)
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


def _number_after(words: str, pattern: str, unit: str = r"mm") -> float | None:
    m = re.search(pattern + r"\D{0,16}?(\d+(?:\.\d+)?)\s*" + unit + r"\b", words.lower())
    return float(m.group(1)) if m else None


def print_warp(job):
    from ..units import parse_quantity
    from .data import FILAMENTS, get_filament
    from .printing import ENCLOSURE_C, run_print_warp
    from .slicer import find_profile, read_profile
    from .warp import WarpStudy

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    words = job.focus
    prof_path = job.value("profile")
    if not prof_path:
        found = find_profile(job.path)
        if found:
            prof_path = str(found)
            job.assume(f"print settings read from {found.name}, the slicer profile next to the part",
                       f"--profile {found.name}")
    prof = read_profile(prof_path) if prof_path else {}
    if prof_path:
        job.use("profile", prof_path)

    def temp(t):
        return parse_quantity(t, "temperature").value if t not in (None, "") else None

    fil = (job.value("filament") or _named(FILAMENTS, job.value("material") or "") or _named(FILAMENTS, words)
           or _named(FILAMENTS, str(prof.get("material", ""))))
    if fil:
        job.use("filament", fil)
    else:
        fil = job.value("filament", "pla", "plastic not named")
    f = get_filament(fil)
    bed = temp(job.value("bed") or _temp_after(words, r"bed")) or prof.get("bed_C")
    if bed is None:
        bed = f.bed
        job.assume(f"bed temperature not given, so Wobbl assumed {bed:g} C, usual for {f.name}", f"--bed '{bed:g} C'")
    chamber = temp(job.value("chamber") or _temp_after(words, r"(?:chamber|enclosure)")) or prof.get("chamber_C")
    if chamber is None and re.search(r"\benclos|\bchamber\b|\bclosed printer", words.lower()):
        chamber = ENCLOSURE_C
        job.assume(f"an enclosure with no temperature given, so Wobbl assumed {ENCLOSURE_C:g} C around the part",
                   f"--chamber '{ENCLOSURE_C:g} C'")
    ambient = temp(job.value("ambient")) or 25.0
    if chamber is None:
        job.assume(f"no enclosure mentioned, so the part cools in open air at {ambient:g} C", "--chamber '45 C'")
    build = job.value("print-direction") or ("auto" if re.search(r"\b(?:best way|which way|orient)", words.lower())
                                             else None)
    if not build:
        build = "z"
        job.assume("print direction not given, so the part prints the way it is drawn, layers along z",
                   "--print-direction auto")
    infill = job.number("infill")
    if infill is None:
        infill = prof.get("infill")
    if infill is not None and infill > 1.0:
        infill /= 100.0
    if infill is None:
        infill = 1.0
        job.assume("infill not given, so Wobbl treated the part as solid, as the calibration plates were",
                   "--infill 20")
    flat = job.number("flatness") or _number_after(words, r"(?:flat|flatness|within)")
    layer = job.number("layer-height") or prof.get("layer_height") or 0.2
    walls = prof.get("walls", 2) * (prof.get("line_width") or 0.45)
    skins = max(prof.get("top_layers", 4), prof.get("bottom_layers", 4)) * layer
    st = WarpStudy(str(job.path), f.key, build, bed, chamber, ambient, layer, infill, max(walls, skins),
                   job.number("brim") or prof.get("brim_mm") or 0.0, flat, job.question,
                   "fast" if job.quick else (job.value("quality") or "normal"), job.number("scale", 1.0) or 1.0,
                   dict(p.split("=", 1) for p in job.values("param") if "=" in p), job.number("mesh-size"),
                   prof, job.number("budget", 600.0) or 600.0)
    job.situate(environment={"printer": "enclosed" if chamber is not None else "open", "bed_C": bed,
                             "air_C": chamber if chamber is not None else ambient, "filament": f.name},
                objects=["print bed"])
    job.ready()
    o = run_print_warp(st, job.out, job.log, keep=job.flag("keep"))
    job.criterion("flatness", "flatness_mm", o.values["tolerance_mm"], "max")
    return o


register(Analysis(
    "part", "print", "3D print warping: flatness and corner lift off the bed, stress while printing", print_warp,
    dry=True, kinds=PART_KINDS, covers=("heat", "cfd"),       # a corner lifting off the bed is not aerodynamic lift
    # a printed part asked whether it holds a load is a strength question, so printing alone does not call this up
    keywords=(r"\bwarp", r"\bbed\b", r"\blift(?:s|ing)?\s+off", r"\bcurl", r"first layer", r"\bbrim\b",
              r"\bslicer\b", r"\bcorners?\s+lift", r"when (?:i |you |we )?print", r"\bprint(?:ing)? (?:it|this)\b",
              r"off the (?:print )?bed", r"\bflat(?:ness)?\b.*\bprint|\bprint.*\bflat"),
    aliases=("warp", "print-warp", "fdm"),
    options=(Option("filament", "Printing plastic: abs, asa, petg, pla or pc", "pla"),
             Option("print-direction", "Axis or vector the layers stack along, or auto to try the ways it can lie",
                    "z"),
             Option("bed", "Bed temperature (default the plastic's usual)"),
             Option("chamber", "Air temperature in an enclosure; left out, an open printer"),
             Option("ambient", "Room temperature", "25 C"),
             Option("infill", "Infill percent (default solid)"),
             Option("layer-height", "Layer height in mm", "0.2"),
             Option("brim", "Brim width in mm"),
             Option("flatness", "Flatness tolerance in mm (default ISO 2768-2 class L)"),
             Option("profile", "A Cura, Creality Print, Orca, Bambu or PrusaSlicer settings file or .3mf project"),
             Option("keep", "Keep the meshes and scratch files", flag=True)),
    example='wobbl simulate lid.step "will this warp when I print it in ABS?"'))


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
                      mesh_size=job.number("mesh-size"), budget=job.number("budget", 600.0) or 600.0)
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
