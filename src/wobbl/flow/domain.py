"""Flow for wobbl simulate: pipes, drag, propellers, hulls and OpenFOAM case folders on the fluid domain, and CFD
around or through any CAD part, with or without heat, on the part domain."""
from __future__ import annotations

import re
from pathlib import Path

from ..simulate import Analysis, Option, register

PART_KINDS = ("step", "stl", "brep", "iges", "parametric")
FLUID = Option("fluid", "Fluid: air, water, seawater, engine-oil, ethylene-glycol, glycerin, iso-vg-32 to 100, or "
                        "custom:rho=..,mu=..")
TEMP = Option("temperature", "Fluid temperature, e.g. '40 C'", "20 C")


def _words(job):
    """Numbers read from this analysis's own clause, with anything it leaves out taken from the whole question."""
    from dataclasses import fields

    from .words import read
    thing = "" if job.path else job.thing
    mine = read(" ".join(x for x in (job.focus, thing) if x))
    if job.focus != job.question:
        whole = read(" ".join(x for x in (job.question, thing) if x))
        for f in fields(mine):
            if f.name != "found" and not getattr(mine, f.name):
                setattr(mine, f.name, getattr(whole, f.name))
    return mine


def _pick(job, name: str, found: str, default: str | None = None, why: str = "") -> str | None:
    """An option as given, else the number read from the question (recorded), else the default (assumed)."""
    v = job.value(name)
    if v:
        return v
    if found:
        job.use(name, found)
        return found
    return job.value(name, default, why) if default is not None else None


def _fluid(job, w, default: str) -> tuple[str, float]:
    from .fluids import qty
    fluid = _pick(job, "fluid", w.fluid, default, "fluid not named")
    temp = _pick(job, "temperature", w.temp, "20 C", "fluid temperature not given")
    return fluid, qty(temp, "temperature")


def _need(value, what: str, example: str):
    if not value:
        raise ValueError(f"the question gives no {what}. Say it like '{example}', or pass --{what.split()[0]} "
                         f"'{example}'")
    return value


# --- the fluid domain: things described in words, and OpenFOAM case folders ---------------------------------------
def pressure_drop(job):
    from .tools import build

    w = _words(job)
    fluid, temp = _fluid(job, w, "water")
    params = {"flow": _need(_pick(job, "flow", w.flow), "flow rate", "30 L/min"),
              "diameter": _need(_pick(job, "diameter", w.size), "diameter of the pipe", "25 mm"),
              "length": _pick(job, "length", w.length, "1 m", "pipe length not given"),
              "fluid": fluid, "temp": temp, "material": job.value("material") or "pvc",
              "fittings": job.value("fittings") or "", "rise": job.value("rise") or "",
              "max_dp": job.value("max-dp") or (w.pressure if w.pressure else ""), "question": job.question,
              "cfd": "off" if job.quick else (job.value("cfd") or "auto")}
    job.situate(environment={"fluid": fluid, "flow": params["flow"]},
                objects=[f"{params['length']} of {params['diameter']} pipe"])
    st, run = build("pipe", params)
    job.ready()
    o = run(st, job.out)
    if st.max_dp:
        job.criterion("pressure drop", "dp_Pa", st.max_dp, "max")
    return o


def drag(job):
    from .tools import build

    w = _words(job)
    fluid, temp = _fluid(job, w, "air")
    shape = job.value("shape") or next((s for s in ("sphere", "cylinder", "disk", "square-plate", "cube")
                                        if s.split("-")[0] in (job.focus + " " + job.thing).lower()), None)
    if not shape:
        shape = job.value("shape", "sphere", "shape not named")
    job.use("shape", shape)
    params = {"shape": shape, "size": _need(_pick(job, "size", w.size), "size of the body", "22 cm"),
              "speed": _need(_pick(job, "speed", w.speed), "speed", "20 m/s"), "fluid": fluid, "temp": temp,
              "length": job.value("length") or "", "max_drag": job.value("max-drag") or "",
              "question": job.question}
    job.situate(environment={"fluid": fluid, "speed": params["speed"]}, objects=[f"{params['size']} {shape}"])
    st, run = build("drag", params)
    job.ready()
    return run(st, job.out)


def thrust(job):
    from .tools import build

    w = _words(job)
    m = re.search(r"\b(apcs?f?_[\w.]+|\d+(?:\.\d+)?x\d+(?:\.\d+)?)\b", f"{job.thing} {job.focus}".lower())
    prop = job.value("prop") or (m.group(1) if m else None) or job.value("prop", "apcsf_10x4.7", "propeller not named")
    job.use("prop", prop)
    params = {"prop": prop, "mass": _pick(job, "mass", w.mass), "rotors": _pick(job, "rotors", w.rotors),
              "thrust": job.value("thrust"), "rpm": job.value("rpm"), "speed": _pick(job, "speed", w.speed),
              "max_rpm": job.value("max-rpm"), "question": job.question}
    if not (params["mass"] or params["thrust"] or params["rpm"]):
        raise ValueError("say what the propeller must do: a mass to hover like '1.2 kg', a thrust or an rpm")
    if params["mass"] and not params["rotors"]:
        params["rotors"] = job.value("rotors", "4", "number of rotors not given")
    job.situate(environment={"fluid": "air"}, objects=[f"{params.get('rotors') or 1} x {prop} propeller"])
    st, run = build("prop", {k: v for k, v in params.items() if v})
    job.ready()
    return run(st, job.out)


def resistance(job):
    from .tools import build

    w = _words(job)
    length = _need(_pick(job, "length", w.length or _first_length(job.focus)), "hull length", "12 m")
    params = {"length": length, "speed": _need(_pick(job, "speed", w.speed), "speed", "8 knots"),
              "beam": job.value("beam"), "draft": job.value("draft"), "displacement": job.value("displacement"),
              "power": _pick(job, "power", w.power), "question": job.question}
    from .fluids import qty
    L = qty(length, "length")
    if not params["beam"]:
        params["beam"] = job.value("beam", f"{0.3 * L:.3g} m", "beam not given, a typical 0.3 of the length")
    if not params["draft"]:
        params["draft"] = job.value("draft", f"{0.075 * L:.3g} m", "draft not given, a typical 0.075 of the length")
    if not params["displacement"]:
        vol = 0.5 * L * qty(params["beam"], "length") * qty(params["draft"], "length")
        params["displacement"] = job.value("displacement", f"{vol:.3g} m3",
                                           "displacement not given, a block coefficient of 0.5")
    job.situate(environment={"fluid": "seawater", "speed": params["speed"]}, objects=[f"{length} hull"])
    st, run = build("hull", {k: v for k, v in params.items() if v})
    job.ready()
    return run(st, job.out)


def _first_length(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|ft|feet|foot)\b", (text or "").lower())
    return f"{m.group(1)} {m.group(2)}" if m else ""


def case(job):
    from .casefolder import run_case
    if job.path is None or not (job.path / "system" / "controlDict").exists():
        raise ValueError(f"{job.thing} is not an OpenFOAM case folder (it has no system/controlDict)")
    job.ready()
    return run_case(job.path, job.out, job.log, quick=job.quick)


# --- the part domain: any CAD geometry ----------------------------------------------------------------------------
def _part_study(job):
    from .fluids import qty
    from .partflow import PartFlowStudy

    if job.path is None:
        raise ValueError(f"file not found: {job.thing}")
    w = _words(job)
    fluid, temp = _fluid(job, w, "air")
    speed = _pick(job, "speed", w.speed)
    flow = _pick(job, "flow", w.flow)
    curve = job.value("fan") or job.value("pump") or w.fan
    pressure = job.value("pressure")
    mode = job.value("mode") or ""
    turb = job.value("turbulence")
    if not turb and re.search(r"vortex|vortices|shedding|unsteady|\bles\b|\bdes\b|eddy|buffet|flutter|"
                              r"\bwake\b.*\b(?:swing|oscillat)", job.focus.lower()):
        turb = "des"
        job.use("turbulence", turb)
    if not mode and w.mode == "internal" and speed and not flow:
        mode = "internal"
    if not (speed or flow or curve or pressure):
        speed = job.value("speed", "10 m/s" if fluid == "air" else "1 m/s", "how fast the fluid moves not given")
    st = PartFlowStudy(str(job.path), fluid, temp, mode or "auto", qty(speed, "velocity") if speed else 0.0,
                       _vector(job.value("direction") or "+x"), _vector(job.value("up")) if job.value("up") else None,
                       qty(flow, "flowrate") if flow else 0.0, inlet=job.value("inlet") or "",
                       outlet=job.value("outlet") or "",
                       params=dict(p.split("=", 1) for p in job.values("param") if "=" in p),
                       quality=job.value("quality") or ("fast" if job.quick else "normal"),
                       turbulence=turb or "auto", budget_minutes=job.number("budget-minutes", 60.0),
                       max_dp=qty(job.value("max-dp"), "pressure") if job.value("max-dp") else 0.0,
                       max_force=qty(job.value("max-force"), "force") if job.value("max-force") else 0.0,
                       keep=job.flag("keep"), question=job.question)
    if curve:
        st.curve = _curve(curve)
    elif pressure:
        p = qty(pressure, "pressure")
        st.curve = ((0.0, p), (1.0, p))
    job.use("quality", st.quality)
    job.situate(environment={"fluid": fluid, "temperature": f"{temp:g} C", "speed": speed, "flow": flow,
                             "fan": curve, "direction": job.value("direction") or "+x",
                             "inlet": st.inlet or "the largest opening"},
                objects=[Path(job.thing).name])
    return st


def _vector(text: str):
    t = (text or "").strip().lower()
    axes = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}
    m = re.match(r"^([+-]?)([xyz])$", t)
    if m:
        v = axes[m.group(2)]
        return tuple(-c if m.group(1) == "-" else c for c in v)
    parts = [float(x) for x in re.split(r"[,\s]+", t) if x]
    if len(parts) != 3:
        raise ValueError(f"a direction is like '+x', '-z' or '1,0,0', not '{text}'")
    return tuple(parts)


def _curve(text: str):
    """'10 CFM, 20 Pa' for a fan's free flow and shut off pressure, or 'Q1 P1; Q2 P2; ...' points."""
    from .fluids import qty
    pts = [x.strip() for x in re.split(r";", text) if x.strip()]
    if len(pts) == 1:
        a, b = [x.strip() for x in pts[0].split(",")]
        return ((0.0, qty(b, "pressure")), (qty(a, "flowrate"), 0.0))
    out = []
    for pt in pts:
        a, b = [x.strip() for x in re.split(r",|\s+@\s+", pt)][:2]
        out.append((qty(a, "flowrate"), qty(b, "pressure")))
    return tuple(out)


def _plan(job, st):
    """Read the part and decide the case, recording each guess the plan makes, before anything is meshed."""
    from .partflow import plan_case
    from .shape import load

    job.log("geometry", f"reading {Path(st.part).name}")
    g = load(st.part, job.out / "work" / "geometry", st.params)
    p = plan_case(st, g)
    overrides = {"inlet": "--inlet 'opening N'", "leaves": "--inlet 'opening N'", "duct": "--mode external",
                 "stream runs": "--direction +x", "fluid itself": "--mode internal"}
    for why in p.why:
        job.assume(why.rstrip("."), next((v for k, v in overrides.items() if k in why), ""))
    job.use("mode", p.mode)
    return g


def cfd(job):
    from .partstudy import run_partflow

    st = _part_study(job)
    g = _plan(job, st)
    job.ready()
    o = run_partflow(st, job.out, job.log, geometry=g)
    if st.max_dp:
        job.criterion("pressure drop", "dp_Pa", st.max_dp, "max")
    if st.max_force:
        job.criterion("drag", "drag_N", st.max_force, "max")
    return o


def cooling(job):
    from .cooling import run_cooling

    st = _part_study(job)
    r = _words(job)
    power = job.value("power") or r.power
    if not power:
        raise ValueError("the question has no heat load I can read. Say it like '15 W', or pass --power '15 W'")
    job.use("power", power)
    from ..structural.domain import material
    mat = material(job, "6061-t6")
    from ..thermal.tools import read_thermal_question
    limit = job.value("limit") or read_thermal_question(job.question).get("limit", "")
    if limit:
        job.use("limit", limit)
    g = _plan(job, st)
    job.ready()
    o = run_cooling(st, job.out, power, job.value("heat-faces") or "", mat, limit, job.log, geometry=g)
    if o.values.get("limit_C") is not None:
        job.criterion("temperature", "t_max_C", o.values["limit_C"], "max")
    return o


FLOW_OPTIONS = (FLUID, TEMP)
CAD_OPTIONS = (
    FLUID, TEMP,
    Option("speed", "Free stream speed for flow around the part, e.g. '10 m/s'"),
    Option("flow", "Flow rate through the part or its duct, e.g. '30 L/min' or '20 CFM'"),
    Option("fan", "Fan curve as free flow and shut off pressure, '20 CFM, 30 Pa', or points 'Q P; Q P; ...'"),
    Option("pump", "Pump curve, given like a fan curve"),
    Option("pressure", "Pressure difference available across the part, e.g. '20 kPa'"),
    Option("direction", "Direction the stream comes from and goes to, e.g. '+x' or '0,1,0'", "+x"),
    Option("up", "Up direction for lift, e.g. '+z'"),
    Option("inlet", "Inlet opening: 'opening 2', 'near x,y,z' or a face selector", "the largest opening"),
    Option("outlet", "Outlet faces of a fluid volume, as a face selector"),
    Option("mode", "external, internal, volume or ducted; worked out from the part and the question when left out"),
    Option("turbulence", "auto (steady RANS, or laminar when slow), laminar, rans, or des and les for a time "
                         "accurate run that resolves the wake's eddies and its shedding frequency", "auto"),
    Option("budget-minutes", "Longest predicted time a des or les run may take before Wobbl offers RANS instead, "
                             "0 for no limit", "60"),
    Option("max-dp", "Most pressure drop allowed, e.g. '5 kPa'"),
    Option("max-force", "Most drag allowed, e.g. '20 N'"),
    Option("keep", "Keep the OpenFOAM case folders", flag=True))

register(Analysis(
    "fluid", "pressure-drop", "Pressure drop through a pipe or duct with its fittings, on the Moody chart and checked "
    "by CFD", pressure_drop, kinds=("pipe",), default=True, legacy="wobbl flow pipe",
    keywords=(r"pressure drop", r"pressure loss", r"head loss", r"\bpipes?\b", r"\bducts?\b", r"\bhoses?\b",
              r"flow rate"),
    aliases=("pipe", "duct", "pipe-flow"),
    options=FLOW_OPTIONS + (Option("flow", "Flow rate, e.g. '30 L/min'"), Option("diameter", "Bore, e.g. '25 mm'"),
                            Option("length", "Pipe length", "1 m"), Option("material", "Wall material", "pvc"),
                            Option("fittings", "e.g. '3 elbow90, gate-valve'"), Option("rise", "Outlet height"),
                            Option("max-dp", "Pressure the supply can give"),
                            Option("cfd", "auto runs the CFD check of the friction factor, off skips it", "auto")),
    example='wobbl simulate "40 m of 25 mm pipe carrying 30 L/min of water, what is the pressure drop?"', dry=True))

register(Analysis(
    "fluid", "drag", "Drag on a sphere, cylinder, disk, plate or cube in a stream, from published drag curves", drag,
    kinds=("drag",), default=True, legacy="wobbl flow drag",
    keywords=(r"\bdrag\b", r"wind load", r"terminal"), aliases=("drag-force",),
    options=FLOW_OPTIONS + (Option("shape", "sphere, cylinder, disk, square-plate or cube"),
                            Option("size", "Diameter or side"), Option("speed", "Stream speed"),
                            Option("length", "Span of a cylinder"), Option("max-drag", "Most drag allowed")),
    example='wobbl simulate "what is the drag on a 22 cm sphere at 20 m/s?"', dry=True))

register(Analysis(
    "fluid", "thrust", "Propeller thrust, power and hover from measured propeller data and blade element theory",
    thrust, kinds=("prop",), default=True, legacy="wobbl flow prop",
    keywords=(r"\bhover", r"\bthrust\b", r"propeller", r"\bprops?\b", r"\bdrones?\b", r"\bquad"),
    aliases=("propeller", "prop", "hover"),
    options=(Option("prop", "UIUC propeller name or diameter x pitch in inches", "apcsf_10x4.7"),
             Option("mass", "Mass to hover"), Option("rotors", "Number of rotors", "4"), Option("thrust", "Thrust"),
             Option("rpm", "Speed"), Option("speed", "Forward speed"), Option("max-rpm", "Top speed")),
    example='wobbl simulate apcsf_10x4.7 "can a 1.2 kg quad with 4 rotors hover?"', dry=True))

register(Analysis(
    "fluid", "resistance", "Hull resistance and power by Holtrop and Mennen, with its range checked", resistance,
    kinds=("hull",), default=True, legacy="wobbl flow hull",
    keywords=(r"\bhulls?\b", r"\bboats?\b", r"\bresistance\b", r"\bknots?\b"), aliases=("hull", "boat"),
    options=(Option("length", "Waterline length"), Option("beam", "Beam"), Option("draft", "Draft"),
             Option("displacement", "Displacement as mass or volume"), Option("speed", "Speed"),
             Option("power", "Installed power")),
    example='wobbl simulate "a 12 m boat hull at 8 knots, how much power does it need?"', dry=True))

register(Analysis(
    "fluid", "cfd", "Run an OpenFOAM case folder as it is, and check its convergence and mass balance", case,
    kinds=("cfd",), default=True, keywords=(r"openfoam", r"\bcase\b"), aliases=("openfoam", "case"),
    example="wobbl simulate ./cavity", dry=True))

register(Analysis(
    "part", "cfd", "Flow around or through the part with OpenFOAM: drag and lift, or pressure drop and flow split",
    cfd, kinds=PART_KINDS, legacy="wobbl flow",
    keywords=(r"\bairflow\b", r"\bdrag\b", r"pressure drop", r"pressure loss", r"\bwind\b", r"\blift\b",
              r"\bflow rate\b", r"\bcfd\b", r"\baero", r"\b\d+(?:\.\d+)?\s*(?:m/s|km/h|mph|knots?)\b",
              r"\b\d+(?:\.\d+)?\s*(?:l/min|lpm|gpm|cfm|m3/h)\b", r"\bthrough (?:it|the)\b", r"\bpump\b",
              r"\bwater flow", r"\bcoolant\b(?!.*\b\d+(?:\.\d+)?\s*(?:k?w|watts?)\b)"),
    aliases=("airflow", "flow", "aerodynamics", "fluid-flow"), options=CAD_OPTIONS,
    example='wobbl simulate manifold.step "what is the pressure drop at 20 L/min of water?"', dry=True))

register(Analysis(
    "part", "cooling", "Forced convection cooling: CFD around or through the part sets the heat transfer, CalculiX "
    "the part's temperature", cooling, kinds=PART_KINDS, covers=("heat", "cfd"), covers_scope="sentence",
    keywords=(r"\bfan\b", r"\bblower\b", r"forced (?:air|convection)", r"\bairflow\b.*\b(?:k?w|watts?)\b",
              r"\b(?:k?w|watts?)\b.*\bairflow\b", r"\bcfm\b.*\b(?:k?w|watts?)\b", r"\bcold ?plate\b",
              r"\bliquid cool", r"\bwater cool", r"\bcoolant\b.*\b(?:k?w|watts?)\b"),
    aliases=("forced-convection", "fan-cooling", "conjugate-heat"),
    options=CAD_OPTIONS + (Option("power", "Heat made in the part, e.g. '15 W'"),
                           Option("heat-faces", "Faces the heat enters through, else through the volume"),
                           Option("limit", "Temperature limit, e.g. '85 C'")),
    example='wobbl simulate heatsink.step "does a 15 W chip stay under 85 C with a 10 CFM fan?"', dry=True))
