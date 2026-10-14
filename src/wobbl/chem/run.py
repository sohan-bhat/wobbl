"""One entry point for chemistry: take anything chemical and a question, work out what to compute and run it.

    from wobbl.chem import simulate
    o = simulate("caffeine", "what does its IR spectrum look like")
    o = simulate("baking soda and vinegar")
    o = simulate("copper", "how stiff is it")

prepare() reads the thing with resolve(), picks the physics from the kind of thing and the question and gathers the
conditions, and execute() runs the module that does it: molecule.py for molecules and ions, react.py for reactions,
material.py for elements and crystals, mixture.py and combustion.py for flames and propellants and runaway.py for
battery cells. wobbl simulate reaches the same through job_entry(), which records each guess on the job and stops
at job.ready() under --plan. Every run writes outcome.json, a report, images and playback.json into its folder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .molecule import Options, wants
from .resolve import Query, ResolveError, conditions, resolve

PHYSICS = ("properties", "structure", "spectrum", "thermo", "dynamics", "reaction", "equilibrium", "path", "crystal",
           "combustion", "propellant", "runaway")
MOLECULE_PHYSICS = {"properties", "structure", "spectrum", "thermo", "dynamics"}
ACCURACY = ("fast", "normal", "high")


@dataclass
class Prepared:
    thing: str
    question: str
    q: Query
    physics: str
    opts: Options
    out: Path
    options: dict = field(default_factory=dict)
    guesses: list[tuple[str, str]] = field(default_factory=list)    # (what was assumed, how to override it)


def plan(q: Query, question: str = "", physics: str | None = None) -> str:
    """The physics to run for a resolved query and a question."""
    if physics:
        if physics not in PHYSICS:
            raise ResolveError(f"no chemistry physics '{physics}'", f"Choose from {', '.join(PHYSICS)}.")
        return physics
    k = q.kind
    if k == "propellant":
        return "propellant"
    if k in ("mixture", "reaction") and wants(question, r"\bisp\b", "specific impulse", r"c\*", "nozzle", "rocket"):
        return "propellant" if q.kind == "propellant" else "combustion"
    if k == "cell":
        return "runaway"
    if k == "mixture":
        flame = wants(question, "flame", r"\bburn", "combust", "adiabatic", "how hot")
        return "equilibrium" if wants(question, "equilibrium", "composition") and not flame else "combustion"
    if k == "reaction":
        if wants(question, "barrier", "transition state", "activation", "mechanism", r"reaction path", r"\bneb\b"):
            return "path"
        if wants(question, "equilibrium", r"\byield", "conversion", r"how much .* at equilibrium", "how far"):
            return "equilibrium"
        if wants(question, "flame", "how hot", "combust") and any(t.species.formula == "O2" for t in q.reactants):
            return "combustion"
        return "reaction"
    if k in ("element", "crystal", "salt", "oxide", "metal", "network", "material"):
        if wants(question, "dynamic", r"\bmd\b", "melt", "diffus", "trajector", "liquid"):
            return "dynamics"
        return "crystal"
    if k == "molecule" and wants(question, "flame", r"\bburn", "combust", "adiabatic", "ignit"):
        return "combustion"
    if wants(question, "dynamic", r"\bmd\b", r"\bmov", "trajector", "jiggl"):
        return "dynamics"
    if wants(question, r"\bir\b", "infrared", "spectr", "vibrat", "frequenc"):
        return "spectrum"
    if wants(question, "entropy", "enthalpy", "gibbs", "heat capacity", "thermochem", r"\bcp\b", "boil", "melt",
             "heat of formation"):
        return "thermo"
    return "properties"


def prepare(thing: str, question: str = "", physics: str | None = None, *, temperature: float | None = None,
            pressure: float | None = None, solvent: str = "", phase: str = "", accuracy: str = "normal",
            out: str | Path | None = None, quick: bool = False, render: bool = True,
            options: dict | None = None) -> Prepared:
    """Everything a run needs, read and guessed, without any solving yet."""
    if accuracy not in ACCURACY:
        raise ResolveError(f"accuracy is fast, normal or high, not '{accuracy}'")
    thing = (thing or "").strip()
    if not thing:
        thing = _from_question(question)
    q = resolve(thing, temperature, pressure, solvent, phase)
    if q.temperature is None or q.pressure is None:
        _, c = conditions(question or "")
        q.temperature = q.temperature if q.temperature is not None else c.get("temperature")
        q.pressure = q.pressure if q.pressure is not None else c.get("pressure")
    phys = plan(q, question, physics)
    guesses = []
    if q.temperature is None:
        guesses.append(("temperature not given: assumed 25 C (298.15 K)", "--temperature '298.15 K'"))
    if q.pressure is None and (phys not in MOLECULE_PHYSICS or q.kind != "molecule"):
        guesses.append(("pressure not given: assumed 1 atm", "--pressure '1 atm'"))
    T = q.temperature or 298.15
    P = q.pressure or 101325.0
    out = Path(out or Path("wobbl-out") / _slug(thing))
    acc = "fast" if quick and accuracy == "normal" else accuracy
    opts = Options(temperature=T, pressure=P, solvent=q.solvent, accuracy=acc, question=question,
                   render=render and not quick)
    return Prepared(thing, question, q, phys, opts, out, dict(options or {}), guesses)


def execute(p: Prepared, log=None):
    """Run the prepared physics and return the outcome."""
    log = log or (lambda stage, msg="": None)
    q, phys, opts, out, options = p.q, p.physics, p.opts, p.out, p.options
    out.mkdir(parents=True, exist_ok=True)
    for line in q.resolved():
        log("resolved", line)
    log("physics", f"{phys} for a {q.kind}")
    if phys in MOLECULE_PHYSICS and q.kind in ("molecule", "ion"):
        return _molecule(q, phys, opts, out, log, options)
    if phys in ("reaction", "equilibrium") and q.kind == "reaction":
        from .react import run_reaction
        return run_reaction(q, opts, out, log)
    if phys == "path":
        from .path import run_path
        return run_path(q, opts, out, log)
    if phys in ("crystal", "dynamics", "thermo", "properties") and q.kind in ("element", "crystal", "salt", "oxide",
                                                                             "metal", "network", "material"):
        from .material import run_material
        return run_material(q, phys, opts, out, log, options)
    if phys == "combustion":
        if q.kind == "reaction" and not all(t.phase == "g" or t.species.phase == "g" or t.species.formula in
                                            ("O2", "N2", "H2") for t in q.reactants):
            from .react import run_reaction      # a solid or liquid burning: the reaction run has its temperature
            return run_reaction(q, opts, out, log)
        return _combustion(q, opts, out, log, options)
    if phys == "propellant":
        return _propellant(q, p.thing, opts, out, options)
    if phys == "runaway":
        return _runaway(q, p.thing, opts, out, options)
    if phys == "equilibrium" and q.kind == "mixture":
        from .mixture import run_mixture
        return run_mixture(q, opts, out, log, options, flame=False)
    if q.kind == "reaction":
        raise ResolveError(f"'{p.thing}' is a reaction, so {phys} applies to its species one at a time",
                           "Ask about one substance, or about the reaction's heat, equilibrium or barrier.")
    raise ResolveError(f"{phys} does not apply to a {q.kind}", "Leave out the physics and Wobbl picks one.")


def simulate(thing: str, question: str = "", physics: str | None = None, *, temperature: float | None = None,
             pressure: float | None = None, solvent: str = "", phase: str = "", accuracy: str = "normal",
             out: str | Path | None = None, quick: bool = False, render: bool = True, log=None,
             options: dict | None = None):
    """Run chemistry on anything. temperature is K and pressure Pa, either overriding what the text says.
    accuracy is fast (GFN2-xTB only), normal (with DFT checks on small molecules) or high (DFT as the answer).
    quick skips images and dynamics. options holds the extras a physics reads: phi, oxidizer and limit for flames,
    chamber_pressure, area_ratio and ambient for propellants, film and from_result for cells, md_fs for dynamics."""
    p = prepare(thing, question, physics, temperature=temperature, pressure=pressure, solvent=solvent, phase=phase,
                accuracy=accuracy, out=out, quick=quick, render=render, options=options)
    o = execute(p, log)
    if p.guesses and hasattr(o, "assumptions"):
        o.assumptions = [f"Wobbl guessed: {t}." for t, _ in p.guesses] + list(o.assumptions)
    return o


def job_entry(job):
    """wobbl simulate's entry for every chemistry analysis: read the options, record each guess, stop at
    job.ready() under --plan, then run."""
    from ..units import parse_quantity
    text = job.focus or job.question
    T = job.value("temperature")
    P = job.value("pressure")
    opts = {}
    for name, key in (("phi", "phi"), ("oxidizer", "oxidizer"), ("limit", "limit"), ("area-ratio", "area_ratio"),
                      ("chamber-pressure", "chamber_pressure"), ("ambient", "ambient"), ("film", "film"),
                      ("from", "from_result"), ("md-time", "md_fs")):
        v = job.value(name)
        if v not in (None, ""):
            opts[key] = v
    for key in ("limit", "ambient", "chamber_pressure"):
        if key in opts and re.search(r"[a-zA-Z]", str(opts[key])):
            hot = re.search(r"\d\s*(?:°\s*)?(?:c|k|f|celsius|kelvin|fahrenheit)\s*$", str(opts[key]), re.I)
            kind = "temperature" if hot else "stress"
            v = parse_quantity(str(opts[key]), kind).value
            opts[key] = v + 273.15 if kind == "temperature" else v * 1e6
    p = prepare(str(job.path) if job.path else job.thing, text, job.analysis.name,
                temperature=parse_quantity(T, "temperature").value + 273.15 if T else None,
                pressure=parse_quantity(P, "stress").value * 1e6 if P else None,
                solvent=job.value("solvent") or "", accuracy=job.value("accuracy") or "normal", out=job.out,
                quick=job.quick, render=not job.quick, options=opts)
    for what, override in p.guesses:
        job.assume(what, override)
    job.use("temperature", f"{p.opts.temperature:g} K")
    job.use("pressure", f"{p.opts.pressure / 1000:g} kPa")
    env = {"temperature": f"{p.opts.temperature:g} K", "pressure": f"{p.opts.pressure / 1000:g} kPa"}
    if p.q.solvent or any(t.phase == "aq" for t in p.q.reactants + p.q.products):
        env["solvent"] = p.q.solvent or "water"
    job.situate(environment=env, objects=["beaker"] if "solvent" in env or p.q.kind == "reaction" else [])
    job.ready()
    return execute(p, job.log)


def _from_question(question: str) -> str:
    """A reaction written inside the question, as in 'what are the products of C3H8 + 5 O2 -> 3 CO2 + 4 H2O?'."""
    m = re.search(r"((?:\d*\s*[A-Z][A-Za-z0-9()]*\s*\+\s*)+\d*\s*[A-Z][A-Za-z0-9()]*\s*(?:->|=>|→|<=>|=)\s*"
                  r"(?:\d*\s*[A-Z][A-Za-z0-9()]*\s*\+\s*)*\d*\s*[A-Z][A-Za-z0-9()]*)", question or "")
    if m:
        return m.group(1)
    raise ResolveError("nothing chemical to simulate", "Give a name, formula, SMILES, equation or a few words.")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s.strip())[:40].strip("-").lower() or "chemistry"


def _molecule(q: Query, phys: str, opts: Options, out: Path, log, options: dict):
    from .molecule import run_molecule
    from .structure import from_file, from_species
    sp = q.main
    if q.path is not None:
        st = from_file(q.path)
        name, formula = q.path.stem, _hill(st.symbols)
    else:
        st = from_species(sp, conformers=4 if opts.accuracy == "fast" else 20)
        name, formula = sp.name, sp.formula
    if phys == "dynamics":
        opts.md, opts.md_fs = True, float(options.get("md_fs", 1000.0))
    extra = {"resolved": q.resolved(), "physics": phys}
    if q.main is not None:
        extra["phases"] = _phase_changes(q.main)
    return run_molecule(st, name, formula, opts, out, extra, log)


def _phase_changes(sp) -> dict:
    """Melting and boiling points at 1 atm from where NASA's phases cross in Gibbs energy, when it lists them."""
    try:
        from .thermodata import lookup
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for a, b, name in (("s", "l", "melting_K"), ("l", "g", "boiling_K")):
        try:
            da, db = lookup(sp, a, 300.0), lookup(sp, b, 300.0)
        except Exception:  # noqa: BLE001
            continue
        if da is None or db is None or da.phase != a or db.phase != b:
            continue
        lo, hi = max(da.lo, db.lo, 150.0), min(da.hi, db.hi, 2500.0)
        f = lambda T: da.G(T) - db.G(T)  # noqa: E731
        if lo >= hi or f(lo) * f(hi) > 0:
            continue
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if f(lo) * f(m) <= 0:
                hi = m
            else:
                lo = m
        out[name] = round(0.5 * (lo + hi), 1)
    return out


def _hill(symbols) -> str:
    from . import formula as F
    counts: dict = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return F.hill(counts)


def _combustion(q: Query, opts: Options, out: Path, log, options: dict):
    """A flame: one fuel in air or oxygen through the validated flame run, any other mix of gases or a written
    reaction through the general mixture equilibrium."""
    from .combustion import FUELS, run_flame
    from .mixture import run_mixture
    limit = options.get("limit")
    phi = float(options.get("phi") or _phi_from(opts.question) or 1.0)
    options["phi"] = phi
    if q.kind == "molecule":
        q.amounts = {"_fuel": [q.main.formula], "_oxidizer": options.get("oxidizer", "air")}
        q.kind = "mixture"
    fuels = q.amounts.get("_fuel") if q.kind == "mixture" else None
    if fuels and len(fuels) == 1:
        sp = next((s for s in q.species if s.formula == fuels[0]), q.main)
        key = next((k for k, v in FUELS.items() if v[0].split(",")[0] == sp.formula or k == sp.name.lower()), None)
        if key:
            ox = "air" if q.amounts.get("_oxidizer") == "air" else "oxygen"
            return run_flame(key, ox, phi, opts.temperature, opts.pressure, out, opts.question,
                             float(limit) if limit else None, render=opts.render)
    return run_mixture(q, opts, out, log, options, flame=True)


def _phi_from(question: str) -> float | None:
    m = re.search(r"(?:phi|equivalence ratio)\s*(?:of|=|is)?\s*(\d+(?:\.\d+)?)", question or "", re.I)
    if m:
        return float(m.group(1))
    if wants(question, r"\blean\b"):
        return 0.8
    if wants(question, r"\brich\b"):
        return 1.2
    return None


def _propellant(q: Query, thing: str, opts: Options, out: Path, options: dict):
    from .combustion import run_propellant
    body, _ = conditions(thing)
    pc = float(options.get("chamber_pressure") or q.pressure or _pressure_from(opts.question) or 6.894757e6)
    eps = options.get("area_ratio")
    m = re.search(r"area ratio (?:of )?(\d+(?:\.\d+)?)", opts.question or "", re.I)
    if eps is None and m:
        eps = float(m.group(1))
    ambient = float(options.get("ambient", 101325.0))
    return run_propellant(body.strip(), pc, float(eps) if eps else None, None, out, opts.question, ambient,
                          render=opts.render)


def _pressure_from(question: str) -> float | None:
    _, c = conditions(question or "")
    return c.get("pressure")


def _runaway(q: Query, thing: str, opts: Options, out: Path, options: dict):
    from .runaway import cell_from, run_runaway, surroundings_from
    fmt_ = re.search(r"18650|21700|26650|14500|\d{2}x\d{2,3}", thing.lower())
    cell = cell_from(fmt_.group(0) if fmt_ else "18650", options.get("film", ""), float(options.get("emissivity",
                                                                                                    0.8)))
    source = ""
    if options.get("from_result"):
        Ta, source = surroundings_from(options["from_result"])
    else:
        Ta = float(options.get("ambient") or 0) or q.temperature or _temperature_from(opts.question) or 298.15
    return run_runaway(cell, Ta, out, opts.question, source, render=opts.render)


def _temperature_from(question: str) -> float | None:
    _, c = conditions(question or "")
    return c.get("temperature")
