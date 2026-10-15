"""Bridges to the studies whose packages have not registered with simulate yet.

After every package's domain.py has loaded, a domain left with no analysis gets a bridge to its existing study
(circuit templates and netlists, mechanism templates and specs, flow studies, field templates, flames, propellants
and cells), so wobbl simulate reaches every physics Wobbl has today. The moment a package registers anything for
its domain, the bridge for that domain stays out of the way.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from .registry import Analysis, Option, analyses, register

ANY = Option("*", "Any value the study reads, e.g. --supply 5V, --turns 800 or --length '40 m'")


def _given(job, skip=()) -> dict:
    return {k: v[-1] for k, v in job.options.items() if k not in skip}


def _record(job, params: dict) -> None:
    for k, v in params.items():
        job.use(k, str(v))


def circuit(job):
    from ..circuits.run import run_circuit, run_netlist
    from ..circuits.tools import read_circuit_question

    if job.kind == "netlist":
        probes = job.values("probe")
        if not probes:
            nodes = re.findall(r"^\s*[RCLVID]\w*\s+(\w+)\s+(\w+)", Path(job.path).read_text(), re.M | re.I)
            names = list(dict.fromkeys(n for pair in nodes for n in pair if n.lower() not in ("0", "gnd")))[:6]
            probes = job.values("probe", [f"v({n})" for n in names] or ["v(1)"], "signals to read not given")
        job.situate(objects=["breadboard"])
        return run_netlist(str(job.path), probes, job.out, job.question)
    tpl, params, guesses = read_circuit_question(job.text)
    tpl = job.kind if job.kind in ("led", "rc", "buck", "mosfet") else tpl
    given = _given(job, ("probe",))
    for g in guesses:
        key = g.split(" not given")[0].replace("supply voltage", "supply").replace("LED color", "led")
        key = {"no resistor given: assumed 150 ohm": "resistor", "thermal resistance": "theta_ja"}.get(key, key)
        if key not in given:
            value = params.get(key) or (re.findall(r"assumed ([^\s(]+)", g) or [""])[0]
            job.assume(g, f"--{key} {shlex.quote(value)}" if value else "")
    params = {**params, **given}
    _record(job, params)
    job.situate(objects=["breadboard"], circuit=tpl)
    return run_circuit(tpl, params, job.out, job.question)


def mechanism(job):
    from ..mechanism.study import run_mechanism
    from ..mechanism.tools import read_mechanism_question

    if job.path is not None:
        return run_mechanism(str(job.path), _given(job), job.out, job.question, job.log)
    tpl, params, guesses = read_mechanism_question(job.text)
    params = {**params, **_given(job)}
    for g in guesses:
        job.assume(g.split(" (override")[0], (re.findall(r"override with (--set [^)]*)", g) or [""])[0]
                   .replace("--set ", "--"))
    _record(job, params)
    return run_mechanism(job.kind if job.kind in ("arm", "fourbar", "gears", "impact", "tip", "pendulum") else tpl,
                         params, job.out, job.question, job.log)


_N = r"(\d+(?:\.\d+)?)"


def _flow_words(kind: str, text: str) -> dict:
    low = text.lower()
    p = {}

    def grab(key, rx):
        m = re.search(rx, low)
        if m:
            p.setdefault(key, " ".join(m.groups()))

    grab("flow", rf"{_N}\s*(l/min|lpm|gpm|cfm|m3/h)")
    grab("speed", rf"{_N}\s*(m/s|km/h|mph|knots?|kn)\b")
    if kind == "pipe":
        grab("diameter", rf"{_N}\s*(mm|cm|in)\b")
        grab("length", rf"{_N}\s*(m|ft)\b(?!/)")
    elif kind == "drag":
        grab("size", rf"{_N}\s*(mm|cm|m)\b(?!/)")
        shape = re.search(r"\b(sphere|cylinder|disk|cube|square-plate)\b", low)
        if shape:
            p["shape"] = shape.group(1)
    elif kind == "prop":
        grab("mass", rf"{_N}\s*(kg|g)\b")
        m = re.search(r"(\d+)\s*(?:rotors|motors|props)\b", low)
        if m:
            p["rotors"] = m.group(1)
    elif kind == "hull":
        grab("length", rf"{_N}\s*(m|ft)\b(?!/)")
    return p


def flow(job):
    from ..flow.tools import build

    kind = job.kind if job.kind in ("pipe", "drag", "prop", "hull") else "drag"
    params = {**_flow_words(kind, job.text), **_given(job)}
    if kind == "prop" and job.thing and not job.path and " " not in job.thing:
        params.setdefault("prop", job.thing)
    _record(job, params)
    study, run = build(kind, {**params, "question": job.question})
    job.situate(environment={"fluid": params.get("fluid", "water" if kind == "pipe" else "air")})
    return run(study, job.out)


FIELD = {"magnetic": "magnetic", "electric": "electric", "trace": "board", "antenna": "antenna", "sound": "sound"}


def field(job):
    from ..fields.tools import TEMPLATES, check_params, runner

    dom = FIELD[job.analysis.name]
    tpl = job.kind if job.kind in TEMPLATES[dom] else TEMPLATES[dom][0]
    if tpl != job.kind:
        job.assume(f"the device is taken as a {tpl}", f"--kind {tpl}")
    params = _given(job, ("keep",))
    check_params(dom, tpl, params)
    _record(job, params)
    return runner(dom)(tpl, params, job.out, job.question, job.flag("keep"), log=job.log)


def _reaction_fuel(text: str) -> tuple[str, str, float] | None:
    """Fuel, oxidizer and equivalence ratio from 'CH4 + 2 O2 -> ...', when a fuel burns in oxygen or air."""
    from .things import parse_equation
    eq = parse_equation(text)
    if not eq:
        return None
    ox = {s.lower(): n for n, s in eq[0] if s.lower() in ("o2", "oxygen", "air")}
    fuels = [(n, s) for n, s in eq[0] if s.lower() not in ("o2", "oxygen", "air", "n2", "nitrogen")]
    if not ox or len(fuels) != 1:
        return None
    n, s = fuels[0]
    atoms = {e: float(c or 1) for e, c in re.findall(r"([A-Z][a-z]?)(\d*)", s)}
    need = n * (atoms.get("C", 0) + atoms.get("H", 0) / 4 - atoms.get("O", 0) / 2)
    o2 = ox.get("o2", ox.get("oxygen", 0)) or ox.get("air", 0) * 0.21
    return s, "oxygen" if "o2" in ox or "oxygen" in ox else "air", need / o2 if o2 else 1.0


def flame(job):
    from ..chem.combustion import run_flame
    from ..units import parse_quantity

    fuel, oxidizer, phi = job.thing or job.text, job.value("oxidizer") or "air", job.number("phi")
    burn = _reaction_fuel(job.thing) if job.kind == "reaction" else None
    if job.kind == "reaction":
        if not burn:
            raise ValueError("Wobbl burns a fuel in oxygen or air today, and other reactions come with the "
                             "chemistry domain")
        fuel, oxidizer = burn[0], job.value("oxidizer") or burn[1]
        phi = phi or round(burn[2], 4)
        job.use("phi", str(phi))
    if phi is None:
        phi = float(job.value("phi", "1", "mixture not given, so a stoichiometric one"))
    P = parse_quantity(job.value("pressure") or "1 atm", "stress").value * 1e6
    T = parse_quantity(job.value("temperature") or "25 C", "temperature").value + 273.15
    limit = job.value("limit")
    lim = parse_quantity(limit, "temperature").value + 273.15 if limit else None
    return run_flame(fuel, oxidizer, phi, T, P, job.out, job.question, lim, render=not job.quick)


def propellant(job):
    from ..chem.combustion import run_propellant
    from ..units import parse_quantity

    m = re.search(r"(\d+(?:\.\d+)?)\s*(mpa|psi|bar|atm)\b", job.question.lower())
    pc = job.value("pressure") or (f"{m.group(1)} {m.group(2)}" if m else None) or \
        job.value("pressure", "6.895 MPa", "chamber pressure not given")
    job.use("pressure", pc)
    ar = job.number("area-ratio")
    m = re.search(r"area ratio (?:of )?(\d+(?:\.\d+)?)", job.question.lower())
    if ar is None and m:
        ar = float(m.group(1))
        job.use("area-ratio", m.group(1))
    amb = parse_quantity(job.value("ambient") or "1 atm", "stress").value * 1e6
    return run_propellant(job.thing, parse_quantity(pc, "stress").value * 1e6, ar, None, job.out, job.question, amb,
                          render=not job.quick)


def runaway(job):
    from ..chem.runaway import cell_from, run_runaway
    from ..units import parse_quantity

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", job.question.lower())
    amb = job.value("ambient") or (f"{m.group(1)} C" if m else None) or job.value("ambient", "25 C",
                                                                                     "surroundings not given")
    job.use("ambient", amb)
    c = cell_from(job.thing if re.match(r"^\d", job.thing) else "18650", job.value("film") or "")
    return run_runaway(c, parse_quantity(amb, "temperature").value + 273.15, job.out, job.question,
                       render=not job.quick)


BRIDGES = {
    "circuit": [Analysis("circuit", "spice", "Circuit simulation with ngspice", circuit, default=True,
                         options=(Option("probe", "Signal to read from a netlist, e.g. 'v(out)'", multiple=True), ANY),
                         legacy="wobbl circuit")],
    "mechanism": [Analysis("mechanism", "dynamics", "Mechanism dynamics with MuJoCo", mechanism, default=True,
                           options=(ANY,), aliases=("motion", "torque"), legacy="wobbl mechanism")],
    "fluid": [Analysis("fluid", k, s, flow, kinds=(k,), default=True, options=(ANY,), legacy=f"wobbl flow {k}")
              for k, s in (("pipe", "Pressure drop through a pipe or duct"), ("drag", "Drag from published curves"),
                           ("prop", "Propeller thrust and hover"), ("hull", "Hull resistance and power"))],
    "field": [Analysis("field", a, s, field, kinds=k, default=True, options=(ANY,), legacy=f"wobbl {FIELD[a]}")
              for a, s, k in (("magnetic", "Magnetic force, inductance and saturation",
                               ("solenoid", "electromagnet", "magnet")),
                              ("trace", "Trace impedance and crosstalk",
                               ("microstrip", "stripline", "coplanar", "coupled", "coax")),
                              ("electric", "Capacitance, peak field and breakdown", ("plates", "coax", "sphere")),
                              ("antenna", "Antenna resonance, gain and range", ("dipole", "monopole", "patch", "loop", "invf", "link")),
                              ("sound", "Room modes, mufflers and panel noise",
                               ("room", "enclosure", "muffler", "panel")))],
    "chemistry": [
        Analysis("chemistry", "flame", "Adiabatic flame temperature and products", flame, kinds=("fuel", "reaction"),
                 default=True, keywords=(r"flame", r"burn", r"combust"), legacy="wobbl flame",
                 options=tuple(Option(n) for n in ("oxidizer", "phi", "pressure", "temperature", "limit"))),
        Analysis("chemistry", "propellant", "Propellant c* and Isp", propellant, kinds=("propellant",), default=True,
                 legacy="wobbl propellant",
                 options=tuple(Option(n) for n in ("pressure", "area-ratio", "ambient"))),
        Analysis("chemistry", "runaway", "Lithium ion thermal runaway onset", runaway, kinds=("cell",), default=True,
                 legacy="wobbl runaway", options=(Option("ambient"), Option("film")))],
}


def fill() -> None:
    """Register a bridge for each domain that no package has registered an analysis for."""
    for dom, items in BRIDGES.items():
        if not analyses(dom):
            for a in items:
                register(a)
