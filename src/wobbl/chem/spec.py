"""A solid motor design from a TOML spec or from flags.

A spec names the propellant, lists the grains from the forward end to the nozzle, and gives the nozzle and the
casing (see examples/knsb-motor/motor.toml). Every quantity takes a unit ("48 mm", "1.5 MPa", "12 deg"), and a bare
number in a grain string is millimeters. The same design as flags reads

    --propellant knsb --grain "bates d=48 core=18 L=80 x4" --throat 13mm --exit 33mm --casing "6061-t6 od=54 wall=2.5"

Wobbl models given designs only and says nothing about making or handling propellant.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..units import UnitError, parse_quantity
from .ballistics import Motor, Nozzle
from .grains import CORES, make_grain
from .propellants import BurnRate, Propellant, get_propellant, parse_composition

_SI = {"length": 1e-3, "mass": 1e3, "stress": 1e6, "density": 1e12, "angle": 1.0}
GRAIN_KEYS = {"shape", "diameter", "core", "length", "count", "inhibited_ends", "points", "point_length",
              "point_width", "fins", "fin_length", "fin_width", "offset", "slot_width", "slot_length"}
SHAPE_SIZES = {"core", "point_length", "point_width", "fin_length", "fin_width", "offset", "slot_width",
               "slot_length"}


class SpecError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


@dataclass
class Casing:
    material: str
    outer: float                    # m
    inner: float                    # m
    length: float = 0.0             # m inside, for the free volume and the motor file
    target_sf: float = 1.5          # on yield at peak pressure
    hardware: float = 0.0           # kg of closures and nozzle, for the motor file

    @property
    def wall(self) -> float:
        return (self.outer - self.inner) / 2


@dataclass
class MotorSpec:
    name: str
    motor: Motor
    casing: Casing | None = None
    question: str = ""
    source: str = ""
    raw: dict = field(default_factory=dict)


def q(value, kind: str, where: str, default_unit: str | None = None) -> float:
    """A quantity in SI, naming the key in any error."""
    try:
        return parse_quantity(value if isinstance(value, str) else float(value), kind,
                              default_unit).value * _SI[kind]
    except UnitError as e:
        raise SpecError(f"could not read {where}, {e}") from None


def _num(value, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SpecError(f"{where} should be a plain number, not '{value}'") from None


def _check(d: dict, known: set[str], where: str):
    extra = set(d) - known
    if extra:
        raise SpecError(f"unknown key{'s' if len(extra) > 1 else ''} {', '.join(sorted(extra))} in [{where}]",
                        f"[{where}] takes {', '.join(sorted(known))}.")


def load(source) -> tuple[dict, str]:
    """A spec as a dict, from a TOML file, TOML text or a dict already parsed."""
    if isinstance(source, dict):
        return source, ""
    text, path = str(source), None
    if "\n" not in text and "=" not in text:
        path = Path(text)
        if not path.exists():
            raise SpecError(f"there is no motor spec at {source}")
        text = path.read_text()
    try:
        import tomllib
    except ModuleNotFoundError:   # Python 3.10
        import tomli as tomllib
    try:
        return tomllib.loads(text), str(path.resolve()) if path else ""
    except tomllib.TOMLDecodeError as e:
        raise SpecError(f"the motor spec is not valid TOML ({e})") from None


def read_spec(source, question: str = "") -> MotorSpec:
    d, path = load(source)
    _check(d, {"name", "propellant", "grain", "nozzle", "casing", "ambient"}, "spec")
    if "propellant" not in d or "grain" not in d or "nozzle" not in d:
        raise SpecError("a motor spec needs [propellant], at least one [[grain]] and [nozzle]",
                        "See examples/knsb-motor/motor.toml.")
    prop, eff = _propellant(d["propellant"])
    grains = []
    for i, g in enumerate(d["grain"] if isinstance(d["grain"], list) else [d["grain"]]):
        grains += _grains(g, f"grain {i + 1}")
    n = d["nozzle"]
    _check(n, {"throat", "exit", "efficiency", "half_angle"}, "nozzle")
    noz = Nozzle(q(n.get("throat", ""), "length", "nozzle.throat"), q(n.get("exit", ""), "length", "nozzle.exit"),
                 _num(n.get("efficiency", 0.90), "nozzle.efficiency"),
                 q(n.get("half_angle", "15 deg"), "angle", "nozzle.half_angle"))
    if not 0 < noz.throat <= noz.exit:
        raise SpecError("the nozzle needs a throat above zero and an exit at least as wide")
    if not 0.5 <= noz.efficiency <= 1.0:
        raise SpecError("nozzle.efficiency is a fraction between 0.5 and 1, typically 0.85 to 0.95")
    casing = _casing(d["casing"]) if "casing" in d else None
    big = max(g.D for g in grains)
    if casing and big > casing.inner * 1.001:
        raise SpecError(f"a {big * 1000:.4g} mm grain does not fit the casing's {casing.inner * 1000:.4g} mm bore")
    stack = sum(g.L for g in grains)
    if casing and casing.length and stack > casing.length * 1.001:
        raise SpecError(f"the grains stack to {stack * 1000:.4g} mm, longer than the casing's "
                        f"{casing.length * 1000:.4g} mm inside")
    vol = math.pi / 4 * casing.inner ** 2 * (casing.length or stack) if casing else 0.0
    amb = q(d.get("ambient", "1 atm"), "stress", "ambient")
    motor = Motor(grains, prop, noz, chamber_volume=vol, ambient=amb, cstar_efficiency=eff)
    return MotorSpec(str(d.get("name") or (Path(path).stem if path else "motor")), motor, casing, question, path, d)


def _propellant(p) -> tuple[Propellant, float]:
    if isinstance(p, str):
        p = {"name": p}
    _check(p, {"name", "density", "a", "n", "bands", "T", "M", "k", "composition", "combustion_efficiency"},
           "propellant")
    eff = _num(p.get("combustion_efficiency", 1.0), "propellant.combustion_efficiency")
    if not 0.5 <= eff <= 1.0:
        raise SpecError("propellant.combustion_efficiency is the fraction of ideal c*, between 0.5 and 1")
    base = None
    if "name" in p:
        try:
            base = get_propellant(str(p["name"]))
        except KeyError as e:
            if not {"a", "bands"} & set(p):
                raise SpecError(str(e).strip("'\"")) from None
    if base is None:
        missing = [k for k in ("density", "T", "M", "k") if k not in p] + ([] if {"a", "bands"} & set(p) else ["a"])
        if missing:
            raise SpecError(f"a custom propellant needs {', '.join(missing)}",
                            "Give density, the burn rate a (mm/s at MPa) and n, and T (K), M (kg/kmol) and k, or "
                            "name a known propellant: knsb, kndx, knsu, ocean-water, cherry-limeade.")
    over = {}
    if "density" in p:
        over["density"] = q(p["density"], "density", "propellant.density")
    if "bands" in p:
        over["burn_rate"] = BurnRate(tuple(tuple(_num(v, "propellant.bands") for v in b) for b in p["bands"]))
    elif "a" in p:
        over["burn_rate"] = BurnRate(((0.0, 1e3, _num(p["a"], "propellant.a"), _num(p.get("n", 0.0),
                                                                                    "propellant.n")),))
    for key in ("T", "M", "k"):
        if key in p:
            over[key] = _num(str(p[key]).split()[0], f"propellant.{key}")
    if "composition" in p:
        try:
            over["composition"] = parse_composition(str(p["composition"]))
        except ValueError as e:
            raise SpecError(f"propellant.composition: {e}") from None
    if base is None:
        prop = Propellant("custom", str(p.get("name", "custom propellant")), source="user supplied", **over)
    else:
        from dataclasses import replace
        prop = replace(base, **over, source=base.source + (" (with user overrides)" if over else ""))
    if prop.k <= 1 or prop.M <= 0 or prop.T <= 0 or prop.density <= 0:
        raise SpecError("the propellant needs T and M above zero, k above 1 and a positive density")
    return prop, eff


def _grains(g: dict, where: str) -> list:
    _check(g, GRAIN_KEYS, where)
    shape = str(g.get("shape", "bates"))
    D = q(g.get("diameter", ""), "length", f"{where}.diameter")
    L = q(g.get("length", ""), "length", f"{where}.length")
    inh = g.get("inhibited_ends", 0)
    inh = {"none": 0, "neither": 0, "one": 1, "top": 1, "forward": 1, "both": 2}.get(str(inh).lower(), inh)
    ends = 2 - int(_num(inh, f"{where}.inhibited_ends"))
    if not 0 <= ends <= 2:
        raise SpecError(f"{where}.inhibited_ends is 0, 1 or 2")
    params = {}
    for key in SHAPE_SIZES & set(g):
        params[key] = q(g[key], "length", f"{where}.{key}")
    for key in ("points", "fins"):
        if key in g:
            params[key] = int(_num(g[key], f"{where}.{key}"))
    count = int(_num(g.get("count", 1), f"{where}.count"))
    if count < 1:
        raise SpecError(f"{where}.count is at least 1")
    try:
        grain = make_grain(shape, D, L, ends, **params)
    except (TypeError, ValueError) as e:
        raise SpecError(f"{where}: {e}".replace("got an unexpected keyword argument", "does not take"),
                        f"Shapes: bates, tube, endburner, {', '.join(CORES)}.") from None
    return [grain] * count


def _casing(c) -> Casing:
    if isinstance(c, str):
        c = parse_casing(c)
    _check(c, {"material", "outer_diameter", "inner_diameter", "wall", "length", "target_sf", "hardware_mass"},
           "casing")
    od = q(c.get("outer_diameter", ""), "length", "casing.outer_diameter")
    if "inner_diameter" in c:
        idia = q(c["inner_diameter"], "length", "casing.inner_diameter")
    elif "wall" in c:
        idia = od - 2 * q(c["wall"], "length", "casing.wall")
    else:
        raise SpecError("the casing needs its wall or inner_diameter")
    if not 0 < idia < od:
        raise SpecError("the casing's inner diameter has to be smaller than its outer diameter")
    from ..materials import resolve_material
    mat = str(c.get("material", "6061-t6"))
    try:
        resolve_material(mat)
    except KeyError as e:
        raise SpecError(str(e).strip("'\"")) from None
    return Casing(mat, od, idia, q(c.get("length", "0 mm"), "length", "casing.length"),
                  _num(c.get("target_sf", 1.5), "casing.target_sf"),
                  q(c.get("hardware_mass", "0 kg"), "mass", "casing.hardware_mass"))


_TOKEN = re.compile(r"^(\w+)\s*=\s*(.+)$")
_ALIAS = {"d": "diameter", "od": "outer_diameter", "id": "inner_diameter", "l": "length", "len": "length",
          "sf": "target_sf", "core": "core", "wall": "wall", "t": "wall", "hardware": "hardware_mass"}


def _with_mm(v: str) -> str:
    return v + " mm" if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v.strip()) else v


def parse_grain(text: str) -> dict:
    """'bates d=48 core=18 L=80 x4 inhibit=1' as a [[grain]] table. Bare numbers are millimeters."""
    toks = text.replace(",", " ").split()
    if not toks:
        raise SpecError("an empty grain")
    out = {"shape": toks[0]}
    for t in toks[1:]:
        m = re.fullmatch(r"[xX](\d+)", t)
        if m:
            out["count"] = int(m.group(1))
            continue
        m = _TOKEN.match(t)
        if not m:
            raise SpecError(f"could not read '{t}' in the grain '{text}'", "Write key=value, like core=18 or x4.")
        k, v = m.group(1).lower(), m.group(2)
        k = {"inhibit": "inhibited_ends", "inhibited": "inhibited_ends", "n": "count"}.get(k, _ALIAS.get(k, k))
        if k in ("count", "points", "fins", "inhibited_ends"):
            out[k] = v
        else:
            out[k] = _with_mm(v)
    return out


def parse_casing(text: str) -> dict:
    """'6061-t6 od=54 wall=2.5 L=400 sf=1.5' as a [casing] table. Bare numbers are millimeters."""
    toks = text.replace(",", " ").split()
    out = {}
    for t in toks:
        m = _TOKEN.match(t)
        if not m:
            out["material"] = t
            continue
        k = _ALIAS.get(m.group(1).lower(), m.group(1).lower())
        v = m.group(2)
        out[k] = v if k == "target_sf" else _with_mm(v) if k != "hardware_mass" else v
    return out


def spec_from_flags(propellant: str, grains: list[str], throat: str, exit_: str, casing: str = "",
                    efficiency: float | None = None, name: str = "motor") -> dict:
    """The TOML table a set of flags stands for."""
    if not propellant or not grains or not throat:
        raise SpecError("a motor needs --propellant, at least one --grain and --throat, or a spec file",
                        'Example: --propellant knsb --grain "bates d=48 core=18 L=80 x4" --throat 13 --exit 33')
    d = {"name": name, "propellant": {"name": propellant}, "grain": [parse_grain(g) for g in grains],
         "nozzle": {"throat": _with_mm(throat), "exit": _with_mm(exit_ or throat)}}
    if efficiency is not None:
        d["nozzle"]["efficiency"] = efficiency
    if casing:
        d["casing"] = parse_casing(casing)
    return d
