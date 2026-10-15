"""The kinds of thing Wobbl reads, and how each is recognized. Analyses register on these from each package's
domain.py; a package with a new kind of thing registers its own Domain the same way."""
from __future__ import annotations

import re
from pathlib import Path

from .registry import Domain, Option, register

PART_OPTIONS = (
    Option("material", "Material name from 'wobbl materials', or custom:E=..,nu=..,yield=..", "6061-t6"),
    Option("load", "Load, e.g. '200 N down @ top' or '300 N -x bearing @ hole d=6.6'", multiple=True),
    Option("fix", "Support, e.g. 'bolt holes d=4.5', 'x=min' or 'elastic 50 N/mm3 @ z=min'", multiple=True),
    Option("target-sf", "Required safety factor"),
    Option("quality", "fast, normal or fine", "normal"),
    Option("scale", "Scale the geometry, 1000 for files in meters", "1"),
    Option("body", "Pick one body from a multi body file"),
    Option("param", "Parameter of a parametric .py part, e.g. thickness=6", multiple=True),
    Option("mesh-size", "Global element size in mm"),
)


def _read(path: Path, limit: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _tables(path: Path) -> set[str]:
    """Top level table and key names of a TOML or JSON spec, lower case."""
    text = _read(path)
    if path.suffix.lower() == ".json":
        import json
        try:
            d = json.loads(text)
            return {str(k).lower() for k in d} if isinstance(d, dict) else set()
        except ValueError:
            return set()
    names = set(re.findall(r"^\s*\[\[?\s*([\w.-]+)", text, re.M))
    names |= set(re.findall(r"^\s*([\w-]+)\s*=", text, re.M))
    return {n.split(".")[0].lower() for n in names}


def sniff_part(path: Path) -> str | None:
    if path.suffix.lower() == ".py":
        text = _read(path)
        return "parametric" if re.search(r"^\s*def build\s*\(", text, re.M) or "PARAMS" in text else None
    return None


def sniff_mechanism(path: Path) -> str | None:
    suf = path.suffix.lower()
    if suf in (".xml", ".mjcf"):
        text = _read(path, 4000).lower()
        return "mjcf" if "<mujoco" in text else "urdf" if "<robot" in text else None
    if suf in (".toml", ".json"):
        t = _tables(path)
        return "spec" if t & {"bodies", "body", "joints", "joint", "drives", "drive", "links"} else None
    return None


def sniff_flight(path: Path) -> str | None:
    if path.suffix.lower() not in (".toml", ".json"):
        return None
    t = _tables(path)
    if t & {"rocket", "launch", "nose", "fins", "drogue"}:
        return "rocket"
    if t & {"grain", "grains", "nozzle", "casing"}:
        return "motor"
    if t & {"body", "throw", "drop"} and not t & {"joints", "joint"}:
        return "body"
    return None


def sniff_fluid(path: Path) -> str | None:
    return "cfd" if path.is_dir() and (path / "system" / "controlDict").exists() else None


register(Domain(
    "part", "A part as a CAD file (STEP, STL, BREP, IGES or a parametric Python part)",
    extensions={".step": "step", ".stp": "step", ".stl": "stl", ".brep": "brep", ".brp": "brep", ".iges": "iges",
                ".igs": "iges", ".py": ""},
    sniff=sniff_part, options=PART_OPTIONS, default_kind="step", priority=2,
    results=("strength", "thermal", "buckling", "vibration", "fatigue", "plastic"),
    example='wobbl simulate bracket.step "will it hold 200 N and stay under 60 C?"'))

register(Domain(
    "circuit", "An electronic circuit, as a SPICE netlist or in words",
    extensions={".cir": "netlist", ".sp": "netlist", ".spice": "netlist", ".net": "netlist", ".ckt": "netlist"},
    words=(r"\bleds?\b", r"\bresistors?\b", r"\bripple\b", r"\bmosfets?\b", r"\bregulators?\b", r"\bbuck\b",
           r"\bconverter\b", r"\bcapacitors?\b", r"\bcircuits?\b", r"\b\d+(?:\.\d+)?\s*v\b", r"\bvolts?\b",
           r"\bamps?\b", r"\bohms?\b|Ω", r"\bnetlist\b", r"\btransistors?\b", r"\bdiodes?\b", r"\bop.?amps?\b",
           r"\btime constant\b", r"\brc\b", r"\bfry\b"),
    kinds={"led": r"\bleds?\b", "buck": r"ripple|\bbuck\b|regulator|converter|step.?down",
           "mosfet": r"mosfet|\bfet\b|transistor", "rc": r"\brc\b|time constant"},
    results=("circuit",), priority=1,
    example='wobbl simulate "will a blue LED on 5V with a 47 ohm resistor fry?"'))

register(Domain(
    "mechanism", "A mechanism, as a spec, a URDF or MJCF model, or in words",
    extensions={".urdf": "urdf", ".mjcf": "mjcf", ".xml": "", ".toml": "", ".json": ""},
    sniff=sniff_mechanism,
    words=(r"\bservos?\b", r"robot(?:ic)? arm", r"pick and place", r"\blinkage\b", r"four.?bar", r"\bcrank\b",
           r"\brocker\b", r"\bgears?\b", r"gearbox", r"\bteeth\b", r"\bpendulum\b", r"tip(?:s|ping)? over",
           r"\btopple", r"fall over", r"\bwiper\b", r"\bmechanism\b", r"\bimpact\b", r"collid|collision",
           r"dropped onto|drop onto", r"peak (?:contact )?force", r"\bhits?\b", r"\btorque\b"),
    kinds={"pendulum": r"\bpendulum\b", "fourbar": r"four.?bar|linkage|wiper|\bcrank\b|\brocker\b",
           "gears": r"\bgears?\b|gearbox|gear train|\bteeth\b|gear ratio",
           "tip": r"tip(?:s|ping)? over|topple|fall over|knock(?:ed)? over",
           "impact": r"\bimpact\b|collid|collision|\bhits?\b|dropped onto|drop onto|peak force",
           "arm": r"\bservos?\b|robot arm|\barm\b|pick and place|\blift\b"},
    default_kind="arm", results=("mechanism",),
    example='wobbl simulate "can two mg996r servos lift 500 g with a robot arm?"'))

register(Domain(
    "flight", "A body or a rocket in flight, or a solid rocket motor, as a spec or in words",
    extensions={".toml": "", ".json": ""}, sniff=sniff_flight,
    words=(r"\brockets?\b", r"\bapogee\b", r"\blaunch(?:ed|es)?\b", r"\bthrown\b|\bthrows?\b", r"\bprojectiles?\b",
           r"\btrajector(?:y|ies)\b", r"\bfall(?:s|ing)? from\b", r"dropped from|drop(?:ped)? from",
           r"\bparachutes?\b", r"terminal velocity", r"\bhow far\b", r"\bballs?\b", r"\bcannon", r"\bfall time\b",
           r"impact speed", r"\blands?\b", r"solid (?:rocket )?motor", r"\bgrains?\b", r"\bbates\b",
           r"\bnozzle\b", r"thrust curve", r"total impulse", r"motor class", r"\bstability margin\b"),
    kinds={"motor": r"solid (?:rocket )?motor|\bgrains?\b|\bbates\b|\bnozzle\b|thrust curve|total impulse|"
                    r"chamber pressure|motor class",
           "rocket": r"\brockets?\b|apogee|parachute|stability margin", "body": r"."},
    default_kind="body", results=("motion", "motor"),
    example='wobbl simulate "a 9 cm rock dropped from 300 m, how fast does it land?"'))

register(Domain(
    "fluid", "A flow case: a pipe or duct, a body in a stream, a propeller, a hull, or a solver case folder",
    sniff=sniff_fluid,
    words=(r"\bpipes?\b", r"\bducts?\b", r"\bhoses?\b", r"\bdrag\b", r"\bpropellers?\b", r"\bprops?\b",
           r"\bdrones?\b", r"\bquad(?:copter)?s?\b", r"multirotor", r"\bhover", r"\bhulls?\b", r"\bboats?\b",
           r"\bships?\b", r"\bvessel\b", r"\bkayak\b", r"pressure drop", r"flow rate", r"l/min|\blpm\b|\bgpm\b|\bcfm\b",
           r"m3/h", r"\bpumps?\b", r"\bfluids?\b", r"\bmixtures?\b", r"viscosity", r"reynolds", r"\bknots?\b",
           r"\d\s*kn\b", r"wind load", r"\bairflow\b", r"\bapcs?f?_", r"\b\d+x\d+(?:\.\d+)?\b"),
    kinds={"pipe": r"\bpipes?\b|\bducts?\b|\bhoses?\b|\btubing\b|pressure drop|flow rate|l/min|\blpm\b|\bgpm\b|"
                   r"\bcfm\b|m3/h|\bpump",
           "prop": r"propeller|\bprops?\b|drone|\bquad|multirotor|\bhover|\brotors?\b|\bapcs?f?_",
           "hull": r"\bhulls?\b|\bboats?\b|\bships?\b|\bvessel\b|kayak|\bknots?\b|\d\s*kn\b|displacement",
           "drag": r"\bdrag\b|wind load|\bsphere\b|\bcylinder\b|\bplate\b|\bcube\b|\bdisk\b"},
    default_kind="drag", results=("flow",),
    example='wobbl simulate "drone with apcsf_10x4.7 props, 1.2 kg, 4 rotors, can it hover?"'))

register(Domain(
    "field", "A device in an electric, magnetic, radio or sound field, described by its dimensions",
    words=(r"\bsolenoids?\b", r"electromagnets?", r"\bmagnets?\b", r"neodymium|\bn(?:35|42|52)\b", r"\bcoils?\b",
           r"\bplunger\b", r"inductance", r"capacitance", r"\bbreakdown\b", r"\b\d+(?:\.\d+)?\s*kv\b", r"dielectric",
           r"\bcoax", r"microstrip", r"stripline", r"coplanar", r"\btraces?\b", r"impedance", r"crosstalk",
           r"\bantennas?\b", r"\bdipole\b", r"\bmonopole\b", r"\bpatch\b", r"\bloop antenna", r"inverted[- ]?f\b", r"wi-?fi|bluetooth|lora|zigbee",
           r"\bdbm\b", r"\b\d+(?:\.\d+)?\s*[gm]hz\b", r"wireless|\brange\b", r"room modes?|\brooms?\b",
           r"muffler|silencer", r"\bacoustic", r"\bsound\b|\bnoise\b|\bdb\b", r"standing waves?", r"\bpcb\b"),
    kinds={"solenoid": r"\bsolenoids?\b|\bplunger\b", "electromagnet": r"electromagnets?",
           "magnet": r"\bmagnets?\b|neodymium|\bn(?:35|42|52)\b", "coupled": r"crosstalk|coupled|differential pair",
           "stripline": r"stripline", "coplanar": r"coplanar", "microstrip": r"microstrip|\btraces?\b|\bpcb\b",
           "coax": r"\bcoax", "plates": r"\bplates?\b|capacitance|capacitor", "sphere": r"\bsphere\b|spark gap",
           "dipole": r"\bdipole\b", "monopole": r"\bmonopole\b|\bwhip\b", "patch": r"\bpatch\b",
           "loop": r"\bloop\b", "invf": r"inverted[- ]?f\b|\bifa\b|\bmeander|trace antenna",
           "link": r"\brange\b|\blink\b|wireless|wi-?fi|bluetooth|lora|zigbee", "room": r"\brooms?\b",
           "muffler": r"muffler|silencer|exhaust", "panel": r"\bpanel\b|vibrating",
           "enclosure": r"\benclosure\b|\bbox\b|cabinet"},
    results=("magnetic", "electric", "board", "antenna", "sound"),
    example='wobbl simulate "solenoid with 800 turns at 0.6 A and a 6 mm stroke, does it pull 8 N?"'))


# chemistry: names, formulas, SMILES and equations are recognized by their shape, not only by words
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y "
            "Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
            "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu").split()
ELEMENT_NAMES = ("hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium "
                 "aluminum aluminium silicon phosphorus sulfur sulphur chlorine argon potassium calcium titanium "
                 "chromium manganese iron cobalt nickel copper zinc gallium germanium arsenic selenium bromine krypton "
                 "silver tin iodine xenon tungsten platinum gold mercury lead uranium").split()
COMPOUNDS = ("water", "ammonia", "ethanol", "acetone", "benzene", "toluene", "carbon dioxide", "ozone", "glucose",
             "caffeine", "aspirin", "nitrous oxide", "hydrogen peroxide", "sulfuric acid", "hydrochloric acid",
             "sodium chloride", "octane", "kerosene", "gasoline", "petrol", "diesel", "hydrazine", "nitromethane",
             "isopropanol", "methyl", "ethyl", "chloride", "oxide", "hydroxide", "nitrate", "sulfate", "carbonate")
CELLS = r"^(?:18650|21700|26650|14500|\d{2}x\d{2,3}(?:\s*mm)?)$"
_SYM = r"(?:[A-Z][a-z]?)"
_FORMULA = re.compile(rf"^(?:{_SYM}\d*(?:\.\d+)?|\({_SYM}\d*(?:{_SYM}\d*)*\)\d*)+$")
_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$/\\%.]+$")


def is_formula(tok: str) -> bool:
    if not _FORMULA.match(tok):
        return False
    return all(s in ELEMENTS for s in re.findall(r"[A-Z][a-z]?", tok))


def is_smiles(tok: str) -> bool:
    if not _SMILES.match(tok) or is_formula(tok) and not re.search(r"([A-Z][a-z]?)\1", tok):
        return False
    atoms = re.findall(r"Cl|Br|[BCNOPSFI]|[cnops]", tok)
    letters = re.sub(r"[^A-Za-z]", "", tok)
    return bool(atoms) and len("".join(atoms)) == len(letters) and (bool(re.search(r"[=#()\[\]]|[A-Za-z]\d", tok))
                                                                    or re.search(r"([A-Z])\1", tok) is not None)


def _species(tok: str) -> bool:
    t = tok.strip().lower()
    from ..chem.combustion import ALIASES, FUELS
    return is_formula(tok.strip()) or is_smiles(tok.strip()) or t in FUELS or t in ALIASES or t in COMPOUNDS \
        or t in ELEMENT_NAMES or t in ("air", "oxygen", "nitrogen")


def parse_equation(text: str) -> tuple[list[tuple[float, str]], list[tuple[float, str]]] | None:
    """'CH4 + 2 O2 -> CO2 + 2 H2O' as reactants and products with their coefficients; reactants alone are fine."""
    sides = re.split(r"\s*(?:<=>|<->|->|→|=>|⇌|=)\s*", text.strip())
    if len(sides) > 2 or not sides[0] or ("+" not in sides[0] and len(sides) == 1):
        return None
    out = []
    for side in sides:
        terms = []
        for term in re.split(r"\s+\+\s+|\s*\+\s*(?=[A-Z\d(])", side.strip()):
            m = re.match(r"^(\d+(?:\.\d+)?)?\s*(.+?)$", term.strip())
            if not m or not _species(m.group(2)):
                return None
            terms.append((float(m.group(1) or 1), m.group(2).strip()))
        out.append(terms)
    return out[0], (out[1] if len(out) > 1 else [])


def chemistry_kind(thing: str) -> tuple[int, str] | None:
    """What a chemistry thing is, from its shape: an equation, a composition, a cell, a fuel or other molecule,
    an element or a material."""
    t = thing.strip()
    if not t:
        return None
    low = t.lower()
    from ..chem.combustion import ALIASES, FUELS
    from ..chem.propellants import INGREDIENTS, PROPELLANTS
    if parse_equation(t):
        return 9, "reaction"
    if re.match(r"^[\w-]+\s*:\s*\d", t) and all(k.strip().lower() in INGREDIENTS
                                                  for k in re.findall(r"([\w-]+)\s*:", t)):
        return 9, "propellant"
    if low in PROPELLANTS or any(low in p.aliases for p in PROPELLANTS.values()):
        return 8, "propellant"
    if re.match(CELLS, low):
        return 8, "cell"
    if low in FUELS or low in ALIASES:
        return 8, "fuel"
    if " " in t and len(t.split()) > 3:
        return None
    if low in ELEMENT_NAMES or t in ELEMENTS and len(t) <= 2:
        return 7, "element"
    if low in COMPOUNDS or is_formula(t) or is_smiles(t):
        return 7, "molecule"
    from ..materials import LIBRARY
    for m in LIBRARY.values():
        if low in (m.key, m.name.lower(), *m.aliases):
            return 6, "material"
    return None


def recognize_chemistry(thing: str, question: str) -> tuple[int, str] | None:
    hit = chemistry_kind(thing) if thing else None
    if hit:
        return hit
    term = r"\d*\s*[A-Za-z0-9()]+"
    m = re.search(rf"((?:\d+\s+)?[A-Za-z0-9()]+(?:\s*\+\s*{term})+\s*(?:->|=>|→|<=>|=)\s*{term}"
                  rf"(?:\s*\+\s*{term})*)", question or "")
    if m and parse_equation(m.group(1)):
        return 6, "reaction"
    return None


register(Domain(
    "chemistry", "Chemistry: a molecule, element, material, reaction, fuel, propellant or battery cell, as a name, "
                 "formula, SMILES or equation",
    recognize=recognize_chemistry,
    words=(r"\bmolecules?\b", r"\bcompounds?\b", r"\breactions?\b", r"\bcombust", r"\bburn(?:s|ing)?\b",
           r"\bflames?\b", r"\bisp\b", r"\bc\*", r"specific impulse", r"\bequilibrium\b", r"\benthalpy\b",
           r"\bpropellants?\b", r"\bfuels?\b", r"\boxidi[sz]er\b", r"stoichiometric", r"equivalence ratio",
           r"thermal runaway", r"\brunaway\b", r"\bli-?ion\b|lithium", r"\bbatter(?:y|ies)\b", r"molar mass",
           r"adiabatic", r"\bknsb\b|\bkndx\b|\bknsu\b"),
    kinds={"reaction": r"->|→|=>|<=>", "cell": r"18650|21700|26650|14500|li-?ion|lithium|\bcells?\b|batter",
           "propellant": r"\bpropellants?\b|\bisp\b|c\*|specific impulse|\bkn(?:sb|dx|su)\b",
           "fuel": r"methane|propane|butane|hydrogen|ethane|ethylene|acetylene|methanol|natural gas|\blpg\b|"
                   r"flame|combust|\bburn"},
    default_kind="molecule", results=("combustion", "battery"),
    example='wobbl simulate "CH4 + 2 O2 -> CO2 + 2 H2O" "how hot is the flame?"'))
