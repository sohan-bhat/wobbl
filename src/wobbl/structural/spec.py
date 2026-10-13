"""What a strength question is made of: a part, a material, loads and fixtures.

Loads and fixtures can be written the way people talk about them:

    force     "200 N down @ top"               total force spread evenly over the faces
              "50 lbf -z @ x=max"
    pressure  "0.2 MPa @ top"                  pushes into the faces (negative pulls)
    bolt      "300 N -x @ hole d=6.6"          a pin or bolt bearing on a hole (shared by all holes picked)
    gravity   "3 g down"                       self weight times a factor (no faces)
    remote    "2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid"
                                               a force at a point off the part, carried by the faces
    moment    "0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25"

    fixed     "holes d=4.5"     every point on the faces held still
    roller    "x=min"           faces can slide in their plane but not move off it
    bolt      "holes d=6.6"     bolted: the hole and the washer area around it held still
    wall      "x=min"           rests against a rigid wall: can push on it, can lift off
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.select import SelectionError, parse_point, parse_vector
from ..units import UnitError, parse_quantity

LOAD_KINDS = ("force", "pressure", "bolt", "gravity", "remote", "moment")
FIXTURE_KINDS = ("fixed", "roller", "bolt", "wall")


class SpecError(ValueError):
    pass


@dataclass
class Load:
    kind: str
    faces: str = ""
    magnitude: float = 0.0               # N, MPa, N mm or mm/s2 depending on kind
    direction: list[float] | None = None
    point: list[float] | None = None     # remote and moment loads
    coupling: str = "distributed"        # remote loads: distributed or rigid
    text: str = ""
    imperial: bool = False

    def vector(self) -> np.ndarray:
        d = np.asarray(self.direction if self.direction is not None else [0, 0, -1], float)
        return self.magnitude * d / np.linalg.norm(d)

    def describe(self) -> str:
        return self.text or f"{self.kind} {self.magnitude:g} on {self.faces}"


@dataclass
class Fixture:
    kind: str
    faces: str
    text: str = ""

    def describe(self) -> str:
        return self.text or f"{self.kind} {self.faces}"


@dataclass
class StructuralStudy:
    part: str
    material: str = "6061-t6"
    loads: list[Load] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    target_sf: float | None = None
    quality: str = "normal"              # fast, normal, fine
    budget: float = 240.0                 # seconds for the whole convergence study
    max_levels: int = 4
    question: str = ""
    imperial: bool = False
    scale: float = 1.0
    body: int | None = None
    mesh_size: float | None = None
    print_direction: str | None = None    # layer stacking axis for printed parts, e.g. "z"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralStudy":
        d = dict(d)
        loads = [Load(**x) if isinstance(x, dict) and "kind" in x and "magnitude" in x else parse_load(x)
                 for x in d.pop("loads", [])]
        fixtures = [Fixture(**x) if isinstance(x, dict) and "kind" in x and "faces" in x and len(x) <= 3
                    else parse_fixture(x) for x in d.pop("fixtures", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(loads=loads, fixtures=fixtures, **known)


_DIR_WORDS = r"(?:[-+]?[xyz]|up|down|left|right|front|back|forward|backward|normal|inward|outward)"
_VEC = r"(?:[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?)"


def _split_selector(text: str) -> tuple[str, str]:
    m = re.search(r"\s(?:@|on)\s", " " + text + " ")
    if "@" in text:
        head, sel = text.rsplit("@", 1)
        return head.strip(), sel.strip()
    if m:
        idx = [mm.start() for mm in re.finditer(r"\son\s", " " + text)]
        k = idx[-1]
        return text[:k].strip(), text[k + 3:].strip()
    return text.strip(), ""


def parse_load(spec, kind: str | None = None) -> Load:
    """Parse '200 N down @ top' style text (or a dict from a study file) into a Load."""
    if isinstance(spec, dict):
        kind = spec.get("kind", spec.get("type", kind))
        text = spec.get("load") or spec.get("value") or ""
        faces = spec.get("faces", "")
        direction = spec.get("direction")
        point = spec.get("point") or spec.get("at")
        extra = " ".join(x for x in [str(text),
                                     direction if isinstance(direction, str) else
                                     (",".join(map(str, direction)) if direction else ""),
                                     f"at {point}" if isinstance(point, str) else
                                     (f"at {','.join(map(str, point))}" if point else ""),
                                     spec.get("coupling", "")] if x)
        return parse_load(f"{extra} @ {faces}" if faces else extra, kind)
    text = str(spec).strip()
    head, faces = _split_selector(text)
    low = head.lower()
    mc = re.search(r"\s+(rigid|distributed)\s*$", faces, re.I)
    if mc:
        faces = faces[:mc.start()].strip()
        low += " " + mc.group(1).lower()
    coupling = "rigid" if re.search(r"\brigid\b", low) else "distributed"
    low = re.sub(r"\b(rigid|distributed)\b", "", low).strip()
    point = None
    m = re.search(r"\bat\s+\(?(" + _VEC + r")\)?", low)
    if m:
        point = parse_point(m.group(1)).tolist()
        low = (low[:m.start()] + low[m.end():]).strip()
    # magnitude: the first number with its unit
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([a-z/*.^\-0-9·²]*?)(?=\s|$)", low)
    if not m:
        raise SpecError(f"could not find a magnitude with units in '{text}' (for example '200 N' or '3 g')")
    num, unit = m.group(1), m.group(2)
    rest = low[m.end():].strip()
    if kind is None:
        kind = _guess_kind(unit, point, faces)
    kind = {"torque": "moment", "pin": "bolt", "bearing": "bolt", "weight": "gravity", "accel": "gravity",
            "acceleration": "gravity"}.get(kind, kind)
    if kind not in LOAD_KINDS:
        raise SpecError(f"unknown load type '{kind}' (use {', '.join(LOAD_KINDS)})")
    dim = {"force": "force", "bolt": "force", "remote": "force", "pressure": "stress", "moment": "moment",
           "gravity": "accel"}[kind]
    try:
        q = parse_quantity(f"{num} {unit}", dim)
    except UnitError as e:
        raise SpecError(f"{e} (in load '{text}')") from None
    direction = None
    rest = rest.replace("(", "").replace(")", "")
    mv = re.search(_VEC, rest)
    md = re.search(r"(?<![\w.])" + _DIR_WORDS + r"(?![\w])", rest)
    if mv:
        direction = parse_vector(mv.group(0)).tolist()
    elif md and md.group(0) not in ("normal", "inward", "outward"):
        direction = parse_vector(md.group(0)).tolist()
    elif md:
        direction = None if md.group(0) != "outward" else "outward"
    if kind in ("force", "bolt", "remote", "moment", "gravity") and direction is None:
        if kind == "gravity":
            direction = [0.0, 0.0, -1.0]
        elif kind == "moment":
            raise SpecError(f"moment '{text}' needs an axis, for example '+z' or '0,0,1'")
        elif md is None or md.group(0) != "normal":
            raise SpecError(f"load '{text}' needs a direction such as -z, down or 0,0,-1")
    if direction == "outward":
        direction = None
    if kind == "remote" and point is None:
        raise SpecError(f"remote load '{text}' needs a point: add 'at x,y,z'")
    if kind != "gravity" and not faces:
        raise SpecError(f"load '{text}' needs faces: add '@ <selector>', for example '@ top' or '@ holes'")
    mag = q.value
    return Load(kind, faces, mag, direction, point, coupling, text, q.imperial)


def _guess_kind(unit: str, point, faces: str) -> str:
    u = unit.lower()
    if u in ("g",) and not faces:
        return "gravity"
    try:
        parse_quantity(f"1 {u}", "stress")
        return "pressure"
    except UnitError:
        pass
    try:
        parse_quantity(f"1 {u}", "moment")
        return "moment"
    except UnitError:
        pass
    if point is not None:
        return "remote"
    if faces and re.search(r"\bholes?\b", faces.lower()):
        return "force"
    return "force"


def parse_fixture(spec, kind: str | None = None) -> Fixture:
    if isinstance(spec, dict):
        k = spec.get("kind", spec.get("type", kind or "fixed"))
        return parse_fixture(spec["faces"], k)
    text = str(spec).strip()
    if kind is None:
        m = re.match(r"^(fixed|fix|roller|slider|symmetry|bolt|bolted|wall|rest|support)\s*[:@]?\s+(.+)$", text, re.I)
        if m:
            kind, text = m.group(1).lower(), m.group(2)
        else:
            kind = "fixed"
    kind = {"fix": "fixed", "slider": "roller", "symmetry": "roller", "bolted": "bolt", "rest": "wall",
            "support": "wall", "clamp": "fixed", "clamped": "fixed"}.get(kind.lower(), kind.lower())
    if kind not in FIXTURE_KINDS:
        raise SpecError(f"unknown fixture type '{kind}' (use {', '.join(FIXTURE_KINDS)})")
    if not text:
        raise SpecError("a fixture needs faces, for example 'holes' or 'x=min'")
    return Fixture(kind, text.strip(), f"{kind} {text.strip()}")


__all__ = ["Load", "Fixture", "StructuralStudy", "parse_load", "parse_fixture", "SpecError", "SelectionError"]
