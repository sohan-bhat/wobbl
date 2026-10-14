"""What a crack growth question is made of: a part, its loads and supports, a crack, a spectrum and a target.

A crack is written the way people describe one:

    "through 1.27 mm at hole d=6.4"          straight through the wall from the hole's most stressed point
    "corner 1.27 mm at hole d=6.4"           a quarter ellipse where the hole meets a face
    "surface a=1 c=2 at hottest"             a half ellipse 1 mm deep and 4 mm long on the most stressed surface
    "surface 1 mm a/c=0.5 at 30,0,10"        the same shape, at the surface point nearest 30,0,10
    "center 2a=20 at 50,75,2 normal y along x"   two tips, as in the middle of a plate
    "through 2 mm at 0,75,2 toward +x normal y"

The crack plane is normal to the largest principal stress at the origin unless a normal is given, and the crack
grows into the part from the surface unless 'toward' says which way.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.select import SelectionError, parse_vector
from ..structural.spec import Fixture, Load, SpecError, parse_fixture, parse_load
from ..units import UnitError, parse_quantity

SHAPES = ("through", "corner", "surface", "center")
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
_VEC = rf"(?:{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}|[-+]?[xyz]|up|down|left|right|front|back)"


@dataclass
class CrackSpec:
    shape: str
    a: float                      # mm: depth of a surface or corner crack, length of a through crack, half length
                                  # of a center crack
    c: float | None = None        # mm: half length of a surface crack along the surface, or a corner crack's reach
                                  # along its second face
    where: str = "hottest"        # hottest, a point x,y,z, or a face selector such as 'hole d=6.4'
    normal: list[float] | None = None
    toward: list[float] | None = None
    text: str = ""

    def describe(self) -> str:
        return self.text or f"{self.shape} {self.a:g} mm at {self.where}"


def _length(text: str) -> float:
    try:
        return parse_quantity(text if re.search(r"[a-z\"]", text, re.I) else f"{text} mm", "length").value
    except UnitError as e:
        raise SpecError(str(e)) from None


def parse_crack(text: str) -> CrackSpec:
    """Read a crack description. Raises SpecError saying what was expected."""
    raw = str(text).strip()
    low = raw.lower()
    m = re.match(r"^(through|corner|surface|center|centre|edge)\b\s*(.*)$", low)
    if not m:
        raise SpecError(f"a crack starts with its shape, one of {', '.join(SHAPES)}, as in 'through 1.27 mm at hole "
                        f"d=6.4' (got '{raw}')")
    shape = {"centre": "center", "edge": "through"}.get(m.group(1), m.group(1))
    rest = m.group(2)
    vecs = {}
    for word in ("normal", "toward", "towards", "along"):
        mv = re.search(rf"\b{word}\s+\(?({_VEC})\)?", rest)
        if mv:
            try:
                vecs["toward" if word in ("towards", "along") else word] = parse_vector(mv.group(1)).tolist()
            except SelectionError as e:
                raise SpecError(f"{e} (in crack '{raw}')") from None
            rest = (rest[:mv.start()] + rest[mv.end():]).strip()
    where = "hottest"
    mw = re.search(r"\bat\s+(.+)$", rest)
    if mw:
        where = mw.group(1).strip()
        rest = rest[:mw.start()].strip()
    a = c = None
    ratio = None
    for key, val in re.findall(rf"(2a|a/c|a|c)\s*=\s*({_NUM}\s*(?:mm|in|um|cm)?)", rest):
        if key == "a/c":
            ratio = float(val)
        elif key == "2a":
            a = 0.5 * _length(val)
        elif key == "a":
            a = _length(val)
        else:
            c = _length(val)
    rest = re.sub(rf"(2a|a/c|a|c)\s*=\s*{_NUM}\s*(?:mm|in|um|cm)?", "", rest).strip()
    if a is None:
        mq = re.match(rf"^({_NUM}\s*(?:mm|in|inch|inches|um|cm|\")?)", rest)
        if not mq:
            raise SpecError(f"crack '{raw}' needs a size, as in 'through 1.27 mm' or 'surface a=1 c=2'")
        a = _length(mq.group(1).strip())
    if a <= 0:
        raise SpecError(f"crack '{raw}' needs a size above zero")
    if shape in ("surface", "corner") and c is None:
        c = a / ratio if ratio else a
    if shape == "surface" and c < a:
        raise SpecError(f"a surface crack's half length c must be at least its depth a (got a={a:g}, c={c:g} mm)")
    if where != "hottest" and not where:
        raise SpecError(f"crack '{raw}': say where it starts after 'at'")
    return CrackSpec(shape, float(a), None if c is None else float(c), where, vecs.get("normal"), vecs.get("toward"),
                     raw)


@dataclass
class FractureStudy:
    part: str
    material: str
    loads: list[Load] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    crack: CrackSpec | None = None
    r_ratio: float = 0.1
    spectrum: str = ""            # file of max,min load factors, one cycle per line
    law: str = "nasgro"           # nasgro or paris
    growth: str = ""              # custom constants C=..,n=..,dkth=..,kic=..
    kc: float | None = None       # toughness override, MPa sqrt(m)
    wheeler: float = 0.0
    target_life: float | None = None
    max_pof: float = 0.01         # allowed share of parts failing before the target life
    detectable: float = 1.27      # crack size an inspection reliably finds, mm
    flaw_scatter: float = 0.3
    growth_scatter: float | None = None
    load_scatter: float = 0.05
    samples: int = 20000
    steps: int = 12
    budget: float = 1200.0
    quality: str = "normal"
    question: str = ""
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    keep: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def study_from(part: str, material: str, loads, fixtures, crack: str, **kw) -> FractureStudy:
    """A FractureStudy from the strings a CLI or MCP call carries."""
    if not material:
        raise SpecError("crack growth needs a material, for example -m 7075-t6")
    st = FractureStudy(part, material, [parse_load(x) for x in loads], [parse_fixture(x) for x in fixtures],
                       parse_crack(crack), **{k: v for k, v in kw.items() if v is not None})
    if not st.loads or not st.fixtures:
        raise SpecError("crack growth needs at least one --load and one --fix")
    if not -5.0 <= st.r_ratio < 1.0:
        raise SpecError(f"R ratio {st.r_ratio:g} is outside -5 to 1")
    if not 0 < st.max_pof < 1:
        raise SpecError("the allowed share of failures must be between 0 and 1")
    return st


def as_vector(v) -> np.ndarray | None:
    return None if v is None else np.asarray(v, float)
