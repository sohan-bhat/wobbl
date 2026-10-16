"""Typical things to run on when the user names one but gives no file: a bracket, a heat sink, a phone case.

A question like "will a bracket hold 200 N" or "drop my phone case from 1.5 m" names a part without its CAD. Rather
than stop and ask for a file, the router picks the closest typical part here, runs on it, and records the choice
as an assumption with how to pass the real one. Each part brings the supports, the load place and the material
its own example uses, each applied only when the user gives none and each recorded as a guess too.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .intent import best

PARTS = Path(__file__).resolve().parent / "parts"
STATIC = ("stress", "plastic", "buckling", "fatigue", "vibration", "submodel", "crack")
FORCE = r"(\d+(?:\.\d+)?)\s*(k?n|newtons?|lbf|kgf)\b"


@dataclass(frozen=True)
class Default:
    """A value a typical part brings: the option, its values, what they mean in words and the analyses that take it.
    A value with {force} in it needs a force in the question and is left out without one."""
    option: str
    values: tuple[str, ...]
    means: str
    analyses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stock:
    file: str
    what: str                      # "a 3D printed L shelf bracket, 80 mm tall with a 90 mm arm"
    words: tuple[str, ...] = ()    # what picks it over the others
    defaults: tuple[Default, ...] = field(default_factory=tuple)

    @property
    def path(self) -> Path:
        return PARTS / self.file


def _mat(name: str, why: str) -> Default:
    return Default("material", (name,), f"material {why}")


PART_STOCK = (
    Stock("bracket.py", "a typical L shelf bracket, 80 mm tall with a 90 mm arm, 6 mm thick and 25 mm wide",
          (r"\bbrackets?\b", r"\bshelf\b", r"\bmounts?\b", r"\bangle\b", r"\bpart\b"),
          (Default("fix", ("bolt holes d=4.5", "wall x=min"), "screwed to a wall through the two holes in its upright",
                   STATIC),
           Default("load", ("{force} down @ top and within 12 of 65,12.5,6",), "the load pressing down near the end "
                                                                               "of the arm", STATIC))),
    Stock("heatsink.py", "a typical extruded aluminum heat sink, 60 by 60 mm with nine 30 mm fins",
          (r"heat\s*sinks?", r"\bfins?\b", r"\bchip\b", r"\bleds?\b", r"\bcpu\b"),
          (_mat("6063-t5", "extruded 6063-T5 aluminum, as heat sinks are"),
           Default("heat-faces", ("bottom and box 17.5,17.5,-1,42.5,42.5,0.5",),
                   "the heat going in through a 25 mm square at the middle of the base", ("heat", "cooling")))),
    Stock("case.py", "a typical handheld housing standing in for the case, 100 by 55 by 22 mm with 2.5 mm "
                     "polycarbonate walls", (r"\b(?:phone|camera|device|handheld|remote|tablet)\b", r"\bcases?\b",
                                             r"\bhousings?\b"),
          (_mat("pc", "polycarbonate, as most hard cases are"),
           Default("mass", ("150 g @ 0,0,6 r=26",), "a 150 g phone inside it, held in its cradle", ("drop",)))),
    Stock("enclosure.py", "a typical electronics enclosure top shell, 90 by 60 by 28 mm with 3 mm walls",
          (r"\benclosures?\b", r"\bbox\b", r"\bshell\b"),
          (_mat("pc", "polycarbonate, as most molded enclosures are"),
           Default("fix", ("z=min",), "resting on its rim", STATIC),
           Default("load", ("{force} down @ top and within 8 of 25,15,28",), "the load pressing on the top", STATIC))),
    Stock("arm.py", "a typical printed quadcopter arm, 150 mm from root to motor",
          (r"\barms?\b", r"\bdrone\b", r"\bquad"),
          (_mat("pa12", "printed PA12, as many drone arms are"),
           Default("fix", ("bolt holes d=3.2 and x<0",), "bolted into the frame at its root", STATIC))),
    Stock("strut.py", "a typical aluminum tube strut, 600 mm long, 20 mm across with a 3 mm wall",
          (r"\bstruts?\b", r"\btubes?\b", r"\bcolumns?\b", r"\bbuckl"),
          (_mat("6061-t6", "6061-T6 aluminum"),
           Default("fix", ("z=min", "fixed-xy z=max"), "held at the bottom and kept from swaying at the top", STATIC),
           Default("load", ("{force} -z @ z=max",), "the load pressing straight down on the top", STATIC))),
    Stock("hook.py", "a typical forged steel J hook that bolts up through the eye in its shank",
          (r"\bhooks?\b",),
          (_mat("mild-steel", "mild steel, as forged hooks are"),
           Default("fix", ("bolt holes d=8.5",), "bolted up through its eye", STATIC),
           Default("load", ("{force} -z @ within 5 of 22,0,-16",), "the load hanging in the bottom of the bend",
                   STATIC))),
    Stock("lid.py", "a typical printed box lid, 220 by 60 mm", (r"\blids?\b", r"\bcovers?\b"),
          (_mat("abs", "ABS, a common printed plastic"),)),
)


def pick(stock: tuple[Stock, ...], text: str) -> Stock | None:
    """The typical thing whose words weigh most in the text, the first one when none fit better."""
    if not stock:
        return None
    scored = [(best(s.words, text)[0], -i, s) for i, s in enumerate(stock)]
    return max(scored, key=lambda r: (r[0], r[1]))[2]


def force_in(text: str) -> str | None:
    m = re.search(FORCE, text or "", re.I)
    if not m:
        return None
    unit = m.group(2).lower()
    unit = {"newton": "N", "newtons": "N", "n": "N", "kn": "kN"}.get(unit, unit)
    return f"{m.group(1)} {unit}"


def defaults(s: Stock, text: str, analyses: list[str]) -> list[tuple[str, list[str], str]]:
    """The values the part brings for this plan's analyses, as (option, values, what they mean), with {force}
    filled from the question and left out when the question gives none."""
    from ..ask import find_material

    out = []
    force = force_in(text)
    named = find_material(text)
    for d in s.defaults:
        if d.analyses and not set(d.analyses) & set(analyses) or d.option == "material" and named:
            continue
        if any("{force}" in v for v in d.values):
            if not force:
                continue
            out.append((d.option, [v.replace("{force}", force) for v in d.values], d.means))
        else:
            out.append((d.option, list(d.values), d.means))
    return out
