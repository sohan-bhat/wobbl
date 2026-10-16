"""A typical hobby solid motor designed from a few words, for a static fire asked about with no spec.

With nothing said it is the 54 mm KNSB motor of examples/knsb-motor: five BATES grains 48 mm across with 16 mm cores
in a 6061-T6 casing, about a J class. Words can name the propellant (KNSB, KNDX, KNSU or sugar), the casing size
("a 38 mm motor") or the class ("an H motor", "I class", "H128"), and the design is sized to them: the grain count
for the class's impulse, the throat for a Kn near the example's, the exit for an area ratio near 6.
"""
from __future__ import annotations

import math
import re

LETTERS = "ABCDEFGHIJKLMNO"
# casing outside diameter and wall, and the liner inside it, in mm
CASINGS = {29: (29.0, 1.25, 1.0), 38: (38.1, 1.5, 1.2), 54: (54.0, 1.6, 1.4), 75: (76.2, 2.4, 1.5), 98: (98.0, 3.2, 2.0)}
DENSITY = 1.80e-6        # kg per mm3, sugar propellants as cast
IMPULSE = 1140.0         # N s per kg delivered, from the 54 mm example's 1238 N s on 1.09 kg
KN = 276.0               # burning area over throat area at ignition, the example's
EXAMPLE = {"propellant": "knsb", "grain": "bates d=48 core=16 L=75 x5", "throat": "12.7 mm", "exit": "32 mm",
           "casing": "6061-t6 od=54 wall=1.6 L=400 sf=2", "efficiency": "0.85"}


def top(letter: str) -> float:
    """The highest total impulse of a motor class, in N s: A is up to 2.5 and each letter doubles it."""
    return 2.5 * 2 ** LETTERS.index(letter.upper())


def class_of(impulse: float) -> str:
    for c in LETTERS:
        if impulse <= top(c):
            return c
    return LETTERS[-1]


def asked_class(text: str) -> str | None:
    """A motor class named in words: 'an H motor', 'I class', 'H128' or 'J350-14'."""
    low = text.lower()
    m = re.search(r"\b([a-o])[\s-]?class\b", low) or \
        re.search(r"\b([a-o])\d{1,4}(?:-\d{1,2}[a-z]?)?\b(?=[^.]*\b(?:motor|engine|rocket|reload)s?\b)", low) or \
        re.search(r"\b(?:an?|the|my)\s+([b-o])\s+(?:motor|engine|reload)s?\b", low)
    return m.group(1).upper() if m else None


def asked_diameter(text: str) -> int | None:
    m = re.search(r"\b(\d{2,3})\s*mm\b", text.lower())
    if not m:
        return None
    d = float(m.group(1))
    return min(CASINGS, key=lambda c: abs(c - d))


def propellant_in(text: str) -> str | None:
    m = re.search(r"\b(knsb|kndx|knsu)\b", text.lower())
    if m:
        return m.group(1)
    return "knsb" if re.search(r"\bsugar\b|\bsorbitol\b|\brcandy\b|\brocket candy\b", text.lower()) else None


def _grain(dia: int) -> tuple[float, float, float]:
    od, wall, liner = CASINGS[dia]
    d = math.floor(od - 2 * wall - 2 * liner)
    core = round(d / 3)
    return d, core, round(1.55 * d)


def design(text: str) -> tuple[dict[str, str], str]:
    """The motor as the option values a spec from flags takes, and the design in one plain sentence."""
    prop = propellant_in(text) or "knsb"
    cls, dia = asked_class(text), asked_diameter(text)
    if not cls and dia in (None, 54):
        flags = dict(EXAMPLE, propellant=prop)
        d, core, length, n, casing_od = 48, 16, 75, 5, 54
    else:
        target = 0.7 * top(cls) if cls else None
        if dia is None:
            dia = next(c for c, lim in ((29, 80), (38, 640), (54, 2560), (75, 10240), (98, 1e9)) if target <= lim)
        d, core, length = _grain(dia)
        per = math.pi / 4 * (d ** 2 - core ** 2) * length * DENSITY * IMPULSE
        n = max(1, min(8, round(target / per))) if target else 4
        burn = n * (math.pi * core * length + 2 * math.pi / 4 * (d ** 2 - core ** 2))
        throat = math.sqrt(4 * burn / KN / math.pi)
        od, wall, _ = CASINGS[dia]
        casing_od = dia
        flags = {"propellant": prop, "grain": f"bates d={d:g} core={core:g} L={length:g} x{n}",
                 "throat": f"{throat:.1f} mm", "exit": f"{2.5 * throat:.1f} mm",
                 "casing": f"6061-t6 od={od:g} wall={wall:g} L={n * length + 25:g} sf=2", "efficiency": "0.85"}
    impulse = n * math.pi / 4 * (d ** 2 - core ** 2) * length * DENSITY * IMPULSE
    said = (f"a typical {casing_od} mm {prop.upper()} hobby motor of {n} BATES grain{'s' if n > 1 else ''} {d:g} mm "
            f"across with {core:g} mm cores, a throat of {flags['throat']} and a 6061-T6 casing, sized for about "
            f"{impulse:.0f} N s, {'an' if class_of(impulse) in 'AEFHILMNO' else 'a'} {class_of(impulse)} class")
    if cls and class_of(impulse) != cls:
        said += f" (the nearest Wobbl can cast to the {cls} class asked)"
    return flags, said
