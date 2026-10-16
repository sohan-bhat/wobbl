"""A typical rocket built from a few words, for a flight asked about with no spec.

The motor comes from its designation when the words give one ("a C6 rocket", "on an H128", "an I class rocket"),
and is a C6-5, the most flown model rocket motor, when they do not. Its thrust curve is a typical one for the class
and the average thrust the designation names: a black powder spike and sustain up to E, a flatter composite burn
from F on. The airframe is sized to the motor: a body tube a little wider than it, twenty calibers long, an ogive
nose, three fins and one parachute at apogee for about 4.5 m/s under canopy.
"""
from __future__ import annotations

import math
import re

LETTERS = "ABCDEFGHIJKLMNO"
# motor diameter in mm by class, and the length of a typical motor of that diameter
DIAMETER = {"A": 18, "B": 18, "C": 18, "D": 24, "E": 24, "F": 29, "G": 29, "H": 38, "I": 38, "J": 54, "K": 54,
            "L": 75, "M": 75, "N": 98, "O": 98}
LENGTH = {18: 70, 24: 95, 29: 124, 38: 200, 54: 320, 75: 500, 98: 700}
THRUST = {"A": 3, "B": 4, "C": 6, "D": 12, "E": 9, "F": 30, "G": 40, "H": 128, "I": 200, "J": 350, "K": 550,
          "L": 1000, "M": 1800, "N": 3000, "O": 5000}


def top(letter: str) -> float:
    return 2.5 * 2 ** LETTERS.index(letter)


def designation(text: str) -> tuple[str, float | None, str]:
    """The motor the words name, as (class, average thrust or None, delay). 'C6-5' is (C, 6, 5)."""
    low = text.lower()
    m = re.search(r"\b([a-o])(\d{1,4})(?:-(\d{1,2}))?[a-z]?\b", low)
    if m and re.search(r"\b(?:motor|engine|rocket|reload|on an?)\b", low):
        return m.group(1).upper(), float(m.group(2)), m.group(3) or ""
    m = re.search(r"\b([a-o])[\s-]?class\b", low) or re.search(r"\b(?:an?|the|my)\s+([b-o])\s+(?:motor|engine)s?\b",
                                                              low)
    if m:
        return m.group(1).upper(), None, ""
    return "", None, ""


def curve(letter: str, avg: float) -> tuple[list[list[float]], float, float]:
    """A typical thrust curve, (seconds, newtons) points, with its total impulse and propellant mass."""
    impulse = 0.8 * top(letter)
    tb = impulse / avg
    powder = letter in "ABCDE"
    shape = ([(0, 0), (0.08, 2.4), (0.2, 1.2), (0.85, 0.85), (1.0, 0)] if powder else
             [(0, 0), (0.05, 1.25), (0.5, 1.05), (0.8, 0.95), (1.0, 0)])
    area = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(shape, shape[1:]))
    pts = [[round(f * tb, 4), round(v * avg / area, 3)] for f, v in shape]
    isp = 80.0 if powder else 190.0
    return pts, impulse, impulse / (isp * 9.81)


def design(text: str) -> tuple[dict, str]:
    """A rocket spec as read_spec takes it, and the rocket in one plain sentence."""
    letter, avg, delay = designation(text)
    named = bool(letter)
    letter = letter or "C"
    avg = avg or THRUST[letter]
    if not named:
        delay = "5"
    pts, impulse, prop = curve(letter, avg)
    md = DIAMETER[letter]
    loaded = prop * (2.0 if letter in "ABCDE" else 1.9)
    body = round(md * 1.4)
    length = 20 * body / 1000
    mass = 0.05 * (body / 25) ** 2.2 * (length / 0.5)
    weight = (mass + loaded) * 9.81
    chute = math.sqrt(4 * 2 * weight / (1.225 * 4.5 ** 2 * 1.5) / math.pi)
    d = body / 1000
    name = f"{letter}{avg:g}" + (f"-{delay}" if delay else "")
    spec = {"name": f"typical-{name.lower()}",
            "rocket": {"mass": f"{mass * 1000:.0f} g", "diameter": f"{body} mm", "length": f"{length:.2f} m",
                       "cg": f"{0.67 * length:.3f} m", "cd": 0.5},
            "nose": {"shape": "ogive", "length": f"{3 * body} mm"},
            "fins": {"count": 3, "root": f"{2.4 * d:.3f} m", "tip": f"{1.0 * d:.3f} m", "span": f"{1.4 * d:.3f} m",
                     "sweep": f"{1.2 * d:.3f} m"},
            "motor": {"name": name, "loaded_mass": f"{loaded * 1000:.1f} g", "propellant": f"{prop * 1000:.1f} g",
                      "length": f"{LENGTH[md]} mm", "thrust": pts},
            "launch": {"rail_length": "0.9 m" if md <= 24 else "1.8 m", "rail_angle": "0 deg"},
            "main": {"diameter": f"{chute:.2f} m", "cd": 1.5}}
    article = "an" if letter in "AEFHILMNO" else "a"
    said = (f"a typical {body} mm rocket, {length:.2f} m long and {mass * 1000:.0f} g, on {article} {name}"
            f"{'' if named else ', the most flown model rocket motor'}, burning {impulse:.3g} N s at {avg:g} N "
            f"average along a {'black powder' if letter in 'ABCDE' else 'composite'} motor's usual curve, with an "
            f"ogive nose, three fins and a {chute:.2f} m parachute at apogee")
    return spec, said


def toml(d: dict) -> str:
    """The spec as TOML text, for the plain tables and values a design holds."""
    def val(v):
        if isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return repr(v)

    lines = [f"{k} = {val(v)}" for k, v in d.items() if not isinstance(v, dict)]
    for k, v in d.items():
        if isinstance(v, dict):
            lines += ["", f"[{k}]"] + [f"{a} = {val(b)}" for a, b in v.items()]
    return "\n".join(lines) + "\n"
