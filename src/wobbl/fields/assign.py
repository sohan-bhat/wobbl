"""What each body of a general scene is made of, read from short lines like "plunger: steel 1010".

A line is a body selector (see scene.Scene.select), a colon or an equals sign, then the material and what drives
it. The same lines work for every physics, and each physics reads the words it understands:

    magnet: n42 magnet +z            a permanent magnet magnetized along +z (or "along 0,0,1")
    core: steel 1010                 nonlinear steel from its BH curve (any magnetic material name)
    shield: mur 5000                 a linear magnetic material
    winding: coil 800 turns 0.6 A    a coil, its axis found from its hole (or "axis z", "axis 0,1,0 at 0,0,5")
    electrode: conductor 5 kV        a perfect conductor held at a voltage ("ground", "0 V" or "floating")
    spacer: ptfe                     a dielectric by name (or "er 3.2")
    case: rigid                      a sound hard wall around an air cavity
    cone: vibrating 2 mm/s           a surface that moves the air (sound)
    patch: metal                     a perfect conductor for antennas
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .params import ParamError, value
from .scene import Scene

NONMAGNETIC = {"air", "copper", "aluminum", "aluminium", "plastic", "brass", "ss304", "304", "abs", "pla", "nylon",
               "wood", "glass", "ptfe", "fr4", "nonmagnetic", "non-magnetic", "none"}


@dataclass
class Assignment:
    body: str
    text: str
    words: str                    # lower case, without the selector


@dataclass
class Assigned:
    items: list[Assignment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def of(self, name: str) -> Assignment | None:
        return next((a for a in self.items if a.body == name), None)


def parse(scene: Scene, lines: list[str]) -> Assigned:
    out = Assigned()
    for line in lines:
        m = re.match(r"^\s*(.+?)\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            raise ParamError(f"'{line}' needs a body and a material, like 'plunger: steel 1010'")
        for b in scene.select(m.group(1)):
            out.items = [a for a in out.items if a.body != b.name]
            out.items.append(Assignment(b.name, line, m.group(2).lower()))
    return out


def _vec(t: str) -> np.ndarray:
    ax = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1),
          "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    t = t.strip()
    if t in ax:
        return np.array(ax[t], float)
    v = np.array([float(x) for x in re.split(r"[,\s]+", t) if x], float)
    if v.shape != (3,) or not np.linalg.norm(v):
        raise ParamError(f"'{t}' is not a direction. Use +x, -z or three numbers like 0,0,1")
    return v / np.linalg.norm(v)


def _point(t: str) -> np.ndarray:
    v = np.array([float(x) for x in re.split(r"[,\s]+", t.strip()) if x], float)
    if v.shape != (3,):
        raise ParamError(f"'{t}' is not a point. Give three numbers in mm like 0,0,5")
    return v * 1e-3


DIR = r"([+-]?[xyz]\b|[-+\d.e]+\s*,\s*[-+\d.e]+\s*,\s*[-+\d.e]+)"


# --- magnetic ---------------------------------------------------------------------------------
@dataclass
class MagBody:
    name: str
    kind: str                     # steel, linear, magnet, coil
    material: str = ""
    mur: float = 1.0
    bh: tuple | None = None
    br: np.ndarray | None = None  # remanence vector, T
    hcj: float = 0.0
    turns: float = 0.0
    current: float = 0.0
    axis: np.ndarray | None = None
    center: np.ndarray | None = None


def magnetic(scene: Scene, a: Assigned) -> tuple[dict[str, MagBody], list[str]]:
    """The magnetic role of every assigned body. Bodies with no magnetic role are left out of the mesh (air)."""
    from ..materials import get_magnetic

    out, notes = {}, []
    for it in a.items:
        w = it.words
        b = next(x for x in scene.bodies if x.name == it.body)
        if re.search(r"\bcoil\b|\bturns?\b|\bwinding\b", w):
            m = re.search(r"(\d+(?:\.\d+)?)\s*turns?", w)
            turns = float(m.group(1)) if m else 1.0
            cur = re.search(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(ma|a)\b", w)
            if not cur:
                raise ParamError(f"'{it.text}' needs a current, like 'coil 800 turns 0.6 A'")
            current = float(cur.group(1)) * (1e-3 if cur.group(2) == "ma" else 1.0)
            ax = re.search(r"axis\s+" + DIR, w)
            axis = _vec(ax.group(1)) if ax else (np.array(b.hole_axis) if b.hole_axis else None)
            if axis is None:
                raise ParamError(f"could not find the winding axis of '{b.name}' from its shape. Say it, like "
                                 f"'{b.name}: coil {turns:g} turns {current:g} A axis z'")
            if not ax:
                notes.append(f"The winding axis of '{b.name}' is its hole's axis, {_fmt_vec(axis)}; current "
                             "circulates right handed about it (write axis -z to reverse it).")
            at = re.search(r"\bat\s+([-\d.e+]+\s*,\s*[-\d.e+]+\s*,\s*[-\d.e+]+)", w)
            center = _point(at.group(1)) if at else np.array(b.center)
            out[b.name] = MagBody(b.name, "coil", "copper coil", turns=turns, current=current, axis=axis,
                                  center=center)
            continue
        m = re.search(r"\bmur\s*=?\s*(\d+(?:\.\d+)?)", w)
        if m:
            out[b.name] = MagBody(b.name, "linear", f"mu_r {m.group(1)}", mur=float(m.group(1)))
            continue
        name = re.sub(r"\b(magnet|magnetized|along|steel core|core)\b", " ", w)
        name = re.sub(r"\s*" + DIR + r"\s*$", "", name).strip() or "1010"
        mat = None
        for cand in (name, name.replace(" steel", ""), name.split()[0] if name.split() else name):
            try:
                mat = get_magnetic(cand)
                break
            except KeyError:
                continue
        if mat is None and re.search(r"\bsteel\b|\biron\b", w):
            mat = get_magnetic("1010")
        if mat is None:
            if any(t in NONMAGNETIC for t in w.split()):
                notes.append(f"'{b.name}' ({w}) is not magnetic, so the field passes through it like air.")
                continue
            raise ParamError(f"'{it.text}' names no magnetic material. Use a steel (1010, m19, pure-iron), a magnet "
                             "grade (n42, ferrite) with a direction, 'mur 3000', or 'coil N turns I A'")
        if mat.br:
            d = re.search(DIR + r"\s*$", w) or re.search(r"along\s+" + DIR, w)
            if not d:
                raise ParamError(f"which way is '{b.name}' magnetized? Add a direction, like '{b.name}: {mat.key} +z'")
            out[b.name] = MagBody(b.name, "magnet", mat.name, mur=mat.mur, br=mat.br * _vec(d.group(1)),
                                  hcj=mat.hcj)
        elif mat.bh:
            out[b.name] = MagBody(b.name, "steel", mat.name, bh=mat.bh)
        else:
            notes.append(f"'{b.name}' ({mat.name}) is not magnetic, so the field passes through it like air.")
    return out, notes


def _fmt_vec(v) -> str:
    v = np.asarray(v, float)
    for i, n in enumerate("xyz"):
        if abs(abs(v[i]) - 1) < 1e-6:
            return ("+" if v[i] > 0 else "-") + n
    return ", ".join(f"{x:.3g}" for x in v)


# --- electric -----------------------------------------------------------------------------------
def electric(scene: Scene, a: Assigned):
    """Conductors with their voltages (None when floating), and dielectrics as (er, strength V/m, name)."""
    from ..materials import get_dielectric

    conds, diel, notes = {}, {}, []
    for it in a.items:
        w = it.words
        if re.search(r"\bconductor|\belectrode|\bground(?:ed)?\b|\bfloating\b|\bmetal\b|\bcopper\b|\bv\b|\bkv\b|"
                     r"\d\s*(?:k|m)?v\b", w):
            if "floating" in w:
                conds[it.body] = None
            elif re.search(r"\bground(?:ed)?\b", w):
                conds[it.body] = 0.0
            else:
                m = re.search(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(kv|mv|v)\b", w)
                if not m:
                    raise ParamError(f"'{it.text}' needs a voltage, like 'conductor 5 kV', 'ground' or 'floating'")
                conds[it.body] = value({"v": f"{m.group(1)} {m.group(2)}"}, "v", 0, "voltage")
            continue
        m = re.search(r"\b(?:er|epsr|permittivity)\s*=?\s*(\d+(?:\.\d+)?)", w)
        if m:
            diel[it.body] = (float(m.group(1)), 10e6, f"a dielectric with er {m.group(1)}")
            notes.append(f"'{it.body}' has no named material, so Wobbl takes 10 kV/mm for its strength.")
            continue
        try:
            d = get_dielectric(w.replace("dielectric", "").strip() or "air")
        except KeyError:
            raise ParamError(f"'{it.text}' is neither a conductor nor a known dielectric. Use 'conductor 5 kV', "
                             "a dielectric name (fr4, ptfe, pe, glass, alumina, polyimide, mineral-oil) or 'er 3.2'"
                             ) from None
        diel[it.body] = (d.epsr, d.strength * 1e6, d.name)
    return conds, diel, notes


# --- sound --------------------------------------------------------------------------------------
@dataclass
class SoundRoles:
    air: list[str] = field(default_factory=list)        # bodies that are air (cavities, rooms)
    walls: list[str] = field(default_factory=list)      # solid bodies whose inside air is wanted
    sources: dict[str, float] = field(default_factory=dict)   # body whose surface vibrates: velocity m/s


def sound(scene: Scene, a: Assigned) -> SoundRoles:
    r = SoundRoles()
    for it in a.items:
        w = it.words
        m = re.search(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(mm/s|m/s|um/s)", w)
        if "vibrat" in w or "speaker" in w or "source" in w or m:
            if not m:
                raise ParamError(f"'{it.text}' needs a speed, like 'vibrating 1 mm/s'")
            r.sources[it.body] = float(m.group(1)) * {"m/s": 1.0, "mm/s": 1e-3, "um/s": 1e-6}[m.group(2)]
        elif re.search(r"\bair\b|\bcavity\b|\broom\b|\binside\b", w):
            r.air.append(it.body)
        elif re.search(r"\brigid\b|\bwall|\benclosure\b|\bbox\b|\bcase\b|\bshell\b", w):
            r.walls.append(it.body)
        else:
            raise ParamError(f"'{it.text}': for sound a body is 'air' (the volume the sound lives in), 'rigid' (a "
                             "solid whose inside cavity holds the air) or 'vibrating 1 mm/s'")
    return r
