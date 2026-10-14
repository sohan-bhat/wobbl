"""The common board model every reader fills in.

All lengths are in meters. x points right and y points up, as on the board
seen from the top, and z is measured up from the bottom face of the board.
Copper layers keep KiCad's names (F.Cu, In1.Cu ... B.Cu) whichever file
they came from, so a board read from Gerbers lines up with its KiCad file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import geom

UM = 1e-6
MM = 1e-3


@dataclass
class Layer:
    """One copper layer. Index 0 is the top copper."""

    name: str
    kind: str = "copper"
    index: int = 0
    z_bottom: float = 0.0
    thickness: float = 35 * UM

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.thickness

    @property
    def z_mid(self) -> float:
        return self.z_bottom + self.thickness / 2.0


@dataclass
class Dielectric:
    """One dielectric sheet (core, prepreg) between copper layers."""

    name: str
    z_bottom: float
    thickness: float
    er: float = 4.5
    loss_tangent: float = 0.02
    material: str = "FR4"

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.thickness


@dataclass
class Stackup:
    """Copper layers and dielectrics, both listed top to bottom, plus where the numbers came from."""

    layers: list[Layer]
    dielectrics: list[Dielectric]
    board_thickness: float
    source: str

    def names(self) -> list[str]:
        return [lay.name for lay in self.layers]

    def layer(self, name: str) -> Layer:
        """The copper layer with this name."""
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise ValueError(f"The board has no copper layer named {name!r}. Its copper layers are "
                         f"{', '.join(self.names())}.")

    def _gap(self, a: str, b: str) -> tuple[float, float]:
        la, lb = self.layer(a), self.layer(b)
        lo, hi = (la, lb) if la.z_bottom <= lb.z_bottom else (lb, la)
        return lo.z_top, hi.z_bottom

    def height_between(self, a: str, b: str) -> float:
        """Total dielectric thickness between two copper layers."""
        z0, z1 = self._gap(a, b)
        return sum(max(0.0, min(d.z_top, z1) - max(d.z_bottom, z0)) for d in self.dielectrics)

    def er_between(self, a: str, b: str) -> float:
        """Thickness weighted relative permittivity of the dielectric between two copper layers."""
        z0, z1 = self._gap(a, b)
        num = den = 0.0
        for d in self.dielectrics:
            t = max(0.0, min(d.z_top, z1) - max(d.z_bottom, z0))
            num += t * d.er
            den += t
        if den > 0:
            return num / den
        near = min(self.dielectrics, key=lambda d: abs((d.z_bottom + d.z_top) / 2 - (z0 + z1) / 2), default=None)
        return near.er if near else 1.0

    def gap_between(self, a: str, b: str) -> float:
        """Vertical distance between the facing surfaces of two copper layers, any copper in between included."""
        z0, z1 = self._gap(a, b)
        return max(0.0, z1 - z0)

    def distance(self, a: str, b: str) -> float:
        """Distance between the mid planes of two copper layers, the length a via runs between them."""
        return abs(self.layer(a).z_mid - self.layer(b).z_mid)


def build_stackup(sheets: list[tuple], source: str) -> Stackup:
    """Stack sheets listed from top to bottom and measure z up from the bottom face.

    Each sheet is ("copper", name, thickness) or ("dielectric", name, thickness, er, loss_tangent, material).
    """
    z = 0.0
    layers, diel = [], []
    for s in reversed(sheets):
        if s[0] == "copper":
            layers.append(Layer(s[1], "copper", 0, z, s[2]))
        else:
            diel.append(Dielectric(s[1], z, s[2], s[3], s[4], s[5]))
        z += s[2]
    layers.reverse()
    diel.reverse()
    for i, lay in enumerate(layers):
        lay.index = i
    return Stackup(layers, diel, z, source)


def default_stackup(names: list[str], thickness: float | None = None) -> Stackup:
    """A plausible stackup for a board whose file does not describe one.

    2 layers is 1.6 mm FR4 with er 4.5 and 35 um copper. 4 layers is a JLCPCB like stack
    with 0.2 mm 7628 prepreg outside, a 1.065 mm core, 35 um outer and 17.5 um inner copper.
    Other counts get equal dielectrics. A board thickness from the file scales the dielectrics.
    """
    n = len(names)
    if n == 0:
        raise ValueError("The board has no copper layers, so there is nothing to simulate. Check that the "
                         "file is a printed circuit board with copper.")
    if n == 1:
        sheets = [("copper", names[0], 35 * UM), ("dielectric", "core", 1.6 * MM - 35 * UM, 4.5, 0.02, "FR4")]
        source = "default 1.6 mm FR4 er 4.5, 35 um copper, single sided"
    elif n == 2:
        sheets = [("copper", names[0], 35 * UM), ("dielectric", "core", 1.6 * MM - 70 * UM, 4.5, 0.02, "FR4"),
                  ("copper", names[1], 35 * UM)]
        source = "default 1.6 mm FR4 er 4.5, 35 um copper"
    elif n == 4:
        sheets = [("copper", names[0], 35 * UM), ("dielectric", "prepreg 1", 0.2 * MM, 4.4, 0.02, "7628 prepreg"),
                  ("copper", names[1], 17.5 * UM), ("dielectric", "core", 1.065 * MM, 4.6, 0.02, "FR4 core"),
                  ("copper", names[2], 17.5 * UM), ("dielectric", "prepreg 2", 0.2 * MM, 4.4, 0.02, "7628 prepreg"),
                  ("copper", names[3], 35 * UM)]
        source = ("default 4 layer JLCPCB like stack, 0.2 mm 7628 prepreg, 1.065 mm core, 35 um outer and 17.5 um "
                  "inner copper")
    else:
        cu = [35 * UM if i in (0, n - 1) else 17.5 * UM for i in range(n)]
        t = (1.6 * MM - sum(cu)) / (n - 1)
        sheets = []
        for i, name in enumerate(names):
            sheets.append(("copper", name, cu[i]))
            if i < n - 1:
                core = i % 2 == 1
                sheets.append(("dielectric", f"{'core' if core else 'prepreg'} {i + 1}", t, 4.6 if core else 4.4,
                               0.02, "FR4 core" if core else "FR4 prepreg"))
        source = f"default {n} layer 1.6 mm FR4 stack with equal dielectrics, 35 um outer and 17.5 um inner copper"
    total = sum(s[2] for s in sheets)
    if thickness and abs(thickness - total) > 0.02 * total:
        cu_total = sum(s[2] for s in sheets if s[0] == "copper")
        k = max(thickness - cu_total, 0.05 * MM) / (total - cu_total)
        sheets = [s if s[0] == "copper" else (s[0], s[1], s[2] * k) + tuple(s[3:]) for s in sheets]
        source += f", dielectrics scaled to the {thickness / MM:.3g} mm board thickness in the file"
    return build_stackup(sheets, source)


@dataclass
class Track:
    """A straight piece of trace. Arcs are split into short tracks."""

    net: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])


@dataclass
class Via:
    """A plated hole joining copper from layers[0] (upper) down to layers[1] (lower)."""

    net: str
    at: tuple[float, float]
    drill: float
    diameter: float
    layers: tuple[str, str]


@dataclass
class Pad:
    """A footprint pad at its absolute position. angle_deg turns the pad counter clockwise as seen from the top."""

    ref: str
    number: str
    net: str
    layers: list[str]
    at: tuple[float, float]
    size: tuple[float, float]
    shape: str
    angle_deg: float = 0.0
    drill: float = 0.0

    @property
    def name(self) -> str:
        """REF.PIN, or an empty string for a pad the files do not name."""
        return f"{self.ref}.{self.number}" if self.ref or self.number else ""

    def contains(self, points, margin: float = 0.0) -> np.ndarray:
        """Whether points lie on the pad (circle, oval, or rotated rectangle for every other shape)."""
        return geom.shape_contains(self.at, self.size, self.shape, self.angle_deg, points, margin)

    def outline(self, n: int = 16) -> np.ndarray:
        """Sample points on the pad edge, used to test overlap with zones."""
        w, h = self.size
        if self.shape == "circle":
            return geom.circle_ring(self.at, max(w, h) / 2.0, n)
        hw, hh = w / 2.0, h / 2.0
        local = np.array([[hw, 0], [hw, hh], [0, hh], [-hw, hh], [-hw, 0], [-hw, -hh], [0, -hh], [hw, -hh]])
        a = math.radians(self.angle_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return local @ rot.T + np.asarray(self.at)


@dataclass
class Zone:
    """Poured copper on one layer, as filled polygons that each have an outer ring and a list of hole rings.

    stroke is the outline pen width KiCad 5 draws around its filled polygons. The copper then reaches
    stroke / 2 beyond the rings. It is 0 for KiCad 6 and newer and for Gerber regions.
    """

    net: str
    layer: str
    polygons: list[tuple[np.ndarray, list[np.ndarray]]]
    stroke: float = 0.0

    def area(self) -> float:
        """Copper area in square meters (holes that overlap each other are subtracted twice)."""
        return sum(geom.ring_area(o) - sum(geom.ring_area(h) for h in holes) for o, holes in self.polygons)

    def contains(self, points) -> np.ndarray:
        """Whether points lie inside any filled polygon of the zone."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.zeros(len(pts), dtype=bool)
        for outer, holes in self.polygons:
            todo = ~out
            if todo.any():
                out[todo] |= geom.points_in_polygon(pts[todo], outer, holes)
        return out

    def bbox(self) -> tuple[float, float, float, float]:
        allp = np.vstack([o for o, _ in self.polygons])
        return float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max())


class Items(NamedTuple):
    """Board items grouped by kind, as returned by Board.items_on and Board.net_items."""

    tracks: list[Track]
    vias: list[Via]
    pads: list[Pad]
    zones: list[Zone]


@dataclass
class Board:
    """A printed circuit board read from KiCad or Gerber files."""

    name: str
    source: str
    stackup: Stackup
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    pads: list[Pad] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    nets: list[str] = field(default_factory=list)
    outline: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _by_net: dict | None = field(default=None, init=False, repr=False, compare=False)

    def copper_layers(self) -> list[str]:
        """Copper layer names from top to bottom."""
        return self.stackup.names()

    def bbox(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the board edge, or of all copper when there is no edge."""
        parts = list(self.outline)
        if not parts:
            parts = [np.array([t.start, t.end]) for t in self.tracks]
            parts += [np.array([p.at]) for p in self.pads] + [np.array([v.at]) for v in self.vias]
            parts += [o for z in self.zones for o, _ in z.polygons]
        if not parts:
            return 0.0, 0.0, 0.0, 0.0
        allp = np.vstack(parts)
        return float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max())

    def via_layers(self, via: Via) -> list[str]:
        """Copper layers a via passes through, top to bottom."""
        names = self.copper_layers()
        try:
            i, j = sorted((names.index(via.layers[0]), names.index(via.layers[1])))
        except ValueError:
            return [n for n in via.layers if n in names]
        return names[i:j + 1]

    def items_on(self, layer: str) -> Items:
        """Tracks, vias, pads and zones that have copper on this layer."""
        self.stackup.layer(layer)
        return Items([t for t in self.tracks if t.layer == layer],
                     [v for v in self.vias if layer in self.via_layers(v)],
                     [p for p in self.pads if layer in p.layers],
                     [z for z in self.zones if z.layer == layer])

    def reindex(self) -> None:
        """Rebuild the per net index after changing the item lists."""
        by = {}
        for kind, seq in enumerate((self.tracks, self.vias, self.pads, self.zones)):
            for it in seq:
                by.setdefault(it.net, ([], [], [], []))[kind].append(it)
        self._by_net = by

    def net_items(self, net: str) -> Items:
        """Everything that belongs to one net."""
        if self._by_net is None:
            self.reindex()
        got = self._by_net.get(net)
        return Items(*(list(g) for g in got)) if got else Items([], [], [], [])

    def summary(self) -> dict:
        """Counts and the stackup source, handy for reports."""
        return {"name": self.name, "source": self.source, "copper_layers": len(self.stackup.layers),
                "nets": len(self.nets), "tracks": len(self.tracks), "vias": len(self.vias),
                "pads": len(self.pads), "zones": len(self.zones), "outline_rings": len(self.outline),
                "board_thickness_mm": round(self.stackup.board_thickness / MM, 4), "stackup": self.stackup.source}
