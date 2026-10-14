"""Grain shapes and how they burn back.

Every exposed surface of a grain recedes at the same rate along its normal, so after a regression x the burning
surface is everything a distance x into the propellant from where it started (Huygens' principle). A grain reports,
at any x, its burning area, the open port at its aft end and the propellant left. Round cores (BATES segments),
tubes that burn inside and out and end burners have closed forms. Any other core, a star, a finocyl with fins, an
offset moon core or an X, is drawn on a fine grid over the grain's face. The distance from each propellant cell to
the nearest port cell is exactly how far the flame travels to reach it, so the port at regression x is every cell
within x and the burning perimeter is the length of that level line, traced with contourpy.

Lengths are in meters. ``ends`` is how many end faces burn: 2 for an uninhibited segment, 0 when both are coated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grain:
    shape: str
    D: float                 # outer diameter
    L: float                 # length
    ends: int = 2            # end faces that burn

    def face(self, x: float) -> float:
        """Propellant area left on the cross section."""
        raise NotImplementedError

    def perimeter(self, x: float) -> float:
        """Burning perimeter of the cross section (core, and outside for a tube)."""
        raise NotImplementedError

    def port(self, x: float) -> float:
        """Open flow area through the grain."""
        return math.pi / 4 * self.D ** 2 - self.face(x)

    @property
    def web_radial(self) -> float:
        raise NotImplementedError

    @property
    def web(self) -> float:
        """Regression at which the grain is gone."""
        return min(self.web_radial, self.L / self.ends if self.ends else math.inf)

    def length(self, x: float) -> float:
        return max(self.L - self.ends * x, 0.0)

    def burn_area(self, x: float) -> float:
        if x >= self.web:
            return 0.0
        return self.perimeter(x) * self.length(x) + self.ends * self.face(x)

    def volume(self, x: float) -> float:
        if x >= self.web:
            return 0.0
        return self.face(x) * self.length(x)

    def hydraulic(self, x: float) -> float:
        """Hydraulic diameter of the port, 4 A / perimeter."""
        p = self.perimeter(x)
        return 4 * self.port(x) / p if p > 0 else 0.0

    def describe(self) -> str:
        return f"{self.shape} {self.D * 1000:.4g} mm x {self.L * 1000:.4g} mm"


@dataclass
class Bates(Grain):
    """A cylinder with a round core, the outside against the casing."""
    core: float = 0.0

    def face(self, x):
        d = min(self.core + 2 * x, self.D)
        return math.pi / 4 * (self.D ** 2 - d ** 2)

    def perimeter(self, x):
        return math.pi * (self.core + 2 * x) if x < self.web_radial else 0.0

    @property
    def web_radial(self):
        return (self.D - self.core) / 2

    def describe(self):
        return f"BATES {self.D * 1000:.4g} mm with a {self.core * 1000:.4g} mm core, {self.L * 1000:.4g} mm long"


@dataclass
class Tube(Grain):
    """A tube burning on its core and its outside, free of the casing. With coated ends its area never changes."""
    core: float = 0.0

    def face(self, x):
        do, di = self.D - 2 * x, self.core + 2 * x
        return math.pi / 4 * max(do ** 2 - di ** 2, 0.0)

    def perimeter(self, x):
        return math.pi * (self.core + self.D) if x < self.web_radial else 0.0

    def port(self, x):
        return math.pi / 4 * (self.core + 2 * x) ** 2

    @property
    def web_radial(self):
        return (self.D - self.core) / 4

    def describe(self):
        return (f"tube {self.D * 1000:.4g} mm with a {self.core * 1000:.4g} mm core, burning inside and out, "
                f"{self.L * 1000:.4g} mm long")


@dataclass
class EndBurner(Grain):
    """A solid cylinder that burns only on its aft face, like a cigarette."""
    ends: int = 1

    def face(self, x):
        return math.pi / 4 * self.D ** 2

    def perimeter(self, x):
        return 0.0

    def port(self, x):
        return 0.0

    @property
    def web_radial(self):
        return math.inf

    def describe(self):
        return f"end burner {self.D * 1000:.4g} mm x {self.L * 1000:.4g} mm"


@dataclass
class MapGrain(Grain):
    """Any core shape, burned back on a grid over the face. ``params`` holds the shape's own sizes in meters."""
    params: dict = field(default_factory=dict)
    cells: int = 500

    def __post_init__(self):
        n = self.cells
        h = self.D / n
        c = (np.arange(n) + 0.5) * h - self.D / 2
        X, Y = np.meshgrid(c, c)
        inside = X ** 2 + Y ** 2 <= (self.D / 2) ** 2
        core = CORES[self.shape](X, Y, self.D, **self.params) & inside
        if not core.any():
            raise ValueError(f"the {self.shape} core is too small to draw on a {n} cell grid")
        from scipy.ndimage import distance_transform_edt

        # distance from each cell center to the nearest port cell center, less half a cell for the port's edge
        dist = distance_transform_edt(~core) * h - h / 2
        dist[core] = -h / 2
        import contourpy

        vals = np.sort(dist[inside & ~core])
        self._web = float(vals[-1]) + h / 2
        self._xs = np.linspace(0.0, self._web, 121)
        # each cell stands for an equal share of the true circle, so the face is gone exactly at burnout
        cell = math.pi / 4 * self.D ** 2 / int(inside.sum())
        self._area = (int(core.sum()) + np.searchsorted(vals, self._xs, side="right")) * cell
        gen = contourpy.contour_generator(c, c, np.ma.masked_invalid(np.where(inside, dist, np.nan)),
                                          line_type="Separate")
        self._perim = np.array([_length(gen.lines(x)) for x in self._xs])
        # right at the port's edge the level line follows the cells' staircase, so the starting perimeter comes
        # from the smooth lines a few cells in, extended back to zero
        x1, x2 = 5 * h, 10 * h
        p1, p2 = _length(gen.lines(x1)), _length(gen.lines(x2))
        self._perim[0] = p1 - (p2 - p1) / (x2 - x1) * x1
        self._perim[-1] = 0.0
        self._gen = gen

    def port_outline(self, x: float) -> list[np.ndarray]:
        """The burning edge at regression x as polylines of (x, y) in m, the level line the perimeter measures."""
        return [ln for ln in self._gen.lines(max(x, 0.0)) if len(ln) > 1]

    def face(self, x):
        a = float(np.interp(x, self._xs, self._area))
        return max(math.pi / 4 * self.D ** 2 - a, 0.0)

    def perimeter(self, x):
        return float(np.interp(x, self._xs, self._perim)) if x < self._web else 0.0

    @property
    def web_radial(self):
        return self._web

    def describe(self):
        sizes = ", ".join(f"{k} {v * 1000:.4g} mm" if isinstance(v, float) else f"{k} {v}"
                          for k, v in self.params.items())
        return f"{self.shape} {self.D * 1000:.4g} mm x {self.L * 1000:.4g} mm ({sizes})"

    def coarea_error(self) -> float:
        """The burning perimeter integrated over the regression has to equal the face area it eats. Their
        difference measures how well the grid resolves the shape."""
        p, x = self._perim, self._xs
        burned = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(x)))
        total = float(self._area[-1] - self._area[0])
        return abs(burned - total) / total


def _length(lines) -> float:
    return float(sum(np.sum(np.hypot(*np.diff(ln, axis=0).T)) for ln in lines if len(ln) > 1))


# --- core shapes on the grid: each returns True on port cells -------------------------------------------------
def _star(X, Y, D, points: int = 5, point_length: float = 0.0, point_width: float = 0.0):
    """Triangular points from the center out to point_length, each point_width wide at its base."""
    th = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    out = np.zeros_like(X, bool)
    for k in range(int(points)):
        a = th - 2 * math.pi * k / points
        u, v = r * np.cos(a), r * np.sin(a)          # along and across the point
        half = point_width / 2 * (1 - u / point_length)
        out |= (u >= 0) & (u <= point_length) & (np.abs(v) <= half)
    return out


def _finocyl(X, Y, D, core: float = 0.0, fins: int = 6, fin_length: float = 0.0, fin_width: float = 0.0):
    """A round core with straight fins of fin_width reaching fin_length out from the core's edge."""
    out = X ** 2 + Y ** 2 <= (core / 2) ** 2
    th = np.arctan2(Y, X)
    r = np.hypot(X, Y)
    for k in range(int(fins)):
        a = th - 2 * math.pi * k / fins
        u, v = r * np.cos(a), r * np.sin(a)
        out |= (u >= 0) & (u <= core / 2 + fin_length) & (np.abs(v) <= fin_width / 2)
    return out


def _moon(X, Y, D, core: float = 0.0, offset: float = 0.0):
    """A round core moved off center by offset."""
    return (X - offset) ** 2 + Y ** 2 <= (core / 2) ** 2


def _xcore(X, Y, D, slot_width: float = 0.0, slot_length: float = 0.0):
    """Two slots crossing at the center, each slot_length end to end."""
    a = (np.abs(X) <= slot_length / 2) & (np.abs(Y) <= slot_width / 2)
    b = (np.abs(Y) <= slot_length / 2) & (np.abs(X) <= slot_width / 2)
    return a | b


def _round(X, Y, D, core: float = 0.0):
    return X ** 2 + Y ** 2 <= (core / 2) ** 2


CORES = {"star": _star, "finocyl": _finocyl, "moon": _moon, "xcore": _xcore, "round": _round}
SHAPES = ("bates", "tube", "endburner", *CORES)


def make_grain(shape: str, D: float, L: float, ends: int = 2, **params) -> Grain:
    s = shape.strip().lower().replace("-", "").replace("_", "")
    s = {"endburn": "endburner", "end": "endburner", "cigarette": "endburner", "core": "bates",
         "cylinder": "bates", "fins": "finocyl", "x": "xcore"}.get(s, s)
    if s not in SHAPES:
        raise ValueError(f"unknown grain shape '{shape}' (use {', '.join(SHAPES)})")
    if D <= 0 or L <= 0:
        raise ValueError("a grain needs a positive diameter and length")
    if s == "bates":
        g = Bates(s, D, L, ends, core=params.get("core", 0.0))
        if not 0 < g.core < D:
            raise ValueError("a BATES grain needs a core between zero and its diameter")
        return g
    if s == "tube":
        g = Tube(s, D, L, ends, core=params.get("core", 0.0))
        if not 0 < g.core < D:
            raise ValueError("a tube needs a core between zero and its diameter")
        return g
    if s == "endburner":
        return EndBurner(s, D, L, 1)
    return MapGrain(s, D, L, ends, params=params)
