"""The 3D scene of a board run: the board as it is built and its circuit ringing on it.

The board is an FR4 slab of the stackup's thickness. Its copper (tracks, pours and pads) stands on the slab at
each layer's height, colored by its net's voltage as ngspice solved the circuit with the layout, frame by frame
over the ringing window. Glows run along every track at a speed set by the current the solve put through that
pad connection, so the hot loop is the copper that flickers and races. The parts stand on their pads as 3D
packages from the bench props, chosen by their reference letter.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

COPPER_MIN = 0.06          # mm, copper is drawn at least this thick so it reads as raised
FR4 = "#2f6b3a"


def _prism(ring, z0: float, z1: float, holes=()) -> tuple[np.ndarray, np.ndarray]:
    """A polygon (mm, with holes) extruded from z0 to z1: its top, its bottom and its walls."""
    from matplotlib.tri import Triangulation

    from .pcb import geom

    rings = [np.asarray(ring, float)] + [np.asarray(h, float) for h in holes]
    rings = [r[:-1] if len(r) > 3 and np.allclose(r[0], r[-1]) else r for r in rings]
    P = np.vstack(rings)
    T = Triangulation(P[:, 0], P[:, 1]).triangles
    c = P[T].mean(axis=1)
    T = T[geom.points_in_polygon(c, rings[0], rings[1:])]
    n = len(P)
    X = np.vstack([np.c_[P, np.full(n, z1)], np.c_[P, np.full(n, z0)]])
    tris = [T, T[:, ::-1] + n]
    k = 0
    for r in rings:
        m = len(r)
        i = np.arange(m) + k
        j = (np.arange(m) + 1) % m + k
        tris += [np.c_[i, j, j + n], np.c_[i, j + n, i + n]]
        k += m
    return X, np.vstack(tris)


def _bar(a, b, w: float, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """A straight track of width w from a to b (mm) between z0 and z1, with square ends half a width out."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.hypot(*d)) or 1e-6
    u = d / L
    v = np.array([-u[1], u[0]]) * w / 2
    a, b = a - u * w / 2, b + u * w / 2
    return _prism(np.array([a - v, b - v, b + v, a + v]), z0, z1)


class _Copper:
    """All copper as one mesh, each vertex remembering its net so it can be colored by that net's voltage."""

    def __init__(self):
        self.X, self.T, self.net = [], [], []
        self.n = 0

    def add(self, XT, net: str):
        X, T = XT
        self.X.append(X)
        self.T.append(T + self.n)
        self.net += [net] * len(X)
        self.n += len(X)


def _node(net: str) -> str:
    from ..circuits.netlist import safe_node
    return safe_node(net.lstrip("/"))


def _parts(sc, b, values: dict, ztop: float):
    """The parts as 3D packages on their pads, by reference letter."""
    from ..viz import props_bench as PB

    refs: dict[str, list] = {}
    for p in b.pads:
        if p.ref and (not p.layers or any(lay.startswith("F.") or lay == "*.Cu" for lay in p.layers)):
            refs.setdefault(p.ref, []).append(p)
    for ref, pads in refs.items():
        pts = np.array([[p.at[0] * 1e3, p.at[1] * 1e3, ztop] for p in pads])
        mt = re.match(r"[A-Za-z]+", ref)
        letter = mt.group(0).upper() if mt else ""
        val = values.get(ref.upper())
        two = len(pts) == 2
        if letter == "C" and two:
            PB.capacitor(sc, pts, val, ref, lift=0.0)
        elif letter == "R" and two:
            PB.resistor(sc, pts[0], pts[1], val, ref, lift=1.6)
        elif letter == "D" and two:
            PB.diode(sc, pts[0], pts[1], ref, lift=1.4)
        elif letter == "L" and two:
            PB.inductor(sc, pts, ref)
        elif letter == "LED" and two:
            PB.led(sc, pts, "#e8453c", ref)
        else:
            # an SMD package over its pads, a connector a taller block with its terminal screws
            lo, hi = pts.min(0), pts.max(0)
            half = np.maximum((hi - lo)[:2] / 2 + 0.6, 1.2)
            tall = 4.0 if letter in ("J", "P", "CN", "X") else 1.2
            color = "#2f6fd6" if tall > 2 else "#1c1d20"
            mid = (lo + hi) / 2
            sc.body("box", [half[0], half[1], tall / 2], pos=mid + [0, 0, 0.3 + tall / 2], color=color, name=ref)
            if tall > 2:
                for q in pts:
                    sc.body("cylinder", [0.7, 0.3], pos=q + [0, 0, 0.3 + tall + 0.3], color="#c3c7cd",
                            name=f"{ref} screw")
        sc.label(ref, pts.mean(0) + [0, 0, 7.0 if letter in ("C", "L") else 4.0], kind="note")


def write(out, o, ex, circ, title: str, question: str):
    from ..viz import Scene

    from . import view as V

    b = ex.board
    st = b.stackup
    th = st.board_thickness * 1e3
    sc = Scene(f"{Path(title).stem} board", domain="board", question=question, length="mm",
               source="The board is drawn from its file at its stackup's thickness. The copper is colored by its "
                      "net's voltage and the glows move with the current, both as ngspice solved the circuit with "
                      "the layout's parasitics.")
    wave = (circ or {}).get("wave") or {}
    m = np.array([0])
    if wave:
        t, yb, yi = wave["time"], wave["board"], wave["ideal"]
        i = int(np.argmax(yb))
        span = max(40 * circ["step"], 0.02 * (t[-1] - t[0]))
        m = np.where((t >= t[i] - 0.3 * span) & (t <= t[i] + span))[0]
        m = m[np.linspace(0, len(m) - 1, min(len(m), 160)).astype(int)]
        sc.timeline((t[m] - t[i]) * 1e9, unit="ns", label="time", seconds=10.0, between="interpolated",
                    note="the board's circuit as ngspice solved it, around the worst overshoot")
        part = circ["worst"]["part"]
        sc.readout(f"{part} with the layout", yb[m], unit="V")
        sc.readout(f"{part} on the schematic alone", yi[m], unit="V")
    nodes = {k: np.asarray(v)[m] for k, v in (wave.get("nodes") or {}).items()}
    # the FR4 slab
    for ring in b.outline[:1]:
        pts = np.asarray(ring) * 1e3
        if len(pts) >= 3:
            X, T = _prism(pts, 0.0, th)
            sc.mesh(X, T, name="FR4 board", color=FR4, opacity=0.9, smooth=False)
    # the copper, raised on its layers
    cu = _Copper()
    names = b.copper_layers()

    def zs(layer):
        lay = st.layer(layer)
        t = max(lay.thickness * 1e3, COPPER_MIN)
        return (th, th + t) if lay.index == 0 else (-t, 0.0) if lay.index == len(names) - 1 else \
            (lay.z_bottom * 1e3, lay.z_bottom * 1e3 + t)
    for z in b.zones[:60]:
        z0, z1 = zs(z.layer)
        for outer, holes in z.polygons[:8]:
            if len(outer) >= 3:
                cu.add(_prism(np.asarray(outer) * 1e3, z0 - 0.002, z1 - 0.004, [np.asarray(h) * 1e3 for h in holes]),
                       z.net)
    pieces = sorted(ex.pieces, key=lambda p: -p.length)[:1500]
    for p in pieces:
        z0, z1 = zs(p.layer)
        cu.add(_bar(np.array(p.a) * 1e3, np.array(p.b) * 1e3, p.width * 1e3, z0, z1), p.net)
    for p in b.pads[:2000]:
        lay = next((x for x in p.layers if x in names), names[0] if p.layers and "*.Cu" in p.layers else None)
        if lay:
            z0, z1 = zs(lay)
            cu.add(_prism(p.outline() * 1e3, z0, z1 + 0.01), p.net)
    for v in b.vias[:2000]:
        ring = np.c_[np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))] * v.diameter * 500 + np.array(v.at) * 1e3
        cu.add(_prism(ring, -COPPER_MIN, th + COPPER_MIN), v.net)
    if cu.n:
        X, T = np.vstack(cu.X), np.vstack(cu.T)
        mesh = sc.mesh(X, T, name="copper", color="#c98a4a", smooth=False, max_vertices=max(12000, cu.n + 1))
        if nodes:
            per = np.array([nodes.get(_node(n), np.zeros(len(m))) for n in cu.net]).T
            hi = max(float(np.max(np.abs(per))), 1e-3)
            mesh.color_by("net voltage", per, unit="V", colormap="viridis", range=[0.0, hi], digits=3)
    _parts(sc, b, wave.get("values") or {}, zs(names[0])[1])
    # current as glows along the tracks, a little above the copper
    cur = wave.get("currents") or []
    if cur and pieces:
        pads = {p.name: np.array(p.at) * 1e3 for p in b.pads}
        by_ref = {}
        for p in b.pads:
            by_ref.setdefault((p.ref, p.net), []).append(np.array(p.at) * 1e3)
        conn = []
        for c in cur:
            ref = c["to"].split(".")[0]
            at = pads.get(c["to"])
            if at is None:
                hits = [v for (r, n), v in by_ref.items() if r == ref and _node(n) == _node(c["net"])]
                at = hits[0][0] if hits else None
            if at is not None:
                conn.append((_node(c["net"]), at, np.asarray(c["i"])[m]))
        top = max([float(np.max(np.abs(i))) for _, _, i in conn] + [1e-6])
        paths, speed = [], []
        for p in pieces:
            mine = [(at, i) for n, at, i in conn if n == _node(p.net)]
            if not mine:
                continue
            a, bb = np.array(p.a) * 1e3, np.array(p.b) * 1e3
            mid = (a + bb) / 2
            at, i = min(mine, key=lambda x: np.hypot(*(x[0] - mid)))
            if np.hypot(*(bb - at)) > np.hypot(*(a - at)):
                a, bb = bb, a                          # point toward the pad the current goes to
            z = zs(p.layer)[1] + 0.25 if zs(p.layer)[1] > 0 else zs(p.layer)[0] - 0.25
            paths.append(np.array([[*a, z], [*bb, z]]))
            speed.append(12.0 * i / top)
        if paths:
            sc.lines(paths, color="#ffe066", width=3.0, flow=np.array(speed).T, spacing=3.0, name="current")
            sc.readout("peak pad current", np.max(np.abs([i for _, _, i in conn]), axis=0), unit="A")
    x0, y0, x1, y1 = b.bbox()
    size = max(x1 - x0, y1 - y0) * 1e3
    target = ((x0 + x1) * 500, (y0 + y1) * 500, th)
    sc.camera(target=target, direction=(0.25, -0.9, 0.75), distance=1.5 * size,
              presets=[{"name": "top", "target": target, "dir": (0, -0.05, 1), "dist": 1.6 * size},
                       {"name": "underside", "target": target, "dir": (0.2, -0.5, -1), "dist": 1.5 * size}])
    word, tn = V.tone(o.verdict)
    sc.verdict(word, o.headline, tn)
    sc.note("Copper is drawn at least 0.06 mm thick so it reads as raised. Its color is its net's voltage with the "
            "layout, and the glows move with the current at each pad connection, faster for more current.")
    return sc.save(out)
