"""Connectivity of a board. The graph of each net, the planes under a point, and nets for unnamed copper.

A net graph has pads, vias (one node per copper layer the via joins) and
junctions where tracks meet or branch. Chains of tracks of one width on one
layer between those nodes become single edges, which is what an RLC ladder
for the net needs. Zones join everything they touch through zero length
edges to one node per filled polygon.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import geom
from .model import Board, Items, Pad, Zone

_EPS = 1e-7   # 0.1 um, absorbs rounding of coordinates that sit exactly on a pad edge


@dataclass
class Edge:
    """A piece of conductor between two nodes.

    kind is "track" (a chain of straight tracks), "via" (a barrel between two copper layers,
    length is the distance between their mid planes), "zone" (zero length tie to a pour) or
    "tie" (zero length, a via inside a pad or two pads that overlap).
    """

    a: str
    b: str
    layer: str
    length: float
    width: float
    polyline: np.ndarray
    kind: str = "track"
    layers: tuple[str, ...] = ()
    drill: float = 0.0
    diameter: float = 0.0


@dataclass
class NetGraph:
    """Nodes and edges of one net. Node values are (x, y, layers)."""

    net: str
    nodes: dict[str, tuple[float, float, tuple[str, ...]]]
    edges: list[Edge]
    pads: list[str]
    unconnected: list[str]
    components: list[list[str]] = field(default_factory=list)

    @property
    def connected(self) -> bool:
        """True when every pad of the net sits in one connected piece of copper."""
        return not self.unconnected

    def length(self, kind: str = "track") -> float:
        """Total length of the edges of one kind."""
        return sum(e.length for e in self.edges if e.kind == kind)

    def edges_at(self, node: str) -> list[Edge]:
        return [e for e in self.edges if e.a == node or e.b == node]


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _tree(points):
    from scipy.spatial import cKDTree
    return cKDTree(points)


class _Builder:
    """Builds nodes and edges for a set of items, all of one net or (for net inference) all of a board."""

    def __init__(self, board: Board, items: Items, tol: float | None):
        self.board = board
        self.items = items
        self.tol = tol
        self.copper = board.copper_layers()
        self.nodes: dict[str, tuple[float, float, tuple[str, ...]]] = {}
        self.edges: list[Edge] = []
        self.pad_nodes: list[str] = []
        self.via_nodes: dict[tuple[int, str], str] = {}
        self.via_spans = [board.via_layers(v) for v in items.vias]
        self.track_nodes: list[list[str]] = []
        self.zone_nodes: dict[tuple[int, int], str] = {}
        self.junctions: set[str] = set()
        self._njunction = 0
        self._names: dict[str, int] = {}

    def unique(self, base: str) -> str:
        k = self._names.get(base, 0)
        self._names[base] = k + 1
        return base if k == 0 else f"{base}#{k + 1}"

    def junction(self, x: float, y: float, layer: str) -> str:
        self._njunction += 1
        name = f"J{self._njunction}"
        self.nodes[name] = (float(x), float(y), (layer,))
        self.junctions.add(name)
        return name

    def via_node(self, vi: int, layer: str) -> str:
        key = (vi, layer)
        if key not in self.via_nodes:
            v = self.items.vias[vi]
            name = f"V{vi + 1}:{layer}"
            self.via_nodes[key] = name
            self.nodes[name] = (v.at[0], v.at[1], (layer,))
        return self.via_nodes[key]

    def build(self) -> None:
        items = self.items
        for i, p in enumerate(items.pads):
            name = self.unique(f"{p.ref}.{p.number}" if (p.ref or p.number) else f"PAD{i + 1}")
            self.nodes[name] = (p.at[0], p.at[1], tuple(p.layers))
            self.pad_nodes.append(name)
        self.track_nodes = [["", ""] for _ in items.tracks]
        splits: dict[int, list] = defaultdict(list)
        by_layer: dict[str, list[int]] = defaultdict(list)
        for k, t in enumerate(items.tracks):
            by_layer[t.layer].append(k)
        for layer, idx in by_layer.items():
            self._attach_layer(layer, np.array(idx), splits)
        self._track_edges(splits)
        self._ties()
        self._zones()
        self._via_edges()

    # track ends
    def _attach_layer(self, layer: str, idx: np.ndarray, splits: dict) -> None:
        tr = self.items.tracks
        n = len(idx)
        S = np.array([tr[k].start for k in idx])
        E = np.array([tr[k].end for k in idx])
        W = np.array([tr[k].width for k in idx])
        P = np.vstack([S, E])
        w = np.concatenate([W, W])
        owner = np.concatenate([idx, idx])
        which = np.concatenate([np.zeros(n, int), np.ones(n, int)])
        node = np.array([""] * (2 * n), dtype=object)
        tree = _tree(P)
        tol = self.tol
        wmax = float(w.max())
        # pads and vias, first ends that lie inside them, then ends whose round cap overlaps them
        lpads = [(k, p) for k, p in enumerate(self.items.pads) if layer in p.layers]
        lvias = [(vi, v) for vi, v in enumerate(self.items.vias) if layer in self.via_spans[vi]]
        base = (tol or 0.0) + _EPS
        for strict in ((True,) if tol is not None else (True, False)):
            reach = base + (0.0 if strict else wmax / 2.0)
            for k, p in lpads:
                cand = tree.query_ball_point(p.at, 0.5 * math.hypot(*p.size) + reach)
                cand = np.array(cand, dtype=int)
                cand = cand[node[cand] == ""]
                if len(cand):
                    inside = _contains(p, P[cand], base if strict else base + w[cand] / 2.0)
                    node[cand[inside]] = self.pad_nodes[k]
            for vi, v in lvias:
                cand = tree.query_ball_point(v.at, v.diameter / 2.0 + reach)
                cand = np.array(cand, dtype=int)
                cand = cand[node[cand] == ""]
                if len(cand):
                    d = np.hypot(*(P[cand] - np.asarray(v.at)).T)
                    hit = cand[d <= v.diameter / 2.0 + (base if strict else base + w[cand] / 2.0)]
                    if len(hit):
                        node[hit] = self.via_node(vi, layer)

        def near(a, b) -> bool:
            return math.hypot(*(P[a] - P[b])) <= (tol if tol is not None else min(w[a], w[b]) / 2.0) + _EPS

        # free ends that meet group together, and a group touching an attached end joins its node
        free = np.nonzero(node == "")[0]
        uf = _UF()
        if len(free) > 1:
            sub = _tree(P[free])
            for i, j in sub.query_pairs((tol if tol is not None else wmax / 2.0) + _EPS):
                if near(free[i], free[j]):
                    uf.union(int(free[i]), int(free[j]))
        groups: dict[int, list[int]] = defaultdict(list)
        for a in free:
            groups[uf.find(int(a))].append(int(a))
        attached = np.nonzero(node != "")[0]
        atree = _tree(P[attached]) if len(attached) else None
        lone = []
        for g in groups.values():
            target = ""
            if atree is not None:
                for a in g:
                    for j in atree.query_ball_point(P[a], (tol if tol is not None else w[a] / 2.0) + _EPS):
                        if near(a, attached[j]):
                            target = node[attached[j]]
                            break
                    if target:
                        break
            if not target:
                pt = P[g].mean(axis=0)
                target = self.junction(pt[0], pt[1], layer)
                if len(g) == 1:
                    lone.append(g[0])
            for a in g:
                node[a] = target
        # a track that runs across a pad or via without ending there is split where it passes
        for k, p in lpads:
            self._pass_through(np.asarray(p.at, dtype=float), S, E, W, idx, node, n, splits,
                               lambda q, wq, p=p: p.contains(q, (tol or 0.0) + wq / 2.0 + _EPS),
                               self.pad_nodes[k])
        for vi, v in lvias:
            name = self.via_node(vi, layer)
            self._pass_through(np.asarray(v.at, dtype=float), S, E, W, idx, node, n, splits,
                               lambda q, wq, v=v: np.hypot(*(q - np.asarray(v.at)).T) <= v.diameter / 2.0 + wq / 2.0
                               + (tol or 0.0) + _EPS, name)
        # a lone end that lands on the body of another track makes a T junction there
        if lone:
            L = np.array(lone)
            lim = (W / 2.0 if tol is None else np.full(n, tol)) + _EPS
            dist, t = geom.segment_distance(P[L], S, E)
            for row, a in enumerate(L):
                d = dist[row]
                d[idx == owner[a]] = np.inf
                ok = np.nonzero(d <= lim)[0]
                if not len(ok):
                    continue
                col = int(ok[np.argmin(d[ok])])
                tt = float(t[row, col])
                old = node[a]
                if tt <= 1e-6 or tt >= 1 - 1e-6:
                    node[a] = node[col if tt <= 1e-6 else col + n]
                    self.nodes.pop(old, None)
                    self.junctions.discard(old)
                else:
                    splits[int(idx[col])].append((tt, old))
                    q = S[col] + tt * (E[col] - S[col])
                    self.nodes[old] = (float(q[0]), float(q[1]), (layer,))
        for r in range(2 * n):
            self.track_nodes[owner[r]][which[r]] = node[r]

    @staticmethod
    def _pass_through(c, S, E, W, idx, node, n, splits, inside, name) -> None:
        d = E - S
        dd = (d * d).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dd > 0, ((c - S) * d).sum(1) / dd, 0.0)
        mid = np.nonzero((t > 1e-6) & (t < 1 - 1e-6))[0]
        if not len(mid):
            return
        q = S[mid] + t[mid, None] * d[mid]
        hit = mid[inside(q, W[mid])]
        for col in hit:
            if node[col] == name or node[col + n] == name:
                continue
            splits[int(idx[col])].append((float(t[col]), name))

    def _track_edges(self, splits: dict) -> None:
        for k, t in enumerate(self.items.tracks):
            a, b = self.track_nodes[k]
            stops = [(0.0, a)] + sorted(splits.get(k, [])) + [(1.0, b)]
            s, e = np.asarray(t.start), np.asarray(t.end)
            for (t0, n0), (t1, n1) in zip(stops[:-1], stops[1:]):
                if n0 == n1:
                    continue
                p0, p1 = s + t0 * (e - s), s + t1 * (e - s)
                self.edges.append(Edge(n0, n1, t.layer, float(np.hypot(*(p1 - p0))), t.width, np.array([p0, p1])))

    def _ties(self) -> None:
        """Zero length ties for a via inside a pad and for pads that overlap each other."""
        pads = self.items.pads
        if not pads:
            return
        centers = np.array([p.at for p in pads], dtype=float)
        half = np.array([0.5 * math.hypot(*p.size) for p in pads])
        tree = _tree(centers)
        hmax = float(half.max())
        for vi, v in enumerate(self.items.vias):
            for k in tree.query_ball_point(v.at, hmax + v.diameter / 2.0):
                p = pads[k]
                shared = [lay for lay in self.via_spans[vi] if lay in p.layers]
                if shared and p.contains([v.at], v.diameter / 2.0)[0]:
                    for lay in shared:
                        self._tie(self.pad_nodes[k], self.via_node(vi, lay), lay)
        for i, j in sorted(tree.query_pairs(2 * hmax)):
            a, b = pads[i], pads[j]
            if math.hypot(a.at[0] - b.at[0], a.at[1] - b.at[1]) > half[i] + half[j]:
                continue
            shared = [lay for lay in a.layers if lay in b.layers]
            if not shared:
                continue
            if (a.contains(np.vstack([[b.at], b.outline()]), _EPS).any()
                    or b.contains(np.vstack([[a.at], a.outline()]), _EPS).any()):
                self._tie(self.pad_nodes[i], self.pad_nodes[j], shared[0])

    def _tie(self, a: str, b: str, layer: str) -> None:
        xa, ya, _ = self.nodes[a]
        xb, yb, _ = self.nodes[b]
        self.edges.append(Edge(a, b, layer, 0.0, 0.0, np.array([[xa, ya], [xb, yb]]), "tie"))

    # zones
    def _zones(self) -> None:
        items = self.items
        count = 0
        for zi, z in enumerate(items.zones):
            layer = z.layer
            names, pts, radius, pads = [], [], [], []
            for k, p in enumerate(items.pads):
                if layer in p.layers:
                    names.append(self.pad_nodes[k])
                    pts.append(p.at)
                    radius.append(0.5 * math.hypot(*p.size))
                    pads.append(p)
            for vi, v in enumerate(items.vias):
                if layer in self.via_spans[vi]:
                    names.append((vi, layer))
                    pts.append(v.at)
                    radius.append(v.diameter / 2.0)
                    pads.append(None)
            for k, t in enumerate(items.tracks):
                if t.layer == layer:
                    for end, xy in ((0, t.start), (1, t.end)):
                        names.append(self.track_nodes[k][end])
                        pts.append(xy)
                        radius.append(t.width / 2.0)
                        pads.append(None)
            if not names:
                continue
            P = np.array(pts, dtype=float)
            R = np.array(radius)
            for pi, (outer, holes) in enumerate(z.polygons):
                x0, y0 = outer.min(axis=0)
                x1, y1 = outer.max(axis=0)
                grow = R + z.stroke / 2.0 + 1e-3
                near = np.nonzero((P[:, 0] >= x0 - grow) & (P[:, 0] <= x1 + grow) & (P[:, 1] >= y0 - grow)
                                  & (P[:, 1] <= y1 + grow))[0]
                if not len(near):
                    continue
                inside = np.zeros(len(near), dtype=bool)
                inside |= geom.points_in_polygon(P[near], outer, holes)
                rest = np.nonzero(~inside)[0]
                if len(rest):
                    verts = np.vstack([outer] + list(holes))
                    order = np.argsort(verts[:, 0])
                    vx = verts[order, 0]
                    for r in rest:
                        m = near[r]
                        p = pads[m]
                        reach = (0.5 * math.hypot(*p.size) if p is not None else R[m]) + z.stroke / 2.0 + _EPS
                        lo, hi = np.searchsorted(vx, [P[m, 0] - reach, P[m, 0] + reach])
                        if lo == hi:
                            continue
                        cand = verts[order[lo:hi]]
                        cand = cand[np.abs(cand[:, 1] - P[m, 1]) <= reach]
                        if not len(cand):
                            continue
                        if p is not None:
                            hit = _contains(p, cand, z.stroke / 2.0 + _EPS).any()
                            if not hit:
                                hit = geom.points_in_polygon(p.outline(), outer, holes).any()
                        else:
                            hit = (np.hypot(*(cand - P[m]).T) <= R[m] + z.stroke / 2.0 + _EPS).any()
                        inside[r] = hit
                members = []
                for r in np.nonzero(inside)[0]:
                    nm = names[near[r]]
                    if isinstance(nm, tuple):
                        nm = self.via_node(*nm)
                    if nm and nm not in members:
                        members.append(nm)
                if not members:
                    continue
                count += 1
                cx, cy = geom.ring_centroid(outer)
                zname = f"Z{count}"
                self.nodes[zname] = (cx, cy, (layer,))
                self.zone_nodes[(zi, pi)] = zname
                for nm in members:
                    x, y, _ = self.nodes[nm]
                    self.edges.append(Edge(zname, nm, layer, 0.0, 0.0, np.array([[cx, cy], [x, y]]), "zone"))

    def _via_edges(self) -> None:
        st = self.board.stackup
        for vi, v in enumerate(self.items.vias):
            span = self.via_spans[vi]
            if not span:
                continue
            used = {span[0], span[-1]} | {lay for (k, lay) in self.via_nodes if k == vi}
            lays = [lay for lay in span if lay in used]
            names = [self.via_node(vi, lay) for lay in lays]
            for (la, na), (lb, nb) in zip(zip(lays[:-1], names[:-1]), zip(lays[1:], names[1:])):
                self.edges.append(Edge(na, nb, la, st.distance(la, lb), v.diameter,
                                       np.array([v.at, v.at], dtype=float), "via", (la, lb), v.drill, v.diameter))

    def merge_chains(self) -> None:
        """Join two tracks that meet at a junction with nothing else attached, if layer and width agree."""
        edges = self.edges
        alive = [True] * len(edges)
        adj: dict[str, set[int]] = defaultdict(set)
        for i, e in enumerate(edges):
            adj[e.a].add(i)
            adj[e.b].add(i)
        for n in sorted(self.junctions, key=lambda s: int(s[1:])):
            ids = [i for i in adj.get(n, ()) if alive[i]]
            if len(ids) != 2:
                continue
            e1, e2 = edges[ids[0]], edges[ids[1]]
            if e1.kind != "track" or e2.kind != "track" or e1.layer != e2.layer:
                continue
            if abs(e1.width - e2.width) > 0.01 * max(e1.width, e2.width):
                continue
            a = e1.a if e1.b == n else e1.b
            b = e2.b if e2.a == n else e2.a
            if a == n or b == n:
                continue
            p1 = e1.polyline if e1.b == n else e1.polyline[::-1]
            p2 = e2.polyline if e2.a == n else e2.polyline[::-1]
            new = Edge(a, b, e1.layer, e1.length + e2.length, e1.width, np.vstack([p1, p2[1:]]))
            for i in ids:
                alive[i] = False
                adj[edges[i].a].discard(i)
                adj[edges[i].b].discard(i)
            edges.append(new)
            alive.append(True)
            adj[a].add(len(edges) - 1)
            adj[b].add(len(edges) - 1)
            adj.pop(n, None)
            self.nodes.pop(n, None)
        self.edges = [e for e, ok in zip(edges, alive) if ok]

    def components(self) -> list[list[str]]:
        uf = _UF()
        for n in self.nodes:
            uf.find(n)
        for e in self.edges:
            uf.union(e.a, e.b)
        groups: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes:
            groups[uf.find(n)].append(n)
        pads = set(self.pad_nodes)
        return sorted(groups.values(), key=lambda g: (-sum(1 for n in g if n in pads), -len(g)))


def _contains(pad: Pad, pts, margin) -> np.ndarray:
    if np.ndim(margin) == 0:
        return pad.contains(pts, float(margin))
    margin = np.asarray(margin, dtype=float)
    out = np.zeros(len(pts), dtype=bool)
    for m in np.unique(margin):
        sel = margin == m
        out[sel] = pad.contains(np.asarray(pts)[sel], float(m))
    return out


def net_graph(board: Board, net: str, tol: float | None = None) -> NetGraph:
    """Connect a net's pads, tracks, vias and zones into a graph.

    A track end touches a pad when it lies inside the pad's shape (or its round end overlaps it),
    a via when it lies on the via's annular ring, and another track end when the two are closer
    than half the smaller width. tol, in meters, replaces those distances when given.
    """
    if net not in board.nets and not any(board.net_items(net)):
        raise ValueError(f"The board has no net named {net!r}. Pick one from board.nets.")
    b = _Builder(board, board.net_items(net), tol)
    b.build()
    b.merge_chains()
    comps = b.components()
    main = set(comps[0]) if comps else set()
    pads = list(b.pad_nodes)
    unconnected = [p for p in pads if p not in main]
    return NetGraph(net, b.nodes, b.edges, pads, unconnected, comps)


def reference_planes(board: Board, layer: str, xy) -> list[tuple[str, str, float]]:
    """Zones on other copper layers that cover the point xy, nearest first.

    Each entry is (layer, net, distance) where distance is the vertical gap between the facing
    surfaces of the two copper layers, any copper between them included.
    """
    st = board.stackup
    st.layer(layer)
    pt = np.array([xy], dtype=float)
    out: dict[tuple[str, str], float] = {}
    for z in board.zones:
        if z.layer == layer or not z.polygons:
            continue
        key = (z.layer, z.net)
        if key in out:
            continue
        for outer, holes in z.polygons:
            if (outer[:, 0].min() <= pt[0, 0] <= outer[:, 0].max() and outer[:, 1].min() <= pt[0, 1]
                    <= outer[:, 1].max() and geom.points_in_polygon(pt, outer, holes)[0]):
                out[key] = st.gap_between(layer, z.layer)
                break
    return sorted(((lay, net, d) for (lay, net), d in out.items()),
                  key=lambda r: (r[2], st.layer(r[0]).index))


def infer_nets(board: Board) -> int:
    """Name the nets of a board whose files carry no net names by tracing which copper touches.

    Each connected piece of copper with a pad gets a net "N1", "N2" ... (largest pad count first).
    Copper that touches no pad keeps an empty net. Returns the number of nets and notes it on the board.
    """
    items = Items(board.tracks, board.vias, board.pads, board.zones)
    b = _Builder(board, items, None)
    b.build()
    comps = b.components()
    label: dict[str, str] = {}
    pads = set(b.pad_nodes)
    k = floating = 0
    for g in comps:
        if not any(n in pads for n in g):
            floating += len(g) > 1
            continue
        k += 1
        for n in g:
            label[n] = f"N{k}"
    for i, p in enumerate(board.pads):
        p.net = label.get(b.pad_nodes[i], "")
    for i, t in enumerate(board.tracks):
        t.net = label.get(b.track_nodes[i][0], "")
    for vi, v in enumerate(board.vias):
        span = b.via_spans[vi]
        v.net = label.get(b.via_nodes.get((vi, span[0]), ""), "") if span else ""
    zones = []
    for zi, z in enumerate(board.zones):
        groups: dict[str, list] = defaultdict(list)
        for pi, poly in enumerate(z.polygons):
            groups[label.get(b.zone_nodes.get((zi, pi), ""), "")].append(poly)
        zones += [Zone(net, z.layer, polys, z.stroke) for net, polys in groups.items()]
    board.zones = zones
    board.nets = sorted({label[n] for n in label}, key=lambda s: int(s[1:]))
    board.reindex()
    board.notes.append(f"The Gerbers carry no net names, so {len(board.nets)} nets were traced from the copper "
                       "and named N1, N2 and so on.")
    if floating:
        board.notes.append(f"{floating} pieces of copper touch no pad and were left without a net.")
    return len(board.nets)
