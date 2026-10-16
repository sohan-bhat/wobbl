"""Parasitics of a real board: every net's traces, the planes under them and their neighbors, from the files.

The board is read from KiCad or Gerber files (fields.pcb). Each track is cut into pieces that see one copper
configuration: its layer, its width, the nearest plane below and above (or none), and the stackup's dielectrics
between. Each distinct configuration is one quasi static field solve of its cross section (fields.xsec), which
gives impedance, inductance, capacitance and delay per length. Neighbors of other nets that run parallel close
by are solved as coupled pairs for near and far end crosstalk. Vias get the closed form inductance of a barrel.

The result is a table per net, a map of the board colored by impedance, and parasitics.json in the form the
circuit domain splices into a schematic (circuits.netlist.join_parasitics). When the board's schematic sits next
to it (a .kicad_sch with the same name) or a netlist is given, the circuit runs with and without the board, and
the report says what the layout changes.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from .forms import MU0
from .xsec import Section, solve_section

RHO_CU = 1.72e-8          # ohm m at 20 C
CAPS = {"fast": 6, "normal": 14, "fine": 40}
GROUND = re.compile(r"(?i)^/?(gnd|agnd|dgnd|pgnd|ground|vss|0|earth|gnda|gndd|gnd_\w*|\w*_gnd)$")


@dataclass
class Piece:
    net: str
    layer: str
    a: tuple
    b: tuple
    width: float
    length: float
    below: tuple | None          # (layer, net) of the plane under the piece
    above: tuple | None
    key: tuple = ()
    z0: float = 0.0
    l: float = 0.0               # H/m
    c: float = 0.0               # F/m
    delay: float = 0.0           # s/m
    solved: bool = True


@dataclass
class Extracted:
    board: object
    pieces: list[Piece]
    sections: dict               # key: solved constants
    vias: list[dict]
    nets: dict                   # net: totals
    pairs: list[dict]            # coupled net pairs, worst first
    graphs: dict
    notes: list[str] = field(default_factory=list)
    verify: dict = field(default_factory=dict)


def _planes(board, layer: str, xy, net: str):
    """Nearest plane of another net below and above a point on a layer."""
    from .pcb import reference_planes

    st = board.stackup
    z = st.layer(layer).z_mid
    below = above = None
    for lay, n, _ in reference_planes(board, layer, xy):
        if n == net or not n:
            continue
        zl = st.layer(lay).z_mid
        if zl < z and below is None:
            below = (lay, n)
        elif zl > z and above is None:
            above = (lay, n)
    return below, above


def section_for(board, layer: str, widths: tuple, gaps: tuple, below, above) -> Section:
    """The cross section of one or more traces on a layer with its planes and the stackup's dielectrics."""
    st = board.stackup
    lay = st.layer(layer)
    slabs = [(d.z_bottom, d.z_top, d.er) for d in st.dielectrics]
    # the space beside inner copper is filled with resin, as er as the dielectric next to it
    for cu in st.layers:
        if 0 < cu.index < len(st.layers) - 1:
            near = min(st.dielectrics, key=lambda d: abs(0.5 * (d.z_bottom + d.z_top) - cu.z_mid), default=None)
            if near is not None:
                slabs.append((cu.z_bottom, cu.z_top, near.er))
    planes = []
    if below:
        planes.append(st.layer(below[0]).z_top)
    if above:
        planes.append(st.layer(above[0]).z_bottom)
    if not below and not above:
        # no pour anywhere near: the far outer copper stands in for the return path, the most it can be
        far = st.layers[-1] if lay.index < len(st.layers) / 2 else st.layers[0]
        planes.append(far.z_top if far.z_mid < lay.z_mid else far.z_bottom)
    x = 0.0
    traces = []
    for i, w in enumerate(widths):
        traces.append((x, lay.z_bottom, x + w, lay.z_top))
        x += w + (gaps[i] if i < len(gaps) else 0.0)
    return Section(tuple(slabs), tuple(planes), tuple(traces))


def extract(board, nets: list[str] | None, work: Path, quality: str = "normal", rise: float = 1e-9,
            log=lambda s, d="": None) -> Extracted:
    from .pcb import net_graph

    names = [n for n in board.nets if n and (not nets or n in nets or n.lstrip("/") in nets)]
    tracks = [t for t in board.tracks if t.net in names and t.length > 0]
    if not tracks:
        raise ValueError("the board has no routed tracks on the nets asked for, so there are no trace parasitics "
                         "to work out")
    pieces = []
    for t in tracks:
        mid = (0.5 * (t.start[0] + t.end[0]), 0.5 * (t.start[1] + t.end[1]))
        below, above = _planes(board, t.layer, mid, t.net)
        p = Piece(t.net, t.layer, tuple(t.start), tuple(t.end), t.width, t.length, below, above)
        p.key = (t.layer, round(t.width, 6), below[0] if below else None, above[0] if above else None)
        pieces.append(p)
    weight = defaultdict(float)
    for p in pieces:
        weight[p.key] += p.length
    keys = sorted(weight, key=lambda k: -weight[k])
    cap = CAPS.get(quality, 14)
    sections: dict = {}
    log("solve", f"{min(len(keys), cap)} cross sections of {len(keys)} trace configurations")
    for i, key in enumerate(keys[:cap]):
        lay, w, bl, ab = key
        sec = section_for(board, lay, (w,), (), (bl, None) if bl else None, (ab, None) if ab else None)
        r = solve_section(sec, work / f"s{i}")
        sections[key] = {"z0": r["z0"], "l": r["l_per_m"], "c": r["c_per_m"], "delay": r["delay"],
                         "eps_eff": r["eps_eff"], "solved": True, "nodes": r["nodes"]}
    for key in keys[cap:]:
        sections[key] = _scaled(board, key, sections)
    for p in pieces:
        s = sections[p.key]
        p.z0, p.l, p.c, p.delay, p.solved = s["z0"], s["l"], s["c"], s["delay"], s["solved"]
    verify = _verify(board, keys[0], sections[keys[0]], work, quality)
    vias = []
    st = board.stackup
    for v in board.vias:
        if v.net not in names:
            continue
        lays = board.via_layers(v)
        h = st.distance(lays[0], lays[-1]) if len(lays) > 1 else st.board_thickness
        d = max(v.drill, 1e-5)
        vias.append({"net": v.net, "at": v.at, "length": h, "drill": d,
                     "l": MU0 * h / (2 * math.pi) * (math.log(4 * h / d) + 1)})
    graphs, totals = {}, {}
    for n in names:
        g = net_graph(board, n)
        graphs[n] = g
        ps = [p for p in pieces if p.net == n]
        if not ps:
            continue
        L = sum(p.l * p.length for p in ps)
        C = sum(p.c * p.length for p in ps)
        R = sum(RHO_CU * p.length / (p.width * st.layer(p.layer).thickness) for p in ps)
        lv = sum(v["l"] for v in vias if v["net"] == n)
        totals[n] = {"length_m": sum(p.length for p in ps), "z0_min": min(p.z0 for p in ps),
                     "z0_max": max(p.z0 for p in ps), "l_H": L + lv, "c_F": C, "r_ohm": R,
                     "delay_s": max(p.delay * p.length for p in ps) if len(ps) == 1 else
                     sum(p.delay * p.length for p in ps), "vias": sum(1 for v in vias if v["net"] == n),
                     "no_plane": sum(p.length for p in ps if not p.below and not p.above),
                     "connected": g.connected, "pads": list(g.pads), "ground": bool(GROUND.match(n))}
    pairs = _coupling(board, pieces, work, quality, rise, log)
    notes = list(board.notes)
    unsolved = sum(1 for k in keys[cap:])
    if unsolved:
        notes.append(f"{cap} trace configurations were solved as cross sections and {unsolved} rarer ones were "
                     "scaled from closed forms by the solved one nearest in width on the same layer.")
    return Extracted(board, pieces, sections, vias, totals, pairs, graphs, notes, verify)


def _closed(board, key) -> float:
    from .board import hammerstad_jensen, wheeler_stripline

    lay, w, bl, ab = key
    st = board.stackup
    t = st.layer(lay).thickness
    if bl and ab:
        b = st.gap_between(bl, ab)
        return wheeler_stripline(w, b, t, st.er_between(bl, ab))
    ref = bl or ab
    if ref is None:
        ref = st.layers[-1].name if st.layer(lay).index == 0 else st.layers[0].name
    return hammerstad_jensen(w, max(st.height_between(lay, ref), 1e-5), t, st.er_between(lay, ref))[0]


def _scaled(board, key, sections) -> dict:
    """A rare configuration: its closed form, scaled by how the nearest solved section compared with its own."""
    same = [k for k in sections if k[0] == key[0] and k[2:] == key[2:]] or list(sections)
    near = min(same, key=lambda k: abs(math.log(k[1] / key[1])))
    ratio = sections[near]["z0"] / _closed(board, near)
    z0 = _closed(board, key) * ratio
    ee = sections[near]["eps_eff"]
    v = 299792458.0 / math.sqrt(ee)
    return {"z0": z0, "l": z0 / v, "c": 1 / (z0 * v), "delay": 1 / v, "eps_eff": ee, "solved": False}


def _verify(board, key, s0: dict, work: Path, quality: str) -> dict:
    """The longest running configuration again on a finer mesh, and against its closed form."""
    lay, w, bl, ab = key
    sec = section_for(board, lay, (w,), (), (bl, None) if bl else None, (ab, None) if ab else None)
    fine = solve_section(sec, work / "verify", k=0.6) if quality != "fast" else None
    return {"key": key, "z0": s0["z0"], "z0_fine": fine["z0"] if fine else None, "closed": _closed(board, key),
            "kind": "stripline" if bl and ab else ("microstrip" if (bl or ab) else "no plane")}


def _coupling(board, pieces: list[Piece], work: Path, quality: str, rise: float, log) -> list[dict]:
    """Parallel neighbors of other nets on the same layer within three heights, solved as coupled pairs."""
    from scipy.spatial import cKDTree

    by_layer = defaultdict(list)
    for i, p in enumerate(pieces):
        by_layer[p.layer].append(i)
    st = board.stackup
    found: dict[tuple, dict] = {}
    seen: set = set()
    for lay, idx in by_layer.items():
        pts, own = [], []
        for i in idx:
            p = pieces[i]
            n = max(2, int(p.length / 2.5e-4) + 1)
            for t in np.linspace(0, 1, n):
                pts.append((p.a[0] + t * (p.b[0] - p.a[0]), p.a[1] + t * (p.b[1] - p.a[1])))
                own.append(i)
        if len(pts) < 2:
            continue
        tree = cKDTree(np.array(pts))
        own = np.array(own)
        reach = 2e-3
        for i, j in tree.query_pairs(reach, output_type="ndarray"):
            pa, pb = pieces[own[i]], pieces[own[j]]
            if pa.net == pb.net or own[i] == own[j]:
                continue
            k = (int(min(own[i], own[j])), int(max(own[i], own[j])))
            if k in seen:
                continue
            seen.add(k)
            r = _parallel(pieces[k[0]], pieces[k[1]], st)
            if r:
                found[k] = r
    pair_of = defaultdict(lambda: {"length": 0.0, "spacing": math.inf, "segments": []})
    for (ia, ib), r in found.items():
        a, b = pieces[ia], pieces[ib]
        key = tuple(sorted((a.net, b.net)))
        q = pair_of[key]
        q["length"] += r["overlap"]
        q["spacing"] = min(q["spacing"], r["s"])
        q["segments"].append((ia, ib, r))
    out = []
    if not pair_of:
        return out
    ranked = sorted(pair_of.items(), key=lambda kv: -kv[1]["length"] / max(kv[1]["spacing"], 1e-6))
    cache = {}
    n_solve = {"fast": 2, "normal": 5, "fine": 12}.get(quality, 5)
    for rank, (nets, q) in enumerate(ranked[:max(n_solve * 4, 10)]):
        ia, ib, r = max(q["segments"], key=lambda x: x[2]["overlap"])
        a, b = pieces[ia], pieces[ib]
        key = (a.layer, round(a.width, 6), round(b.width, 6), round(r["s"], 5), a.key[2], a.key[3])
        if key not in cache and len(cache) < n_solve:
            log("solve", f"coupling of {nets[0]} and {nets[1]}")
            bl = (a.key[2], None) if a.key[2] else None
            ab = (a.key[3], None) if a.key[3] else None
            sec = section_for(board, a.layer, (a.width, b.width), (r["s"],), bl, ab)
            cache[key] = solve_section(sec, work / f"x{len(cache)}")
        sol = cache.get(key)
        if sol is None:
            continue
        td = q["length"] * sol["delay"]
        nxt = sol["kb"] * min(1.0, 2 * td / rise)
        fxt = abs(sol["kf"]) * td / rise
        out.append({"nets": list(nets), "length_m": q["length"], "spacing_m": q["spacing"], "next": nxt, "fext": fxt,
                    "kb": sol["kb"], "kf": sol["kf"], "layer": a.layer})
    out.sort(key=lambda d: -max(d["next"], d["fext"]))
    return out


def _parallel(a: Piece, b: Piece, st) -> dict | None:
    da = np.subtract(a.b, a.a)
    db = np.subtract(b.b, b.a)
    la, lb = np.linalg.norm(da), np.linalg.norm(db)
    if la < 1e-6 or lb < 1e-6:
        return None
    ua, ub = da / la, db / lb
    if abs(float(ua @ ub)) < 0.985:
        return None
    n = np.array([-ua[1], ua[0]])
    d = abs(float(np.subtract(b.a, a.a) @ n))
    s = d - 0.5 * (a.width + b.width)
    h = _height(a, st)
    if s <= 0 or s > 3 * h:
        return None
    t0, t1 = sorted((float(np.subtract(b.a, a.a) @ ua), float(np.subtract(b.b, a.a) @ ua)))
    overlap = min(la, t1) - max(0.0, t0)
    if overlap <= 0:
        return None
    return {"overlap": overlap, "s": s}


def _height(p: Piece, st) -> float:
    refs = [x[0] for x in (p.below, p.above) if x]
    if not refs:
        return st.board_thickness
    return min(st.gap_between(p.layer, r) for r in refs)


# --- the circuit of the same board ----------------------------------------------------------------
ACTIVE = ("U", "IC", "Q", "M", "S", "T", "K")       # chips, transistors, switches: what drives a net


def anchor(graph, pads: list[str]) -> str:
    """The pad the rest of a net is measured from: a chip's or a switch's, the part that drives the net, then
    the part with the most pins on it, then the first one."""
    if not pads:
        return ""
    count = defaultdict(int)
    for p in pads:
        count[p.split(".")[0]] += 1

    def rank(p):
        ref = p.split(".")[0]
        letters = re.match(r"[A-Za-z]+", ref)
        active = bool(letters) and letters.group(0).upper() in ACTIVE
        return (active, count[ref], -pads.index(p))
    return max(pads, key=rank)


def _paths(graph, start: str, pieces_of: dict) -> dict:
    """R, L along the copper from the anchor pad to every other node (shortest path by length)."""
    import heapq

    adj = defaultdict(list)
    for e in graph.edges:
        adj[e.a].append((e.b, e))
        adj[e.b].append((e.a, e))
    best = {start: (0.0, 0.0, 0.0)}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > best[u][0] + 1e-15:
            continue
        for v, e in adj[u]:
            r, ind = pieces_of(e)
            nd = d + e.length
            if v not in best or nd < best[v][0] - 1e-15:
                best[v] = (nd, best[u][1] + r, best[u][2] + ind)
                heapq.heappush(heap, (nd, v))
    return best


def parasitic_spec(ex: Extracted, source: str) -> dict:
    """The board's parasitics as the circuit domain takes them: a trace R and L from each net's anchor pad to every
    other pad, and each net's copper capacitance to ground."""
    st = ex.board.stackup
    per_m = defaultdict(list)
    for p in ex.pieces:
        per_m[(p.net, p.layer, round(p.width, 6))].append(p)
    def rl_of(net):
        def rl(e):
            if e.kind == "via":
                return 0.0, MU0 * e.length / (2 * math.pi) * (math.log(4 * e.length / max(e.drill, 1e-5)) + 1)
            if e.kind != "track":
                return 0.0, 0.0
            ps = per_m.get((net, e.layer, round(e.width, 6)))
            lpm = float(np.mean([p.l for p in ps])) if ps else 0.0
            r = RHO_CU * e.length / (max(e.width, 1e-6) * st.layer(e.layer).thickness)
            return r, lpm * e.length
        return rl

    poured = {z.net for z in ex.board.zones}
    traces, nets = [], {}
    for net, g in ex.graphs.items():
        tot = ex.nets.get(net)
        if not tot or tot["ground"] and net in poured:
            continue                 # a ground pour is the reference the other nets' inductance already counts
        start = anchor(g, g.pads)
        if not start:
            continue
        name = "0" if tot["ground"] else net.lstrip("/")
        best = _paths(g, start, rl_of(net))
        for pad in g.pads:
            if pad == start or pad not in best:
                continue
            _, r, ind = best[pad]
            traces.append({"net": name, "to": pad.split("#")[0], "r": r, "l": ind})
        if not tot["ground"]:
            nets[name] = {"c": tot["c_F"]}
    return {"traces": traces, "nets": nets, "source": source}


def _schematic(path: Path, netlist: str | None) -> Path | None:
    if netlist:
        p = Path(netlist)
        return p if p.exists() else None
    folder = path if path.is_dir() else path.parent
    same = folder / (path.stem + ".kicad_sch")
    if same.exists():
        return same
    sch = list(folder.glob("*.kicad_sch"))
    return sch[0] if len(sch) == 1 else None


def _match(c, spec: dict) -> dict:
    """The parasitic traces and capacitances that land on parts and nets this circuit has."""
    from ..circuits.netlist import safe_node

    nets = set(c.nets())
    keep = []
    for tr in spec["traces"]:
        ref, _, pin = tr["to"].partition(".")
        e = c.element(ref)
        net = safe_node(tr["net"])
        if e is None or net not in nets:
            continue
        on = [i for i, n in enumerate(e.nodes) if n == net]
        if len(on) == 1:                 # the part touches the net once: no need to match pad numbers to pins
            keep.append({**tr, "to": ref})
        elif any(str(i + 1) == pin or e.pins[i] == pin.lower() for i in on):
            keep.append(tr)
    return {"traces": keep, "nets": {n: v for n, v in spec["nets"].items() if safe_node(n) in nets},
            "source": spec["source"]}


def _with_board(c0, used: dict):
    """The circuit with the board's traces spliced in and each net's copper capacitance spread over its net node
    and the pin ends of its traces, a pi model of the copper, so no pin hangs on an inductance with nothing to
    ground behind it."""
    from ..circuits.netlist import Element, join_parasitics, safe_node

    c1 = c0.copy()
    join_parasitics(c1, {**used, "nets": {}})
    ends = defaultdict(list)
    for e in c1.elements:
        if e.kind == "parasitic" and e.name.lower().startswith(("lpar", "rpar")):
            n = e.nodes[1]
            if "_pe" in n and n not in ends[n.split("_pe")[0]]:
                ends[n.split("_pe")[0]].append(n)
    for k, (net, v) in enumerate(used["nets"].items()):
        node = safe_node(net)
        spots = [node] + ends.get(node, [])
        for j, s in enumerate(spots):
            c1.elements.append(Element(f"Cpar{k}_{j}", "c", [s, "0"], [f"{v['c'] / len(spots):.6g}"], ["1", "2"],
                                       "parasitic"))
    return c1


# parts a fast edge's current loop closes through: capacitors, switches, diodes and transistors. Inductors and
# resistors hold the current back, and a supply is only taken when no capacitor closes the loop
FAST = ("c", "s", "w", "d", "m", "q", "j", "z")
# the nodes a part's main current runs between: a switch's contacts, not its control pair, a transistor's drain
# and source or collector and emitter, not its gate or base
MAIN = {"s": (0, 1), "w": (0, 1), "m": (0, 2), "q": (0, 2), "j": (0, 2), "z": (0, 2)}


def _main(x) -> list[str]:
    k = MAIN.get(x.letter)
    return [x.nodes[i] for i in k if i < len(x.nodes)] if k else list(x.nodes)


def hot_loop(c, part: str) -> list[str]:
    """The parts of the shortest loop through a part that its fast current takes: from one of its nodes back to
    the other through capacitors, switches, diodes and transistors (or a supply when nothing else closes it),
    with every part in parallel on the way. The board's traces to these parts are the loop's inductance."""
    e = c.element(part)
    if e is None or len(e.nodes) < 2:
        return [part]
    for letters in (FAST, FAST + ("v",)):
        edges = defaultdict(list)
        for x in c.elements:
            if x is e or x.letter not in letters or len(x.nodes) < 2:
                continue
            ends = _main(x)
            for a in ends:
                for b in ends:
                    if a != b:
                        edges[a].append((b, x))
        start, goal = e.nodes[0], e.nodes[1]
        prev, todo = {start: None}, [start]
        while todo and goal not in prev:
            nxt = []
            for n in todo:
                for m, x in edges[n]:
                    if m not in prev:
                        prev[m] = (n, x)
                        nxt.append(m)
            todo = nxt
        if goal in prev:
            refs, n = [e.name], goal
            while prev[n] is not None:
                a, x = prev[n]
                refs += [y.name for y in c.elements if y.letter in letters and {a, n} <= set(_main(y))
                         and y.name not in refs and y is not e]
                n = a
            return refs
    return [part]


LOOP_FACTORS = np.round(np.linspace(0.0, 3.0, 13), 3)        # times the board's own loop inductance


def sweep_loop(c0, used: dict, part: str, card: str, work: Path, log) -> dict | None:
    """The board's circuit solved again with the hot loop's trace inductance scaled from none to three times the
    board's, every other trace as the board has it. Each value keeps the peak voltage across the part, every
    net's peak voltage and every pad connection's peak current."""
    from ..circuits.netlist import safe_node
    from ..solvers import base
    from ..units import parse_spice_value

    loop = hot_loop(c0, part)
    mine = [k for k, tr in enumerate(used["traces"]) if tr["to"].split(".")[0].lower() in
            {r.lower() for r in loop}]
    L0 = sum(used["traces"][k]["l"] for k in mine)
    if not mine or L0 <= 0:
        return None
    sp = base.for_domain("circuit")
    rows = []
    for f in LOOP_FACTORS:
        u = {**used, "traces": [{**tr, "l": tr["l"] * f} if k in mine else tr for k, tr in enumerate(used["traces"])]}
        c = _with_board(c0, u)
        nodes = [n for n in c.nodes() if n != "0"]
        log("solve", f"the board's circuit with {fmt(f * L0 * 1e9)} nH in the hot loop")
        d, _, _ = sp.run(c.spice(analyses=[card]), work / "loop" / f"x{f:g}", [f"v({n})" for n in nodes], "loop",
                         timeout=900)

        def v(n):
            return np.zeros_like(d["time"]) if n == "0" else d[f"v({n})"]

        e = c.element(part)
        y = np.abs(v(e.nodes[0]) - v(e.nodes[1]))
        cur = []
        for k, tr in enumerate(u["traces"]):
            r = c.element(f"Rpar{k}")
            if r is not None:
                rr = max(parse_spice_value(r.rest[0]), 1e-9)
                cur.append(float(np.max(np.abs(v(r.nodes[0]) - v(r.nodes[1])))) / rr)
            else:
                cur.append(0.0)
        rows.append({"peak": float(y.max()), "current": cur,
                     "nodes": {n: float(np.max(np.abs(v(n)))) for n in c.nodes() if n != "0"
                               and not re.search(r"_p[te]\d+$", n)}})
    return {"factors": [float(f) for f in LOOP_FACTORS], "L0": L0, "loop": loop,
            "traces": [used["traces"][k]["to"] for k in mine], "part": part,
            "peak": [r["peak"] for r in rows],
            "current": [r["current"] for r in rows],
            "nodes": {n: [r["nodes"].get(n, 0.0) for r in rows] for n in rows[0]["nodes"]},
            "trace_to": [tr["to"] for tr in used["traces"]], "trace_net": [safe_node(tr["net"]) for tr in used["traces"]]}


def rating(path: Path, part: str) -> tuple[float | None, str]:
    """A part's voltage rating as Wobbl's parts library knows it from the netlist's note on that line (as
    '; part=SS34'), with where it comes from, or None."""
    from ..circuits.bind import bind
    from ..circuits.netlist import parse_spice

    try:
        c = parse_spice(Path(path).read_text(), Path(path).parent)
        bind(c)
    except Exception:  # noqa: BLE001
        return None, ""
    e = c.element(part)
    if e is None:
        return None, ""
    r = (e.part or {}).get("ratings") or {}
    key = next((k for k in ("vr_V", "vds_V", "vceo_V", "voltage_V", "vka_V", "vin_V") if r.get(k)), None)
    return (float(r[key]), str((e.part or {}).get("key") or "")) if key else (None, "")


def compare_spice(path: Path, spec: dict, work: Path, log) -> dict | None:
    """A SPICE netlist with a transient: run it as drawn and with the board's parasitics on a time step fine
    enough for the ringing they add, and compare every part's peak voltage and every net's swing."""
    from ..circuits.netlist import parse_spice
    from ..solvers import base
    from ..units import parse_spice_value

    c0 = parse_spice(Path(path).read_text(), Path(path).parent)
    tran = next((a for a in c0.analyses if a.lower().startswith(".tran")), None)
    if tran is None:
        return None
    used = _match(c0, spec)
    if not used["traces"] and not used["nets"]:
        return {"error": "no net of the board matched a net of the netlist by name"}
    c1 = _with_board(c0, used)
    bare = [e.name for e in c0.elements if e.letter in ("s", "w") and not any(
        x.letter == "c" and set(x.nodes) == set(e.nodes[:2]) for x in c0.elements)]
    tok = tran.split()
    nums = [t for t in tok[1:] if not t.lower().startswith("uic")]
    step, stop = parse_spice_value(nums[0]), parse_spice_value(nums[1])
    start = parse_spice_value(nums[2]) if len(nums) > 2 else 0.0
    lmax = max([t["l"] for t in used["traces"]] + [1e-9])
    f_ring = 1 / (2 * math.pi * math.sqrt(lmax * 100e-12))
    tmax = min(step, 1 / (40 * f_ring))
    uic = " uic" if "uic" in tran.lower() else ""
    card = f".tran {tmax:.4g} {stop:.6g} {start:.6g} {tmax:.4g}{uic}"
    sp = base.for_domain("circuit")
    out = {}
    for name, c in (("ideal", c0), ("board", c1)):
        nodes = [n for n in c.nodes() if n != "0"]
        log("solve", f"the netlist {'with the board' if name == 'board' else 'as drawn'}")
        data, _, _ = sp.run(c.spice(analyses=[card]), work / name, [f"v({n})" for n in nodes], name, timeout=900)
        out[name] = (c, data)
    c_i, d_i = out["ideal"]
    c_b, d_b = out["board"]

    def v(d, n):
        return np.zeros_like(d["time"]) if n == "0" else d[f"v({n})"]

    parts = []
    for e in c_i.elements:
        eb = c_b.element(e.name)
        if eb is None or len(e.nodes) != 2:
            continue
        pi = float(np.max(np.abs(v(d_i, e.nodes[0]) - v(d_i, e.nodes[1]))))
        pb = float(np.max(np.abs(v(d_b, eb.nodes[0]) - v(d_b, eb.nodes[1]))))
        parts.append({"part": e.name, "ideal_V": pi, "board_V": pb})
    parts.sort(key=lambda p: -(p["board_V"] - p["ideal_V"]))
    worst = parts[0] if parts else None
    wave = {}
    if worst:
        e, eb = c_i.element(worst["part"]), c_b.element(worst["part"])
        wave = {"time": d_b["time"], "board": np.abs(v(d_b, eb.nodes[0]) - v(d_b, eb.nodes[1])),
                "ideal": np.interp(d_b["time"], d_i["time"], np.abs(v(d_i, e.nodes[0]) - v(d_i, e.nodes[1])))}
        t, y = wave["time"], wave["board"]
        i = int(np.argmax(y))
        seg = y[i:]
        m = 0.5 * (seg.max() + seg.min()) if len(seg) > 10 else 0.0
        cross = t[i:][1:][np.diff(np.sign(seg - m)) != 0]
        worst["ring_Hz"] = float((len(cross) - 1) / (2 * (cross[-1] - cross[0]))) if len(cross) > 4 else 0.0
    if wave:
        # every net's voltage and every pad connection's current with the layout, for the 3D board
        wave["nodes"] = {n: v(d_b, n) for n in c_b.nodes() if not re.search(r"_p[te]\d+$", n)}
        cur = []
        for k, tr in enumerate(used["traces"]):
            e = c_b.element(f"Rpar{k}")
            if e is None:
                continue
            r = max(parse_spice_value(e.rest[0]), 1e-9)
            cur.append({"net": tr["net"], "to": tr["to"], "i": (v(d_b, e.nodes[0]) - v(d_b, e.nodes[1])) / r})
        wave["currents"] = cur
        vals = {}
        for e in c_i.elements:
            try:
                vals[e.name.upper()] = float(parse_spice_value(e.rest[0]))
            except Exception:  # noqa: BLE001
                pass
        wave["values"] = vals
    loop = None
    if worst:
        try:
            loop = sweep_loop(c0, used, worst["part"], card, work, log)
        except Exception as e:  # noqa: BLE001  the sweep feeds the 3D view, never the result
            log("solve", f"the hot loop sweep was skipped ({e})")
        if loop:
            loop["rating_V"], loop["rated_as"] = rating(path, worst["part"])
            loop["ideal_V"] = worst["ideal_V"]
    return {"parts": parts, "worst": worst, "wave": wave, "bare_switches": bare, "traces_used": len(used["traces"]),
            "traces_all": len(spec["traces"]), "nets_used": len(used["nets"]), "step": tmax, "schematic": str(path),
            "loop": loop}


def run_circuits(sch: Path, spec: dict, question: str, out: Path, log) -> dict | None:
    """The board's circuit with and without its parasitics: a SPICE netlist with a transient is compared part by
    part on a fine time step, anything else (a KiCad schematic) goes through the circuit study twice."""
    work = out / "work" / "circuit"
    if sch.suffix.lower() in (".cir", ".sp", ".spice", ".ckt", ".net") and \
            not sch.read_text(errors="replace").lstrip().startswith("(export"):
        try:
            r = compare_spice(sch, spec, work, log)
        except Exception as e:  # noqa: BLE001
            return {"error": f"the netlist run failed ({e})"}
        if r is not None:
            return r
    from ..circuits.study import load, simulate_circuit

    try:
        c = load(str(sch), {}, None)
    except Exception as e:  # noqa: BLE001
        return {"error": f"the schematic could not be read ({e})"}
    used = _match(c, spec)
    if not used["traces"] and not used["nets"]:
        return {"error": "no net of the board matched a net of the schematic by name"}
    log("solve", "the board's circuit without and with its parasitics")
    try:
        bare = simulate_circuit(str(sch), question, {}, out / "circuit-ideal", quick=True)
        laid = simulate_circuit(str(sch), question, {"parasitics": used}, out / "circuit-board", quick=True)
    except Exception as e:  # noqa: BLE001
        return {"error": f"the circuit run failed ({e})"}
    return {"bare": bare, "board": laid, "traces_used": len(used["traces"]), "traces_all": len(spec["traces"]),
            "nets_used": len(used["nets"]), "schematic": str(sch)}


# --- the study -------------------------------------------------------------------------------------
def run_parasitics(path: str, out_dir, question: str = "", netlist: str | None = None, nets: list | None = None,
                   quality: str = "normal", log=lambda s, d="": None, job=None, rise: float = 1e-9, keep=False):
    from ..outcome_generic import GenericOutcome
    from .pcb import read_board

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    log("read", Path(path).name)
    board = read_board(path)
    if not board.nets or all(not n for n in board.nets):
        from .pcb import infer_nets
        infer_nets(board)
    sch = _schematic(Path(path), netlist)
    if job is not None:
        st = board.stackup
        job.situate(objects=[{"board": board.name, "layers": board.copper_layers(), "nets": len(board.nets),
                              "thickness_mm": round(st.board_thickness * 1e3, 3)}])
        if not sch:
            job.assume("no schematic next to the board, so the parasitics are reported without a circuit run",
                       "--netlist board.kicad_sch")
        job.ready()
    ex = extract(board, nets, work, quality, rise, log)
    spec = parasitic_spec(ex, f"the board study of {Path(path).name}")
    (out / "parasitics.json").write_text(json.dumps(spec, indent=1))
    circ = run_circuits(sch, spec, question, out, log) if sch else None
    checks = []
    v = ex.verify
    if v.get("z0_fine"):
        d = abs(v["z0_fine"] - v["z0"]) / v["z0_fine"]
        checks.append(Check("mesh convergence", "pass" if d < 0.01 else ("warn" if d < 0.03 else "fail"),
                            f"the most used trace's impedance moved from {fmt(v['z0'])} to {fmt(v['z0_fine'])} ohm on "
                            f"a finer mesh ({100 * d:.2g}%)"))
    if v["kind"] != "no plane":
        d = abs(v["closed"] - v["z0"]) / v["closed"]
        checks.append(Check("closed form", "pass" if d < 0.05 else ("warn" if d < 0.12 else "info"),
                            f"the most used trace, a {v['kind']} on {v['key'][0]}, is {fmt(v['z0'])} ohm and the "
                            f"closed form for its average dielectric gives {fmt(v['closed'])} ohm ({100 * d:.2g}% "
                            "apart, the field solve keeps each layer of the stackup)"))
    if circ and circ.get("bare_switches"):
        sw = ", ".join(circ["bare_switches"])
        checks.append(Check("switch capacitance", "warn",
                            f"{sw} in the netlist has no capacitance across it, so when it opens the trace "
                            "inductance in series has nowhere to put its current but the switch's off resistance, "
                            "and the overshoot is far above what a real part with its output capacitance sees"))
    split = [n for n, t in ex.nets.items() if not t["connected"]]
    checks.append(Check("connectivity", "pass" if not split else "warn",
                        "every net's pads are joined by its copper" if not split else
                        f"{len(split)} net{'s are' if len(split) > 1 else ' is'} not joined by copper alone "
                        f"({', '.join(split[:4])}), so the board is unrouted there or joins through a part"))
    noplane = [n for n, t in ex.nets.items() if t["no_plane"] > 0.3 * t["length_m"] and not t["ground"]]
    if noplane:
        checks.append(Check("return path", "warn",
                            f"{len(noplane)} nets run mostly over no plane ({', '.join(noplane[:4])}), so their "
                            "return current takes whatever copper is near and their impedance is only a bound"))
    if circ and "board" in circ:
        checks += [c for c in circ["board"].checks if c.status != "pass"][:2]
    lvl, reasons = confidence(checks)
    sig = {n: t for n, t in ex.nets.items() if not t["ground"]}
    longest = max(sig, key=lambda n: sig[n]["length_m"]) if sig else None
    zs = [p.z0 for p in ex.pieces if not GROUND.match(p.net)] or [p.z0 for p in ex.pieces]
    head = (f"{Path(path).name} has {len(ex.nets)} routed nets on {len(board.copper_layers())} layers, with trace "
            f"impedances from {fmt(min(zs))} to {fmt(max(zs))} ohm.")
    if longest:
        t = sig[longest]
        head += (f" The longest, {longest}, runs {fmt(t['length_m'] * 1e3)} mm with {fmt(t['l_H'] * 1e9)} nH, "
                 f"{fmt(t['c_F'] * 1e12)} pF and {fmt(t['r_ohm'] * 1e3)} mohm.")
    verdict = "ok"
    if ex.pairs:
        w = ex.pairs[0]
        xt = max(w["next"], w["fext"])
        head += (f" The worst crosstalk is {100 * xt:.2g}% between {w['nets'][0]} and {w['nets'][1]} over "
                 f"{fmt(w['length_m'] * 1e3)} mm at a {fmt(rise * 1e9)} ns edge.")
        if xt > 0.05:
            verdict = "marginal"
    if circ and "worst" in circ and circ["worst"]:
        w = circ["worst"]
        head += (f" In {Path(circ['schematic']).name} the layout raises the peak across {w['part']} from "
                 f"{fmt(w['ideal_V'])} V on the schematic alone to {fmt(w['board_V'])} V"
                 + (f", ringing at {fmt(w['ring_Hz'] / 1e6)} MHz." if w.get("ring_Hz") else "."))
        if w["board_V"] > 1.3 * w["ideal_V"]:
            verdict = "marginal"
    elif circ and "board" in circ:
        b, a = circ["bare"], circ["board"]
        head += (f" Its circuit with the layout: {a.headline} Without it: {b.headline}")
        if a.verdict != b.verdict:
            verdict = a.verdict
    elif circ:
        head += f" The circuit run was skipped: {circ['error']}."
    fixes = []
    for w in ex.pairs[:2]:
        if max(w["next"], w["fext"]) > 0.05:
            fixes.append({"kind": "spacing", "title": f"Space {w['nets'][0]} and {w['nets'][1]} further apart",
                          "summary": f"They run {fmt(w['length_m'] * 1e3)} mm side by side {fmt(w['spacing_m'] * 1e3)} "
                                     "mm apart. Three times the height to the plane cuts coupling roughly tenfold.",
                          "auto": False})
    for n in noplane[:2]:
        fixes.append({"kind": "plane", "title": f"Put a ground pour under {n}",
                      "summary": "A plane right under the trace gives the return current a path next to it, which "
                                 "sets its impedance and cuts its loop inductance.", "auto": False})
    images = {"board": _map(out / "images" / "board.png", ex)}
    if circ and circ.get("wave"):
        from . import plots
        wv = circ["wave"]
        t = wv["time"]
        i = int(np.argmax(wv["board"]))
        span = max(40 * circ["step"], 0.02 * (t[-1] - t[0]))
        m = (t >= t[i] - span * 0.3) & (t <= t[i] + span)
        images["ringing"] = plots.curve(out / "images" / "ringing.png", (t[m] - t[i]) * 1e9,
                                        {"with the board": wv["board"][m], "schematic alone": wv["ideal"][m]},
                                        f"ns from the peak across {circ['worst']['part']}", "volts")
    metrics = [("nets", str(len(ex.nets))), ("impedance", f"{fmt(min(zs))} to {fmt(max(zs))} ohm"),
               ("worst crosstalk", f"{100 * max(ex.pairs[0]['next'], ex.pairs[0]['fext']):.2g}%" if ex.pairs else
                "none found"), ("sections solved", str(sum(1 for s in ex.sections.values() if s["solved"])))]
    st = board.stackup
    assumptions = [f"Stackup: {st.source}.", "Quasi static TEM cross sections of each trace with its planes; solder "
                   "mask, plating and surface roughness are left out, and loss is only the DC resistance of the "
                   "copper.", f"Crosstalk is for a {fmt(rise * 1e9)} ns edge on the aggressor."] + ex.notes
    if circ and "error" not in circ:
        assumptions.append(f"Parasitics joined {circ['traces_used']} of {circ['traces_all']} pad connections and "
                           f"{circ['nets_used']} net capacitances to the schematic {Path(circ['schematic']).name}, "
                           "each pad hanging off its net's main chip pad through the copper between them.")
        if "parts" in circ:
            assumptions.append("The parts are modeled as the netlist gives them. Ideal switches and diodes without "
                               "reverse recovery ring longer than real ones, whose losses damp the ringing.")
    values = {"nets": ex.nets, "pairs": ex.pairs[:20], "vias": len(ex.vias),
              "sections": [{"layer": k[0], "width_mm": k[1] * 1e3, "below": k[2], "above": k[3], **v}
                           for k, v in ex.sections.items()], "parasitics": "parasitics.json",
              "circuit": ({"parts": circ["parts"], "step_s": circ["step"]} if circ and "parts" in circ else None),
              "_figures": [["board", "Traces colored by impedance"], ["ringing", "The worst part with and without "
                                                                                 "the board"]]}
    o = GenericOutcome("board", Path(path).name, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       values, images, fixes, solver="GetDP cross sections with gmsh", t_total=time.time() - t0)
    _playback(out, ex)
    _view(out, o, ex, circ, Path(path).name, question)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _view(out, o, ex: Extracted, circ, title: str, question: str):
    """The board in 3D with its circuit ringing on it (fields.pcbscene)."""
    try:
        from . import pcbscene
    except ImportError:
        return None
    return pcbscene.write(out, o, ex, circ, title, question)


def _map(path, ex: Extracted) -> str:
    from matplotlib.collections import LineCollection

    from ..report.render import BG, FG
    from . import plots

    plt = plots._mpl()
    b = ex.board
    x0, y0, x1, y1 = b.bbox()
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    fig, ax = plots._axes(plt, (float(np.clip(5.0 * w / h + 1.6, 4.8, 11)), 5.6))
    for z in b.zones:
        for outer, _ in z.polygons:
            ax.fill(outer[:, 0] * 1e3, outer[:, 1] * 1e3, color="#2a3a4a", alpha=0.35, lw=0)
    for ring in b.outline:
        ax.plot(ring[:, 0] * 1e3, ring[:, 1] * 1e3, color="#8a8f98", lw=0.8)
    segs = [[(p.a[0] * 1e3, p.a[1] * 1e3), (p.b[0] * 1e3, p.b[1] * 1e3)] for p in ex.pieces]
    z = np.array([p.z0 for p in ex.pieces])
    lc = LineCollection(segs, array=z, cmap="viridis", linewidths=[max(0.6, p.width * 1e3 * 2.2) for p in ex.pieces])
    ax.add_collection(lc)
    for pr in ex.pairs[:3]:
        ax.annotate(f"{100 * max(pr['next'], pr['fext']):.2g}% crosstalk", xy=(0.02, 0.97 - 0.05 * ex.pairs.index(pr)),
                    xycoords="axes fraction", color=FG, fontsize=8)
    ax.set_xlim(x0 * 1e3 - 2, x1 * 1e3 + 2)
    ax.set_ylim(y0 * 1e3 - 2, y1 * 1e3 + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("mm", color=FG, fontsize=9)
    ax.set_title("traces colored by impedance, pours shaded", color=FG, fontsize=10)
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("impedance, ohm", color=FG, fontsize=9)
    cb.ax.tick_params(colors=FG, labelsize=8)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return plots.save(fig, path)


def _playback(out, ex: Extracted):
    from . import playback as pb

    pieces = sorted(ex.pieces, key=lambda p: -p.length)[:3000]
    return pb.write(out, "board-parasitics", {
        "tracks": "each track piece as [x0, y0, x1, y1] in mm with its layer index, width in mm and impedance in "
                  "ohm, longest first (at most 3000)",
        "layers": "copper layer names, top to bottom", "pairs": "the worst coupled net pairs with their near and "
                                                               "far end crosstalk fractions",
        "nets": "per net: length (m), inductance (H), capacitance (F), resistance (ohm)"},
        {"layers": ex.board.copper_layers(),
         "tracks": [[round(p.a[0] * 1e3, 3), round(p.a[1] * 1e3, 3), round(p.b[0] * 1e3, 3), round(p.b[1] * 1e3, 3),
                     ex.board.stackup.layer(p.layer).index, round(p.width * 1e3, 3), round(p.z0, 2)] for p in pieces],
         "pairs": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()} for d in ex.pairs[:10]],
         "nets": {n: {k: round(v, 15) for k, v in t.items() if isinstance(v, float)}
                  for n, t in list(ex.nets.items())[:200]}})
