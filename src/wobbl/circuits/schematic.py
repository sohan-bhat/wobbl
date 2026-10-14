"""Schematics from a SPICE netlist, laid out by hand with no drawing library.

A netlist that reads as a ladder is drawn as a schematic: one source from a node to ground on the
left, a chain of series parts along the top rail, and from each node branches of one or more parts
down to a shared ground rail. Switches and transistors get their drive source beside them. That
covers the built in templates and most small circuits. Anything else (bridges, parts in parallel
between two nodes, subcircuits, controlled sources) is drawn as a tidy list of the parts and the nets
that join them, which still shows the whole circuit at a glance.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..units import fmt, parse_spice_value
from .bench import BG, FG, FG2, MUTED, YEL, H, W, Svg, eng, num, poly

WIRE, INKED = "#b9bdb6", "#f2f2ee"
KIND = {"r": "resistor", "c": "capacitor", "l": "inductor", "d": "diode", "v": "voltage source",
        "i": "current source", "b": "behavioral source", "s": "switch", "w": "switch", "m": "MOSFET",
        "q": "transistor", "j": "JFET", "e": "controlled source", "f": "controlled source", "g": "controlled source",
        "h": "controlled source", "x": "subcircuit", "k": "coupling", "t": "line"}


@dataclass
class Part:
    name: str
    kind: str                 # the SPICE letter, lower case
    nodes: list[str]          # the two terminals the current runs between
    value: str
    pins: list[str]           # every node it touches, for the parts list
    ctrl: list[str] = field(default_factory=list)  # the control node of a switch or transistor


def _node(n: str) -> str:
    n = n.lower()
    return "0" if n in ("gnd", "gnd!") else n


def parse(net: str) -> tuple[str, list[Part], dict[str, str]]:
    """The title line, the parts and the .model lines of a netlist. Continuation lines are joined and
    subcircuit bodies and dot commands left out."""
    raw = net.splitlines()
    title = raw[0].lstrip("*").strip() if raw and raw[0].startswith("*") else ""
    lines: list[str] = []
    for ln in raw:
        s = ln.strip()
        if not s or s[0] in "*;":
            continue
        if s[0] == "+" and lines:
            lines[-1] += " " + s[1:]
        else:
            lines.append(s)
    parts, models, depth = [], {}, 0
    for s in lines:
        low = s.lower()
        if low.startswith(".subckt"):
            depth += 1
        elif low.startswith(".ends"):
            depth = max(depth - 1, 0)
        elif low.startswith(".model") and not depth:
            t = s.split(None, 2)
            if len(t) == 3:
                models[t[1].lower()] = t[2]
        elif not s.startswith(".") and not depth:
            t = s.split()
            k = t[0][0].lower()
            nd = [_node(x) for x in t[1:]]
            if k in "rcldvib" and len(t) >= 3:
                parts.append(Part(t[0], k, nd[:2], " ".join(t[3:]), nd[:2]))
            elif k == "s" and len(t) >= 5:
                parts.append(Part(t[0], k, nd[:2], " ".join(t[5:]), nd[:4], nd[2:3]))
            elif k == "m" and len(t) >= 5:
                parts.append(Part(t[0], k, [nd[0], nd[2]], " ".join(t[5:6]), nd[:4], nd[1:2]))
            elif k in "qj" and len(t) >= 5:
                parts.append(Part(t[0], k, [nd[0], nd[2]], t[4], nd[:3], nd[1:2]))
            elif k == "x":
                parts.append(Part(t[0], k, nd[:2], t[-1], nd[:-1]))
            elif len(t) >= 2:
                parts.append(Part(t[0], k, nd[:2], " ".join(t[3:]), nd[:4]))
    return title, parts, models


def _other(p: Part, n: str) -> str:
    return p.nodes[1] if p.nodes[0] == n else p.nodes[0]


def ladder(parts: list[Part]):
    """Read the parts as a ladder: (source, spine nodes, series parts, branches per node, drives), or None.

    A branch is a chain of (part, top node, bottom node) ending at ground; a drive is the source that
    feeds a switch's control node and nothing else.
    """
    if not parts or any(p.kind not in "rcldvibsmqj" for p in parts):
        return None
    ctrl = {c: p for p in parts if p.kind in "smqj" for c in p.ctrl if c != "0"}
    drives, power = {}, []
    for p in parts:
        hot = _other(p, "0") if "0" in p.nodes else None
        if p.kind == "v" and hot in ctrl and not any(hot in q.nodes for q in parts if q is not p):
            drives[ctrl[hot].name] = p
        else:
            power.append(p)
    if any(p.kind in "smqj" and p.name not in drives for p in power):
        return None
    main = next((p for p in power if p.kind in "vi" and "0" in p.nodes and p.nodes[0] != p.nodes[1]), None)
    if main is None:
        return None
    rest, used = [p for p in power if p is not main], set()
    spine, series, branches = [_other(main, "0")], [], {}

    def chain(p, start):
        out, cur = [], start
        while True:
            nxt = _other(p, cur)
            out.append((p, cur, nxt))
            if nxt == "0":
                return out
            if nxt in spine:
                return None
            nb = [q for q in rest if q.name not in used and nxt in q.nodes and all(q is not c[0] for c in out)]
            if len(nb) != 1:
                return None
            p, cur = nb[0], nxt

    while True:
        node = spine[-1]
        chains, onward = [], []
        for e in (q for q in rest if q.name not in used and node in q.nodes):
            if e.nodes[0] == e.nodes[1]:
                return None
            ch = chain(e, node)
            (chains if ch else onward).append(ch or e)
        long = [ch for ch in chains if len(ch) > 1]
        if not onward and len(spine) == 1 and len(chains) == 1 and len(chains[0]) > 1:
            onward, chains = [chains[0][0][0]], []  # a lone chain off the source: its first part runs along the top
        elif not onward and len(long) == 1 and long[0][0][0].kind in "rl":
            # the last stage of a ladder: its resistor or inductor runs on along the top, the rest hangs down
            onward, chains = [long[0][0][0]], [ch for ch in chains if ch is not long[0]]
        if len(onward) > 1:
            return None
        used.update(q.name for ch in chains for q, _, _ in ch)
        branches[node] = chains
        if not onward:
            break
        nxt = _other(onward[0], node)
        if nxt == "0" or nxt in spine:
            return None
        used.add(onward[0].name)
        series.append(onward[0])
        spine.append(nxt)
    if len(used) != len(rest):
        return None
    return main, spine, series, branches, drives


def _label(p: Part, models: dict[str, str]) -> list[str]:
    """The value lines under a part's name: 47 Ω, pulse 0 to 5 V at 500 kHz, a diode's model."""
    v = p.value.strip()
    if p.kind in "rcl":
        try:
            return [eng(parse_spice_value(v.split()[0]), {"r": "Ω", "c": "F", "l": "H"}[p.kind])]
        except Exception:  # noqa: BLE001  a parameter such as {rload}
            return [v[:14]]
    if p.kind in "vi":
        unit = "V" if p.kind == "v" else "A"
        m = re.match(r"(?i)^(?:dc\s*)?([-+0-9.][^\s(]*)\s*$", v)
        if m:
            x = parse_spice_value(m.group(1))
            return ["0 V, current sense"] if x == 0 and p.kind == "v" else [eng(x, unit)]
        m = re.match(r"(?i)^pulse\s*\(([^)]*)\)", v)
        if m:
            a = [parse_spice_value(x) for x in m.group(1).replace(",", " ").split()]
            out = [f"pulse {fmt(a[0])} to {fmt(a[1])} {unit}"]
            if len(a) >= 7 and 0 < a[6] < 1:
                out.append(f"{eng(1 / a[6], 'Hz')}, {100 * a[5] / a[6]:.0f}% on")
            return out
        m = re.match(r"(?i)^sin\s*\(([^)]*)\)", v)
        if m:
            a = [parse_spice_value(x) for x in m.group(1).replace(",", " ").split()]
            return [f"sine {fmt(a[1])} {unit}"] + ([eng(a[2], "Hz")] if len(a) > 2 else [])
        return [v[:14]]
    model = v.split()[0] if v else ""
    if p.kind == "s":
        ron = re.search(r"(?i)\bron\s*=\s*([0-9.eE+-]+[a-zA-Z]*)", models.get(model.lower(), ""))
        return [f"on {eng(parse_spice_value(ron.group(1)), 'Ω')}"] if ron else [model]
    return [model[:14]]


# ---- symbols, drawn along x from 0 to L and turned into place -----------------------------

def _symbol(s: Svg, p: Part, a, b, start: str, models, side: int = -1):
    """Draw p from point a to point b, with start the node at a. side is where a control lead leaves:
    -1 above a part that runs left to right, +1 to the left of one that runs downward."""
    (ax, ay), (bx, by) = a, b
    length = math.hypot(bx - ax, by - ay)
    ang = round(math.degrees(math.atan2(by - ay, bx - ax)))
    m = length / 2
    k = p.kind
    fwd = p.nodes[0] == start  # the part's first node is at the start
    s.open(transform=f"translate({num(ax)} {num(ay)}) rotate({ang})", fill="none", stroke=INKED, stroke_width=2,
           stroke_linejoin="round", stroke_linecap="round")
    body = {"r": 24, "c": 5, "l": 24, "d": 9, "v": 17, "i": 17, "b": 17, "s": 16, "m": 14, "q": 14, "j": 14}[k]
    s.add("path", d=f"M0 0H{num(m - body)}M{num(m + body)} 0H{num(length)}", stroke=WIRE)
    if k == "r":
        s.add("path", d=poly([(m - 24, 0), *((m - 24 + 4 * i, -7 if i % 2 else 7) for i in range(1, 12)),
                              (m + 24, 0)]))
    elif k == "c":
        s.add("path", d=f"M{num(m - 5)} -14V14M{num(m + 5)} -14V14")
    elif k == "l":
        s.add("path", d=f"M{num(m - 24)} 0" + "a6 6 0 0 1 12 0" * 4)
    elif k == "d":
        tip, base = (m + 9, m - 9) if fwd else (m - 9, m + 9)
        s.add("path", d=f"M{num(base)} -10L{num(tip)} 0L{num(base)} 10Z", fill=INKED)
        s.add("path", d=f"M{num(tip)} -10V10")
        if "led" in p.value.lower():  # light leaving the junction
            u = 1 if ang == 90 else -1  # away from the label, which sits right of an upright part
            for dx in (-3, 5):
                s.add("path", d=f"M{num(m + dx)} {13 * u}l6 {9 * u}m-4.5 {-0.4 * u} 4.5 {0.4 * u} 0.3 {-4.5 * u}",
                      stroke=YEL, stroke_width=1.5)
    elif k in "vib":
        s.add("circle", cx=m, r=17)
        lines = _label(p, models)
        if k == "v" and lines[0].startswith("0 V"):
            s.add("text", "A", x=m, y=5, fill=INKED, stroke="none", font_size=14, text_anchor="middle",
                  transform=f"rotate({-ang} {num(m)} 0)")
        elif k == "v" and lines[0].startswith("pulse"):
            s.add("path", d=f"M{num(m - 9)} 5H{num(m - 4)}V-5H{num(m + 3)}V5H{num(m + 9)}", stroke_width=1.6,
                  transform=f"rotate({-ang} {num(m)} 0)")
        elif k == "v":
            pl, mn = (m - 9, m + 9) if fwd else (m + 9, m - 9)
            minus = f"M{num(mn - 4)} 0h8" if ang % 180 == 0 else f"M{num(mn)} -4v8"  # level on the page
            s.add("path", d=f"M{num(pl - 4)} 0h8M{num(pl)} -4v8" + minus, stroke_width=1.6)
        else:
            s.add("path", d=f"M{num(m - 9)} 0H{num(m + 9)}m-6 -5 6 5 -6 5" if fwd else
                  f"M{num(m + 9)} 0H{num(m - 9)}m6 -5 -6 5 6 5", stroke_width=1.6)
    elif k == "s":
        s.add("circle", cx=m - 16, r=2.6, fill=INKED)
        s.add("circle", cx=m + 16, r=2.6, fill=INKED)
        s.add("path", d=f"M{num(m - 16)} 0L{num(m + 14)} {num(side * 11)}")
        s.add("path", d=f"M{num(m)} {num(side * 6)}V{num(side * 26)}", stroke_dasharray="3 3", stroke_width=1.4)
    else:  # a transistor: its channel or base bar, the drive lead leaving on the control side
        bar = side * 10
        s.add("path", d=f"M{num(m - 14)} 0L{num(m - 8)} {num(bar)}M{num(m + 14)} 0L{num(m + 8)} {num(bar)}")
        s.add("path", d=f"M{num(m - 12)} {num(bar)}H{num(m + 12)}" + (f"M{num(m - 12)} {num(bar + side * 6)}"
                                                                     f"H{num(m + 12)}" if k == "m" else ""))
        s.add("path", d=f"M{num(m)} {num(bar + side * (6 if k == 'm' else 0))}V{num(side * 26)}", stroke_width=1.6)
    s.close()
    if k not in "smqj":
        return None
    return (ax - side * 26, ay + m) if ang == 90 else (ax + m, ay + side * 26)


def _ground(s: Svg, x, y):
    s.add("path", d=f"M{num(x)} {num(y)}v8M{num(x - 12)} {num(y + 8)}h24M{num(x - 7)} {num(y + 13)}h14"
                    f"M{num(x - 2.5)} {num(y + 18)}h5", stroke=WIRE, stroke_width=2, fill="none",
          stroke_linecap="round")


def _names(s: Svg, x, y, p: Part, models, anchor="start"):
    s.text(x, y, p.name, size=13, fill=FG, weight=600, anchor=anchor)
    for i, line in enumerate(_label(p, models)):
        s.text(x, y + 16 + 15 * i, line, size=12, fill=YEL if i == 0 else FG2, anchor=anchor)


def _head(s: Svg, title: str):
    s.add("rect", width=s.w, height=s.h, fill=BG)
    s.text(24, 32, "SCHEMATIC", size=10.5, fill=MUTED, ls=1.3, weight=500)
    if title:
        s.text(24 + 9 * 7.6 + 18, 32, title[:58], size=12, fill=FG2)


def schematic(net: str, notes: dict[str, str] | None = None, ns: str = "sch") -> str:
    """schematic.svg for a netlist. notes put a reading next to a node, such as {"out": "5.07 V"}."""
    title, parts, models = parse(net)
    lad = ladder(parts)
    notes = {_node(k): v for k, v in (notes or {}).items()}
    return (_draw_ladder(lad, title, models, notes, ns) if lad else None) or _draw_list(title, parts, ns)


def _draw_ladder(lad, title, models, notes, ns):
    main, spine, series, branches, drives = lad
    place, col = [], 0  # (column, chain) for every branch, in drawing order
    for ch in branches[spine[0]]:
        col += 1
        place.append((col, ch))
    spans, node_col = [], {spine[0]: 0}
    for q, start, node in zip(series, spine, spine[1:]):
        spans.append((col, col + 1, q, start))
        # a switch hanging from this node has its drive on its left, so leave that column free
        col += 1 + any(p.kind in "smqj" for p, _, _ in (branches[node][0][:1] if branches[node] else []))
        node_col[node] = col
        for j, ch in enumerate(branches[node]):
            col += j > 0
            place.append((col, ch))
    depth = max([1] + [len(ch) for _, ch in place])
    # a switch in the bottom row needs a row under it for its drive
    depth += any(q.kind in "smqj" and i == len(ch) - 1 for _, ch in place for i, (q, _, _) in enumerate(ch))
    top_drive = any(q.kind in "smqj" for _, _, q, _ in spans)
    cw, rh = (170.0 if col <= 3 else 150.0 if col <= 5 else 125.0), 110.0
    yg = depth * rh

    def X(c):
        return c * cw

    s = Svg(ns)
    _head(s, title)
    at = len(s.body)
    ext = [X(0) - 20, X(col) + 20, -(90 if top_drive else 30), yg + 26]  # left, right, top, bottom
    last = max([c for c, _ in place] + [0])
    s.add("path", d=f"M0 0H{num(X(col))}M0 {num(yg)}H{num(X(last))}", stroke=WIRE, stroke_width=2, fill="none")
    _ground(s, X(last) / 2, yg)
    _symbol(s, main, (0, 0), (0, yg), spine[0], models, side=1)
    names = [(-26, yg / 2 - 8, main, "end")]
    gates = []
    for c0, c1, q, start in spans:
        g = _symbol(s, q, (X(c0), 0), (X(c1), 0), start, models)
        names.append(((X(c0) + X(c1)) / 2, 30, q, "middle"))
        if g:
            gates.append((g, q, -1))
    for c, ch in place:
        for i, (q, a, b) in enumerate(ch):
            y1, y2 = i * rh, (yg if i == len(ch) - 1 else (i + 1) * rh)
            g = _symbol(s, q, (X(c), y1), (X(c), y2), a, models, side=1)
            names.append((X(c) + 24, (y1 + y2) / 2 - 8, q, "start"))
            if g:
                gates.append((g, q, 1))
        if 0 < c < col:
            s.add("circle", cx=X(c), cy=0, r=3.6, fill=WIRE)
        if 0 < c < last:
            s.add("circle", cx=X(c), cy=yg, r=3.6, fill=WIRE)
    for (gx, gy), q, side in gates:
        d = drives[q.name]
        if side < 0:  # above a switch on the top rail: the drive sits up and to the left
            top = -66
            s.add("path", d=poly([(gx, gy), (gx, top), (gx - 30, top)]), stroke=WIRE, stroke_width=2, fill="none")
            _symbol(s, d, (gx - 30, top), (gx - 100, top), d.nodes[0], models)
            s.add("path", d=f"M{num(gx - 100)} {num(top)}v10", stroke=WIRE, stroke_width=2)
            _ground(s, gx - 100, top + 10)
            names.append((gx + 14, top - 6, d, "start"))
        else:  # beside a switch in a branch: the drive stands to its left, down to its own ground
            xl = gx - 40
            s.add("path", d=poly([(gx, gy), (xl, gy)]), stroke=WIRE, stroke_width=2, fill="none")
            _symbol(s, d, (xl, gy), (xl, gy + rh * 0.8), d.nodes[0], models, side=1)
            _ground(s, xl, gy + rh * 0.8)
            names.append((xl - 24, gy + 18, d, "end"))
            ext[3] = max(ext[3], gy + rh * 0.8 + 22)
    for x, y, q, anchor in names:
        w = 7.2 * max(len(line) for line in [q.name, *_label(q, models)])
        ext[0] = min(ext[0], x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0))
        ext[1] = max(ext[1], x + (0 if anchor == "end" else w / 2 if anchor == "middle" else w))
        _names(s, x, y, q, models, anchor)
    for node, c in node_col.items():
        tag = node + (f" {notes[node]}" if node in notes else "")
        s.text(X(c) + 8, -14, tag, size=11, fill=MUTED if node not in notes else FG2)
        ext[1] = max(ext[1], X(c) + 8 + 6.6 * len(tag))
    # scale the whole drawing to fill the panel under the title, never past 1.3 times or under 0.55
    k = min(1.3, (W - 48) / (ext[1] - ext[0]), (H - 66) / (ext[3] - ext[2]))
    if k < 0.55:
        return None
    tx = W / 2 - k * (ext[0] + ext[1]) / 2
    ty = 50 + (H - 58 - k * (ext[3] - ext[2])) / 2 - k * ext[2]
    s.body.insert(at, f'<g transform="translate({num(tx)} {num(ty)}) scale({k:.3f})">')
    s.close()
    return s.svg(f"Schematic of {title or 'the netlist'}, drawn from the netlist.")


def _draw_list(title: str, parts: list[Part], ns: str) -> str:
    """The fallback: every part with its nodes and value, then every net with what it joins."""
    nets: dict[str, list[str]] = {}
    for p in parts:
        for n in dict.fromkeys(p.pins):
            nets.setdefault(n, []).append(p.name)
    rows = max(len(parts), len(nets), 1)
    s = Svg(ns, W, max(H, 120 + 22 * rows))
    _head(s, title)
    s.text(24, 58, "Drawn as a list, since this circuit is not a simple ladder.", size=11, fill=MUTED)
    s.text(24, 92, "PARTS", size=10.5, fill=MUTED, ls=1.3)
    s.text(392, 92, "NETS", size=10.5, fill=MUTED, ls=1.3)
    for i, p in enumerate(parts):
        y = 118 + 22 * i
        s.text(24, y, p.name[:8], size=12, fill=FG, weight=600)
        s.text(96, y, " ".join(p.pins)[:16], size=12, fill=FG2)
        s.text(236, y, (p.value or KIND.get(p.kind, ""))[:18], size=12, fill=YEL)
    for i, (n, names) in enumerate(sorted(nets.items(), key=lambda kv: (kv[0] != "0", kv[0]))):
        y = 118 + 22 * i
        s.text(392, y, ("ground" if n == "0" else n)[:10], size=12, fill=FG, weight=600)
        s.text(480, y, ", ".join(names)[:22], size=12, fill=FG2)
    if not parts:
        s.text(24, 118, "No parts found in the netlist.", size=12, fill=FG2)
    return s.svg(f"Parts and nets of {title or 'the netlist'}, listed from the netlist.")
