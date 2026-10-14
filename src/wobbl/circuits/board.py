"""Where every part of a circuit sits on a breadboard and how it is wired, as plain data.

plan() builds any circuit the way a person would on a solderless breadboard. Each net gets a run of
column strips in the top half, in the order the current reaches them from the supply. A part between
two nets lies across from one run to the other. A part to ground stands across the center channel
into a strip of its own that a jumper ties to the ground rail. Transistors, ICs and modules get strips
of their own, joined to their nets by jumpers. The supply feeds the top rail, other sources and meters
sit beside the board with leads into their nets, and every wire carries the current the run measured.

The result is dicts, lists and numbers, so the SVG bench (draw.py) and the 3D viewer place the same
parts in the same holes. Positions are in breadboard pitch (0.1 inch, 2.54 mm) from the board's top
left corner, x across the columns and y down the rows:

    board        {"cols", "width", "height", "rows": {row: y}, "rails": {"t+": net, "t-": "0", "b-": "0"}}
    parts        [{"name", "kind", "body", "label", "value", "model", "color", "pins": [{"pin", "net", "hole",
                   "x", "y", "current"}], "current", "k", "power", "heat"}]
    wires        [{"kind": "rail" | "link" | "tap" | "jumper", "net", "color", "points": [[x, y], ...],
                   "current", "k"}]
    instruments  [{"name", "kind": "supply" | "generator" | "current" | "meter", "readings": [[text, unit]],
                   "leads": [{"post": "+" | "-", "net", "hole", "x", "y", "current", "k"}]}]
    nets         {net: {"volts", "color"}}

A hole is [column, row], rows a to j and the rails t+, t-, b- and b+. A part's heat is its power over
its rating (its current over its rating for an LED), so 1 is at the limit. A current is in amps, positive
along the wire's points, from a part's first pin to its second, or out of an instrument's + post, and
"k" holds 25 samples of its size over the run from 0 to 1 when it changes (None when it is steady).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..units import parse_spice_value
from .bench import Breadboard, eng
from .netlist import Circuit, Element
from .parts import lookup

ROWS = Breadboard.ROWS
WIRE = {"+": "#c8352b", "0": "#2b2e33"}
NET_COLORS = ["#e8842a", "#3fae5a", "#3d7bff", "#a56cf0", "#e6e6e6", "#e0c22a", "#2fb5b5", "#e0609a"]
BODY = {"resistor": "axial-resistor", "capacitor": "radial-capacitor", "electrolytic": "radial-capacitor",
        "inductor": "inductor", "diode": "axial-diode", "schottky": "axial-diode", "zener": "axial-diode",
        "led": "led-5mm", "npn": "to92", "pnp": "to92", "jfet": "to92", "nmos": "to220", "pmos": "to220",
        "regulator": "to220", "tl431": "to92"}
WAVE = ("pulse", "sin", "pwl", "exp", "sffm", "am(")


def xy(hole) -> tuple[float, float]:
    col, row = hole
    return 2.2 + col - 1, ROWS[row]


def rail_col(col: int) -> int:
    """The nearest column with a rail hole: every sixth one is a gap between groups of five."""
    return col - 1 if (col - 1) % 6 == 5 else col


def kind_of(e: Element, c: Circuit) -> str:
    """What a part is: the bound library kind, else read from its letter and model."""
    if e.kind:
        return e.kind
    fixed = {"r": "resistor", "c": "capacitor", "l": "inductor", "s": "switch", "w": "switch"}
    if e.letter in fixed:
        return fixed[e.letter]
    p = lookup(e.model) if e.model else None
    if p is not None:
        return p.kind
    model = c.models.get(e.model.lower(), "").lower() if e.model else ""
    if e.letter == "d":
        return "led" if "led" in (e.model + e.name).lower() else "diode"
    if e.letter == "q":
        return "pnp" if re.search(r"\bpnp\b", model) else "npn"
    if e.letter == "m":
        return "pmos" if re.search(r"\bpmos\b|pchan", model) else "nmos"
    if e.letter in "jz":
        return "jfet"
    return "ic" if e.letter == "x" else "module"


def source_role(e: Element) -> tuple[str, float | None]:
    """A source as a bench instrument: a supply, a signal generator, a current source or, for 0 V, an ammeter."""
    text = " ".join(e.rest).lower()
    if any(w in text for w in WAVE):
        return "generator", None
    m = re.search(r"(?:^|\s)(?:dc\s+)?([-+0-9.][^\s(]*)", text)
    try:
        dc = parse_spice_value(m.group(1)) if m else 0.0
    except Exception:  # noqa: BLE001
        dc = None
    if e.letter == "i":
        return "current", dc
    return ("meter" if dc == 0 and "ac" not in text.split() else "supply"), dc


def readings(c: Circuit, data: dict | None) -> tuple[dict, dict, float]:
    """Currents into each part's pins and node voltages from a run's vectors, with the time they span.

    An instrumented run (Circuit.spice(instrument=True)) names a 0 V source in front of every pin, so
    each part's current is read straight off. Otherwise resistors fall back on Ohm's law and sources on
    their own current."""
    data = data or {}
    t = np.asarray(data["time"], float) if "time" in data and len(data["time"]) > 1 else None
    vec = {k.lower(): np.atleast_1d(np.asarray(v, float)) for k, v in data.items() if k != "time"}
    volts = {k[2:-1]: v for k, v in vec.items() if k.startswith("v(") and k.endswith(")")}
    pins: dict[str, list] = {}
    for e, i, name in c.probes():
        s = vec.get(f"i({name})")
        if s is not None:
            pins.setdefault(e.name, [None] * len(e.nodes))[i] = s
    for e in c.elements:
        if e.name in pins or len(e.nodes) != 2:
            continue
        if e.letter in "vil" and f"i({e.name.lower()})" in vec:
            pins[e.name] = [vec[f"i({e.name.lower()})"], None]
        elif e.letter == "r" and e.value and all(n == "0" or n in volts for n in e.nodes):
            va, vb = (volts.get(n, np.zeros(1)) for n in e.nodes)
            pins[e.name] = [(va - vb) / e.value, None]
    for name, ps in pins.items():  # what enters one pin of a two pin part leaves the other
        if len(ps) == 2 and ps[1] is None and ps[0] is not None:
            ps[1] = -ps[0]
    return pins, volts, float(t[-1] - t[0]) if t is not None else 0.0


def _steady(s) -> float:
    """One number for a current: its mean over the second half of the run, or its rms when it alternates."""
    if s is None:
        return 0.0
    s = np.atleast_1d(s)
    tail = s[len(s) // 2:]
    mean, rms = float(np.mean(tail)), float(np.sqrt(np.mean(tail ** 2)))
    return mean if abs(mean) >= 0.25 * rms else rms


def _shape(s, peak: float):
    """25 samples of a current's size over the run, as a share of the largest current on the board."""
    if s is None or len(np.atleast_1d(s)) < 3 or peak <= 0:
        return None
    s = np.abs(np.atleast_1d(s))
    k = np.interp(np.linspace(0, len(s) - 1, 25), np.arange(len(s)), s) / peak
    return None if np.ptp(k) < 0.02 else [round(float(x), 3) for x in k]


class _Board:
    """Strips, holes and who holds them while the plan is laid out."""

    def __init__(self):
        self.taken: dict[tuple[int, str], str] = {}
        self.rows: dict[str, list[tuple[int, int]]] = {r: [] for r in "abcdefghij"}

    def free(self, col, row):
        return (col, row) not in self.taken

    def take(self, col, row, who):
        self.taken[(col, row)] = who
        return [col, row]

    def span_free(self, row, c0, c1):
        lo, hi = min(c0, c1), max(c0, c1)
        return all(hi < a or lo > b for a, b in self.rows[row]) and self.free(c0, row) and self.free(c1, row)


def plan(c: Circuit, data: dict | None = None) -> dict:
    """Lay the circuit out on a breadboard, wired, with the run's currents on every wire. See the module doc."""
    c = c.copy()  # the layout trims a MOSFET's body pin; the caller's circuit stays as it was
    els = [e for e in c.elements if e.letter not in "ka" and e.nodes]
    pins, volts, span = readings(c, data)
    peak = max([float(np.max(np.abs(s))) for ps in pins.values() for s in ps if s is not None] + [0.0])
    sources = [e for e in els if e.letter in "vi"]
    roles = {e.name: source_role(e) for e in sources}
    main = next((e for e in sources if roles[e.name][0] == "supply" and "0" in e.nodes and
                 e.nodes[0] != e.nodes[1]), None)
    supply_net = next((n for n in main.nodes if n != "0"), None) if main else None
    onboard = [e for e in els if e.letter not in "vi"]

    for e in onboard:
        if e.letter == "m" and len(e.nodes) == 4 and e.nodes[3] == e.nodes[2]:
            e.nodes, e.pins = e.nodes[:3], e.pins[:3]

    def ctrl_nodes(e):  # the control inputs of E, G and S elements sit after their current carrying nodes
        return [x.lower() for x in e.rest[:2]] if e.letter in "egs" and len(e.rest) >= 2 else []

    two = [e for e in onboard if len(e.nodes) == 2 and not ctrl_nodes(e)]
    blocks = [e for e in onboard if e not in two]
    shunts = {}
    for e in two:
        if "0" in e.nodes and e.nodes[0] != e.nodes[1]:
            shunts.setdefault(e.nodes[1] if e.nodes[0] == "0" else e.nodes[0], []).append(e)
    # the order the nets are laid out in: outward from the supply along the parts that join them
    links: dict[str, list[str]] = {}
    for e in onboard + sources:
        ns = [n for n in e.nodes + ctrl_nodes(e) if n != "0"]
        for a in ns:
            links.setdefault(a, []).extend(b for b in ns if b != a)
    order, todo = [], [supply_net] if supply_net else []
    todo += [n for n in links if n not in todo]
    while todo:
        n = todo.pop(0)
        if n in order or n is None:
            continue
        order.append(n)
        todo = [m for m in links.get(n, []) if m not in order] + todo
    degree = {n: sum(n in e.nodes + ctrl_nodes(e) for e in els) for n in order}
    runs, col = {}, 3
    for n in order:
        k = len(shunts.get(n, []))
        w = max(1, 2 * k - 1, math.ceil(degree[n] / 4))
        while any(rail_col(col + 2 * i) != col + 2 * i for i in range(k)):
            col += 1
        runs[n] = (col, col + w - 1)
        col += w + 2
    b = _Board()
    color = {n: WIRE["+"] if n == supply_net else NET_COLORS[i % len(NET_COLORS)] for i, n in enumerate(order)}
    color["0"] = WIRE["0"]
    wires, parts, instruments = [], [], []

    def wire(kind, net, a, z, cur=0.0, k=None, mid=None):
        pts = [list(xy(a))] + ([list(m) for m in mid] if mid else []) + [list(xy(z))]
        wires.append({"kind": kind, "net": net, "color": color.get(net, NET_COLORS[0]), "points": pts,
                      "current": cur, "k": k, "ends": [list(a), list(z)]})

    def free_hole(net, who, rows="abcde"):
        c0, c1 = runs[net]
        for cc in range(c0, c1 + 1):
            for r in rows:
                if b.free(cc, r):
                    return b.take(cc, r, who)
        return None

    # the runs: neighbours joined by short links, the supply's run fed from the top rail
    for n, (c0, c1) in runs.items():
        for cc in range(c0, c1):
            wire("link", n, b.take(cc, "a", "link"), b.take(cc + 1, "a", "link"))
        if n == supply_net:
            rc = rail_col(c0)
            wire("rail", n, [rc, "t+"], b.take(c0, "a" if b.free(c0, "a") else "b", "rail"))

    def part(e, holes, body=None):
        kd = kind_of(e, c)
        ps = pins.get(e.name, [None] * len(e.nodes))
        cur = _steady(ps[0]) if ps and ps[0] is not None else None
        val = e.value
        into = [_steady(s) if s is not None else None for s in ps] + [None, None]
        vs = [0.0 if n == "0" else (_steady(volts[n]) if n in volts else None) for n in e.nodes]
        known = [(v, i) for v, i in zip(vs, into) if v is not None and i is not None]
        power = sum(v * i for v, i in known) if len(known) == len(e.nodes) else None
        rated = next((float(v) for k, v in (e.part.get("ratings") or {}).items()
                      if k in ("power", "pd", "p_max", "power_W", "pd_W") and v), 0.25 if kd == "resistor" else None)
        if kd == "led" and cur is not None:
            r_ = e.part.get("ratings") or {}
            heat = abs(cur) / float(r_.get("if_A") or r_.get("current") or 0.03)
        else:
            heat = abs(power) / rated if power is not None and rated else None
        body = body or _package(e, c, kd)
        parts.append({"name": e.name, "kind": kd, "body": body,
                      "label": _label(e, kd), "value": val, "model": e.model,
                      "color": _led_color(e) if kd == "led" else None,
                      "pins": [{"pin": p, "net": n, "hole": h, "x": xy(h)[0], "y": xy(h)[1], "current": i}
                               for p, n, h, i in zip(e.pins + ["c+", "c-"], e.nodes + ctrl_nodes(e), holes, into)],
                      "current": cur, "k": _shape(ps[0], peak) if ps and ps[0] is not None else None,
                      "power": power, "heat": round(heat, 3) if heat is not None else None})

    # parts to ground stand across the channel, a jumper from their strip to the ground rail
    for n, es in shunts.items():
        if n not in runs:
            continue
        for i, e in enumerate(es):
            cc = runs[n][0] + 2 * i
            top, bot = b.take(cc, "e", e.name), b.take(cc, "f", e.name)
            holes = [top, bot] if e.nodes[0] == n else [bot, top]
            part(e, holes, "inductor-radial" if e.letter == "l" else None)
            b.take(cc, "j", "tap")
            wire("tap", "0", [cc, "j"], [rail_col(cc), "b-"])
    # parts between two nets lie across from one run to the other, a row each where they would overlap
    later = []
    for e in two:
        if "0" in e.nodes or e.nodes[0] == e.nodes[1]:
            continue
        (a0, a1), (z0, z1) = runs[e.nodes[0]], runs[e.nodes[1]]
        ca, cz = (a1, z0) if a1 < z0 else (a0, z1)
        row = next((r for r in "bdc" if b.span_free(r, ca, cz)), None)  # b then d, so neighbours have room
        if row is None or abs(cz - ca) > 12:
            later.append(e)
            continue
        b.rows[row].append((min(ca, cz), max(ca, cz)))
        part(e, [b.take(ca, row, e.name), b.take(cz, row, e.name)])
    # the rest get strips of their own past the runs, each pin's strip joined to its net
    for e in later + blocks:
        nodes = e.nodes + ctrl_nodes(e)
        kd = kind_of(e, c)
        body = "axial-resistor" if e in later else _package(e, c, kd)
        prm = e.part.get("params") or {}
        pinout, count = prm.get("pinout") or {}, prm.get("pin_count")
        if body == "dip" and count and all(p in pinout for p in e.pins):     # the real pinout from the library
            half = count // 2
            holes = [[col + pinout[p] - 1, "e"] if pinout[p] <= half else [col + count - pinout[p], "f"]
                     for p in e.pins] + [[col + half + i, "c"] for i in range(len(nodes) - len(e.pins))]
            width = half
        elif body == "dip":
            half = math.ceil(len(nodes) / 2)
            holes = ([[col + i, "e"] for i in range(half)] +  # pin 1 top left, counting round anticlockwise
                     [[col + half - 1 - i, "f"] for i in range(len(nodes) - half)])
            width = half
        else:
            step = 4 if e in later else 1
            holes = [[col + step * i, "c"] for i in range(len(nodes))]
            width = step * (len(nodes) - 1) + 1
        for h in holes:
            b.take(*h, e.name)
        part(e, holes, body if e not in later else None)
        for h, n in zip(holes, nodes):
            top = h[1] in "abcde"
            if n == "0":
                rr = "a" if top else "j"
                wire("tap", "0", b.take(h[0], rr, "tap"), [rail_col(h[0]), "t-" if top else "b-"])
            elif n in runs:
                end = next((r for r in ("abcde" if top else "jihgf") if b.free(h[0], r)), None)
                there = free_hole(n, "jumper")
                if end and there:
                    wire("jumper", n, b.take(h[0], end, "jumper"), there)
        col += width + 2
    # sources and meters beside the board, their leads into the rails or their nets' runs
    for e in sources:
        role, dc = roles[e.name]
        ps = pins.get(e.name, [None, None])
        out = -_steady(ps[0]) if ps[0] is not None else None  # out of the + post into the circuit
        leads = []
        for post, n in zip(("+", "-"), e.nodes):
            if e is main:
                hole = [2, "t+"] if n != "0" else [1, "t-"]
            elif n == "0":
                hole = [rail_col(max(2, 3 + len(instruments))), "t-"]
            else:
                hole = free_hole(n, e.name) if n in runs else None
            if hole:
                leads.append({"post": post, "net": n, "hole": hole, "x": xy(hole)[0], "y": xy(hole)[1],
                              "current": (out if post == "+" else -out) if out is not None else None,
                              "k": _shape(ps[0], peak) if ps[0] is not None else None})
        instruments.append({"name": e.name, "kind": "supply" if e is main else role, "leads": leads,
                            "readings": _readings(e, role, dc, out), "current": out})
    if any(w["ends"][1][1] == "b-" for w in wires) and main is not None:
        wire("rail", "0", [col, "t-"], [col, "b-"])  # the two ground rails tied at the far end
    _jumper_currents(wires, parts, instruments)
    cols = max(col + 1, 18)
    return {"board": {"cols": cols, "width": cols - 1 + 4.4, "height": 18.2, "rows": dict(ROWS),
                      "rails": {"t+": supply_net, "t-": "0", "b-": "0"}},
            "parts": parts, "wires": wires, "instruments": instruments, "span": span, "peak": peak,
            "nets": {n: {"volts": _steady(volts.get(n)) if n in volts else None, "color": color.get(n)}
                     for n in order}}


def _jumper_currents(wires, parts, instruments):
    """Each jumper's current by Kirchhoff's current law, leaf strips first: what a strip's parts and leads
    take out of it has to come in through its one remaining jumper."""
    def node(h):
        col, row = h
        return ("rail", row) if row in ("t+", "t-", "b-", "b+") else ("strip", col, row in "abcde")

    out: dict = {}
    for p in parts:
        for pin in p["pins"]:
            if pin["current"] is not None:
                out[node(pin["hole"])] = out.get(node(pin["hole"]), 0.0) + pin["current"]
    for ins in instruments:
        for ld in ins["leads"]:
            if ld["current"] is not None:
                out[node(ld["hole"])] = out.get(node(ld["hole"]), 0.0) - ld["current"]
    pending = [w for w in wires if w["kind"] in ("rail", "link", "tap", "jumper")]
    for w in pending:
        w["current"] = None
    while pending:
        degree: dict = {}
        for w in pending:
            for h in w["ends"]:
                degree[node(h)] = degree.get(node(h), 0) + 1
        leaf = next(((w, i) for w in pending for i, h in enumerate(w["ends"]) if degree[node(h)] == 1), None)
        if leaf is None:
            break
        w, i = leaf
        a, z = node(w["ends"][i]), node(w["ends"][1 - i])
        flow = out.pop(a, 0.0)               # the leaf strip takes this much out, so it flows in along w
        w["current"] = flow if i == 1 else -flow
        out[z] = out.get(z, 0.0) + flow
        pending.remove(w)


def _label(e: Element, kind: str) -> str:
    v = e.value
    if v is not None:
        return eng(v, {"r": "Ω", "c": "F", "l": "H"}.get(e.letter, ""))
    return e.model or kind


def _led_color(e: Element) -> str:
    text = (e.model + " " + e.name + " " + str(e.meta.get("note", ""))).lower()
    return next((k for k in ("white", "blue", "green", "yellow", "orange", "uv", "ir", "red") if k in text), "red")


def _readings(e: Element, role: str, dc: float | None, out: float | None) -> list[list[str]]:
    """What the instrument's display shows: its setting and the current it measures."""
    rows = []
    if role == "generator":
        text = " ".join(e.rest).lower()
        m = re.search(r"(pulse|sin)\s*\(([^)]*)\)", text)
        try:
            a = [parse_spice_value(x) for x in m.group(2).replace(",", " ").split()] if m else []
        except Exception:  # noqa: BLE001
            a = []
        if m and m.group(1) == "pulse" and len(a) >= 7 and a[6] > 0:
            rows = [_hz(1 / a[6]), [f"{a[1]:.2f}", "V"]]
        elif m and m.group(1) == "sin" and len(a) >= 3 and a[2] > 0:
            rows = [_hz(a[2]), [f"{a[1]:.3g}", "V"]]
        return rows or [["WAVE", ""]]
    if role in ("supply", "current") and dc is not None:
        rows.append([f"{dc:.2f}" if role == "supply" else f"{dc * 1000:.1f}", "V" if role == "supply" else "mA"])
    if out is not None:
        amps = abs(out)
        rows.append([f"{amps:.3f}" if amps < 10 else f"{amps:.2f}", "A"])
    return rows


def _hz(f: float) -> list[str]:
    for scale, unit in ((1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")):
        if f >= scale * 0.999:
            return [f"{f / scale:.2f}", unit]
    return [f"{f:.3g}", "Hz"]


def _package(e: Element, c: Circuit, kind: str) -> str:
    """The body a part comes in: the library's package when it knows it, else a MOSFET is a TO-92 unless it is
    rated or modeled as a power part."""
    pkg = str((e.part.get("params") or {}).get("package", "")).upper()
    if pkg.startswith(("DIP", "SOIC", "SOP")):
        return "dip"
    if pkg.startswith(("TO-220", "TO-126", "TO-3", "TO-263", "D2PAK", "SOT-223", "DPAK")):
        return "to220"
    if pkg.startswith(("TO-92", "SOT-23", "SOT-89")):
        return "to92"
    if kind in ("nmos", "pmos"):
        rated = float((e.part.get("ratings") or {}).get("id", 0) or (e.part.get("ratings") or {}).get("id_A", 0) or 0)
        kp = re.search(r"\bkp\s*=\s*([0-9.eE+-]+)", c.models.get(e.model.lower(), "").lower()) if e.model else None
        power = rated >= 2 or (kp is not None and float(kp.group(1)) >= 0.5)
        return "to220" if power else "to92"
    return BODY.get(kind, "dip" if e.letter == "x" else "module")
