"""KiCad schematics and netlist exports as Circuits, with each symbol's value, footprint and part number.

A .kicad_sch file (KiCad 6 and later) is traced the way eeschema connects it: each symbol's pins are placed
from its library drawing through the symbol's rotation and mirror, wires join at their ends, at junctions and
where an end lands on another wire, labels name the wires they sit on, power symbols and global labels join
nets across the sheet, hidden power pins join the net of their name, and hierarchical sheets connect through
their sheet pins. A netlist export (.net starting with '(export') already lists the nets and is read directly.

Each symbol then becomes a part: by its library (Device:R, Device:LED, Transistor_FET:...), its reference
letter and its value or part number field, mapped onto library models by pin name. Symbols that only mark
things (test points, mounting holes, logos, power flags) are left out, and a rail that nothing on the sheet
drives (a +5V power symbol fed from off the sheet) gets a supply at the voltage its name says.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .netlist import Circuit, NetlistError, safe_node

SKIP_REFS = ("TP", "H", "MH", "FID", "LOGO", "G", "#", "NT", "MK", "REF")
FIELDS = ("mpn", "manufacturer_part_number", "part number", "part_number", "partnumber", "mfr part", "mfr_pn",
          "manufacturer part number", "lcsc", "digikey", "mouser", "part")


# --- s-expressions ---------------------------------------------------------------------------------------------
def sexp(text: str) -> list:
    tokens = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text)
    stack: list[list] = [[]]
    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            if not stack:
                raise NetlistError("the KiCad file has an unmatched ')'")
            stack[-1].append(done)
        elif t.startswith('"'):
            stack[-1].append(bytes(t[1:-1], "utf-8").decode("unicode_escape", errors="replace"))
        else:
            stack[-1].append(t)
    if len(stack) != 1 or not stack[0]:
        raise NetlistError("the KiCad file is cut short or not an s-expression")
    return stack[0][0]


def _items(node: list, name: str) -> list[list]:
    return [x for x in node[1:] if isinstance(x, list) and x and x[0] == name]


def _one(node: list, name: str):
    r = _items(node, name)
    return r[0] if r else None


def _props(node: list) -> dict[str, str]:
    return {str(p[1]): str(p[2]) for p in _items(node, "property") if len(p) >= 3}


def _xy(node) -> tuple[float, float]:
    return round(float(node[1]), 3), round(float(node[2]), 3)


# --- symbol geometry -------------------------------------------------------------------------------------------------
def _apply(t, temp):
    return (t[0] * temp[0] + t[2] * temp[1], t[1] * temp[0] + t[3] * temp[1],
            t[0] * temp[2] + t[2] * temp[3], t[1] * temp[2] + t[3] * temp[3])


def transform(angle: float, mirror: str | None):
    """eeschema's symbol transform: library coordinates (y up) to sheet coordinates (y down), rotated
    counterclockwise on screen in steps of 90 degrees, then mirrored about x or y."""
    t = (1, 0, 0, -1)
    for _ in range(int(round(angle / 90)) % 4):
        t = _apply(t, (0, 1, -1, 0))
    if mirror == "x":
        t = _apply(t, (1, 0, 0, -1))
    elif mirror == "y":
        t = _apply(t, (-1, 0, 0, 1))
    return t


@dataclass
class Pin:
    number: str
    name: str
    kind: str          # passive, power_in, input, output ...
    x: float
    y: float
    unit: int
    hidden: bool = False


def _lib_pins(lib: list, libs: dict) -> list[Pin]:
    """Every pin of a library symbol with its unit, following 'extends' to the parent symbol."""
    ext = _one(lib, "extends")
    if ext is not None and str(ext[1]) in libs:
        lib = libs[str(ext[1])]
    out = []
    base = str(lib[1]).split(":")[-1]
    for sub in _items(lib, "symbol"):
        m = re.match(rf"^{re.escape(base)}_(\d+)_(\d+)$", str(sub[1]).split(":")[-1]) or \
            re.match(r"^.*_(\d+)_(\d+)$", str(sub[1]))
        unit, style = (int(m.group(1)), int(m.group(2))) if m else (0, 1)
        if style == 2:
            continue
        for p in _items(sub, "pin"):
            at = _one(p, "at")
            name = _one(p, "name")
            num = _one(p, "number")
            out.append(Pin(str(num[1]) if num else "", str(name[1]) if name else "", str(p[1]) if len(p) > 1 else "",
                           float(at[1]), float(at[2]), unit, "hide" in p[1:] or _one(p, "hide") is not None))
    return out


@dataclass
class Sym:
    ref: str
    lib_id: str
    value: str
    footprint: str
    fields: dict
    unit: int
    pins: list[tuple[Pin, tuple[float, float]]]
    power: bool = False
    dnp: bool = False
    prefix: str = ""                               # the sheet it sits on


# --- tracing ---------------------------------------------------------------------------------------------------------
class _Union:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _on_segment(p, a, b) -> bool:
    (x, y), (x1, y1), (x2, y2) = p, a, b
    if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) > 1e-3 * max(abs(x2 - x1) + abs(y2 - y1), 1):
        return False
    return min(x1, x2) - 1e-3 <= x <= max(x1, x2) + 1e-3 and min(y1, y2) - 1e-3 <= y <= max(y1, y2) + 1e-3


def _sheet(root: list, path: Path | None, uf: _Union, prefix: str, syms: list[Sym], names: dict, depth: int = 0,
           inst_path: str = "") -> dict[str, tuple]:
    """Trace one sheet into the shared union find; returns its hierarchical labels by name."""
    libs = {str(s[1]): s for s in (_one(root, "lib_symbols") or [None])[1:] if isinstance(s, list)}
    wires = []
    for w_ in _items(root, "wire"):
        pts = _one(w_, "pts")
        xs = [_xy(p) for p in pts[1:]] if pts else []
        wires += list(zip(xs, xs[1:]))
    pts = set()
    for a, b in wires:
        pts.add(a)
        pts.add(b)
    junctions = [_xy(_one(j, "at")) for j in _items(root, "junction")]
    labels = []
    for kind in ("label", "global_label", "hierarchical_label"):
        for lb in _items(root, kind):
            labels.append((kind, str(lb[1]), _xy(_one(lb, "at"))))
    pinpts = []
    for s in _items(root, "symbol"):
        lid = _one(s, "lib_id")
        if lid is None:
            continue
        lib = libs.get(str(lid[1]))
        if lib is None:
            continue
        at = _one(s, "at")
        mir = _one(s, "mirror")
        unit_n = _one(s, "unit")
        unit = int(unit_n[1]) if unit_n else 1
        t = transform(float(at[3]) if len(at) > 3 else 0.0, str(mir[1]) if mir else None)
        pr = _props(s)
        ref = pr.get("Reference", "?")
        for inst in _items(s, "instances"):
            for proj in _items(inst, "project"):
                for pth in _items(proj, "path"):
                    r = _one(pth, "reference")
                    if r is not None and (not inst_path or str(pth[1]).endswith(inst_path) or len(
                            _items(proj, "path")) == 1):
                        ref = str(r[1])
        placed = []
        for p in _lib_pins(lib, libs):
            if p.unit not in (0, unit):
                continue
            x = round(float(at[1]) + t[0] * p.x + t[1] * p.y, 3)
            y = round(float(at[2]) + t[2] * p.x + t[3] * p.y, 3)
            placed.append((p, (x, y)))
            pinpts.append((x, y))
        power = _one(lib, "power") is not None
        dnp = any(x == ["dnp", "yes"] for x in s[1:] if isinstance(x, list))
        syms.append(Sym(ref, str(lid[1]), pr.get("Value", ""), pr.get("Footprint", ""),
                        {k.lower(): v for k, v in pr.items()}, unit, placed, power, dnp, prefix))
        for p, xy in placed:
            key = (prefix, xy)
            uf.find(key)
            if power and not ref.startswith("#FLG") and "PWR_FLAG" not in pr.get("Value", "").upper():
                uf.join(key, ("global", pr.get("Value", p.name)))
                names[("global", pr.get("Value", p.name))] = pr.get("Value", p.name)
            elif p.hidden and p.kind == "power_in" and p.name:
                uf.join(key, ("global", p.name))
                names[("global", p.name)] = p.name
    # what lies on each wire joins it: wire ends, pins, labels and junctions (two wires crossing do not join)
    marks = list(pts) + junctions + pinpts + [xy for _, _, xy in labels]
    for a, b in wires:
        uf.join((prefix, a), (prefix, b))
        for m in marks:
            if m != a and m != b and _on_segment(m, a, b):
                uf.join((prefix, m), (prefix, a))
    for m in set(marks):
        uf.find((prefix, m))
    hier = {}
    for kind, name, xy in labels:
        key = ("global", name) if kind == "global_label" else (prefix, "label", name)
        uf.join((prefix, xy), key)
        names[key] = name if kind == "global_label" else (prefix.strip("/") + "/" + name).strip("/")
        if kind == "hierarchical_label":
            hier[name] = (prefix, xy)
    for sh in _items(root, "sheet"):
        pr = _props(sh)
        f = pr.get("Sheetfile") or pr.get("Sheet file")
        if not f or path is None or depth > 6:
            continue
        sub = path.parent / f
        if not sub.exists():
            continue
        sname = pr.get("Sheetname") or pr.get("Sheet name") or sub.stem
        uid = _one(sh, "uuid")
        child = _sheet(sexp(sub.read_text(errors="replace")), sub, uf, f"{prefix}/{sname}", syms, names, depth + 1,
                       str(uid[1]) if uid else "")
        for p in _items(sh, "pin"):
            name = str(p[1])
            xy = _xy(_one(p, "at"))
            uf.find((prefix, xy))
            for a, b in wires:
                if _on_segment(xy, a, b):
                    uf.join((prefix, xy), (prefix, a))
            if name in child:
                uf.join((prefix, xy), child[name])
    return hier


def parse_schematic(text: str, path: Path | None = None, params: dict | None = None) -> Circuit:
    root = sexp(text)
    if not root or root[0] != "kicad_sch":
        raise NetlistError("not a KiCad schematic (.kicad_sch from KiCad 6 or later)")
    uf, syms, names = _Union(), [], {}
    _sheet(root, Path(path) if path else None, uf, "", syms, names)
    # name each net: a power or global name first, then a label, else the first pin on it
    netname: dict = {}
    for key, nm in names.items():
        netname.setdefault(uf.find(key), []).append((0 if key[0] == "global" else 1, nm))
    merged: dict[str, list] = {}
    for s in syms:
        if s.power or s.dnp:
            continue
        pins = []
        for p, xy in s.pins:
            r = uf.find((s.prefix, xy))
            cand = sorted(netname.get(r, []))
            if not cand:
                netname[r] = cand = [(2, f"N_{s.ref}_{p.number or p.name}")]
            pins.append((p.number, p.name, cand[0][1], p.unit))
        merged.setdefault(s.ref, [s, []])[1].extend(pins)    # the units of one part come together
    return build([(s, pins) for s, pins in merged.values()],
                 "KiCad schematic " + (Path(path).name if path else ""), params or {})


def parse_netlist_export(text: str, params: dict | None = None) -> Circuit:
    root = sexp(text)
    if not root or root[0] != "export":
        raise NetlistError("not a KiCad netlist export")
    libpins = {}
    for lp in _items(_one(root, "libparts") or ["libparts"], "libpart"):
        key = (str(_one(lp, "lib")[1]) if _one(lp, "lib") else "", str(_one(lp, "part")[1]) if _one(lp, "part") else "")
        libpins[key] = {str(_one(p, "num")[1]): str(_one(p, "name")[1]) if _one(p, "name") else ""
                        for p in _items(_one(lp, "pins") or ["pins"], "pin")}
    nets: dict[str, list] = {}
    for n in _items(_one(root, "nets") or ["nets"], "net"):
        name = str(_one(n, "name")[1])
        for nd in _items(n, "node"):
            ref, pin = str(_one(nd, "ref")[1]), str(_one(nd, "pin")[1])
            fn = _one(nd, "pinfunction")
            nets.setdefault(ref, []).append((pin, str(fn[1]) if fn else "", name.lstrip("/"), 0))
    comps = []
    for cp in _items(_one(root, "components") or ["components"], "comp"):
        ref = str(_one(cp, "ref")[1])
        ls = _one(cp, "libsource")
        lib = (str(_one(ls, "lib")[1]), str(_one(ls, "part")[1])) if ls is not None else ("", "")
        flds = {str(_one(f, "name")[1]).lower() if _one(f, "name") else str(f[1]).lower(): str(f[-1])
                for f in _items(_one(cp, "fields") or ["fields"], "field")}
        for k in ("value", "footprint"):
            if _one(cp, k) is not None:
                flds[k] = str(_one(cp, k)[1])
        pins = []
        for num, fn, net, u in nets.get(ref, []):
            pins.append((num, fn or libpins.get(lib, {}).get(num, ""), net, u))
        s = Sym(ref, f"{lib[0]}:{lib[1]}", flds.get("value", ""), flds.get("footprint", ""), flds, 1, [])
        comps.append((s, pins))
    return build(comps, "KiCad netlist", params or {})


# --- symbols into parts ----------------------------------------------------------------------------------------------
def build(comps: list, title: str, params: dict) -> Circuit:
    from .symbols import attach, power_rails

    c = Circuit(title=title.strip(), source="kicad")
    for s, pins in comps:
        u = s.ref.upper()
        if not pins or u.startswith(("TP", "MH", "FID", "LOGO", "#", "REF")) or re.match(r"^H\d", u) or \
                s.lib_id.lower().startswith(("mechanical:", "graphic:")):
            continue
        for _, _, net, _ in pins:
            c.names[safe_node(net)] = net
        attach(c, s, pins, params)
    power_rails(c, params)
    return c


def mpn(s: Sym) -> str:
    for k in FIELDS:
        v = s.fields.get(k, "")
        if v and v not in ("~", "-", "DNP"):
            return v
    return ""

