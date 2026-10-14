"""Reader for Gerber RS-274X files with X2 attributes, the KiCad .gbrjob stackup and Excellon drills.

Gerber already has y pointing up, so coordinates only change units. Copper
layers get KiCad's names (L1 is F.Cu, the last is B.Cu, the rest In1.Cu ...).
Line draws become tracks, flashes become pads or vias, and regions become
zones. A region or draw tagged with a pad (%TO.P) becomes that pad instead.
Files without X2 attributes are sorted by their names (.gtl, .gbl, .g1 ...).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import geom
from .kicad import unescape_name
from .model import MM, Board, Pad, Track, Via, Zone, build_stackup, default_stackup

_CMD = re.compile(r"%([^%]*)%|([^%*]*)\*")
_WORD = re.compile(r"([GDMXYIJ])([+-]?[\d.]+)")
_UESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_EXPR = re.compile(r"\s*(\$\d+|\d*\.\d+|\d+\.?|[-+xX*/()])")


def _x2(s: str) -> str:
    """Undo the \\uXXXX escapes of X2 attribute values and the {slash} style escapes KiCad leaves in names."""
    return unescape_name(_UESC.sub(lambda m: chr(int(m.group(1), 16)), s) if "\\u" in s else s)


def _eval(expr: str, var: dict) -> float:
    """Evaluate an aperture macro expression with + - x / and parentheses, never with eval()."""
    toks = _EXPR.findall(expr.replace(" ", ""))
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else ""

    def factor():
        nonlocal pos
        t = peek()
        pos += 1
        if t in ("-", "+"):
            v = factor()
            return -v if t == "-" else v
        if t == "(":
            v = add()
            if peek() == ")":
                pos += 1
            return v
        if t.startswith("$"):
            return var.get(int(t[1:]), 0.0)
        try:
            return float(t)
        except ValueError:
            return 0.0

    def mul():
        nonlocal pos
        v = factor()
        while peek() in ("x", "X", "*", "/"):
            op = peek()
            pos += 1
            r = factor()
            v = v / r if op == "/" and r != 0 else (v * r if op != "/" else 0.0)
        return v

    def add():
        nonlocal pos
        v = mul()
        while peek() in ("+", "-"):
            op = peek()
            pos += 1
            r = mul()
            v = v + r if op == "+" else v - r
        return v

    return add() if toks else 0.0


def _rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    if not deg:
        return pts
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return pts @ np.array([[c, s], [-s, c]])


def _rect(cx, cy, w, h) -> np.ndarray:
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
                     [cx - w / 2, cy + h / 2]])


def macro_shapes(statements: list[str], params: list[float]):
    """Rings of the exposed primitives of an aperture macro, plus the codes it could not draw."""
    var = {i + 1: v for i, v in enumerate(params)}
    rings, unknown = [], set()
    for st in statements:
        st = st.strip()
        if not st:
            continue
        if st.startswith("$") and "=" in st:
            name, expr = st.split("=", 1)
            try:
                var[int(name[1:])] = _eval(expr, var)
            except ValueError:
                pass
            continue
        parts = st.split(",")
        head = parts[0].strip().split(" ")[0]
        try:
            code = int(float(head))
        except ValueError:
            unknown.add(head)
            continue
        if code == 0:
            continue
        m = [_eval(p, var) for p in parts[1:]]

        def arg(i, default=0.0, m=m):
            return m[i] if i < len(m) else default

        ring = None
        on = True
        if code == 1:
            on = arg(0, 1) != 0
            ring = _rotate(geom.circle_ring((arg(2), arg(3)), arg(1) / 2.0, 24), arg(4))
        elif code in (2, 20):
            on = arg(0, 1) != 0
            (x1, y1), (x2, y2), w = (arg(2), arg(3)), (arg(4), arg(5)), arg(1)
            d = math.hypot(x2 - x1, y2 - y1) or 1.0
            nx, ny = -(y2 - y1) / d * w / 2, (x2 - x1) / d * w / 2
            ring = _rotate(np.array([[x1 + nx, y1 + ny], [x2 + nx, y2 + ny], [x2 - nx, y2 - ny],
                                     [x1 - nx, y1 - ny]]), arg(6))
        elif code == 21:
            on = arg(0, 1) != 0
            ring = _rotate(_rect(arg(3), arg(4), arg(1), arg(2)), arg(5))
        elif code == 22:
            on = arg(0, 1) != 0
            ring = _rotate(_rect(arg(3) + arg(1) / 2, arg(4) + arg(2) / 2, arg(1), arg(2)), arg(5))
        elif code == 4:
            on = arg(0, 1) != 0
            n = int(arg(1))
            pts = np.array(m[2:2 + 2 * (n + 1)]).reshape(-1, 2)
            ring = _rotate(pts, arg(2 + 2 * (n + 1)))
        elif code == 5:
            on = arg(0, 1) != 0
            n = max(3, int(arg(1)))
            t = 2 * math.pi * np.arange(n) / n
            ring = _rotate(np.column_stack([arg(2) + arg(4) / 2 * np.cos(t), arg(3) + arg(4) / 2 * np.sin(t)]),
                           arg(5))
        elif code == 6:
            ring = _rotate(geom.circle_ring((arg(0), arg(1)), arg(2) / 2.0, 24), arg(8))
        elif code == 7:
            ring = _rotate(geom.circle_ring((arg(0), arg(1)), arg(2) / 2.0, 24), arg(5))
        else:
            unknown.add(str(code))
        if ring is not None and on and len(ring):
            rings.append(ring)
    return rings, unknown


@dataclass
class Aperture:
    """A Gerber aperture reduced to a pad shape, with kind, size and angle in meters and degrees."""

    shape: str
    size: tuple[float, float]
    angle: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)
    function: str = ""
    thermal: bool = False

    @property
    def width(self) -> float:
        return min(self.size) if self.size[0] and self.size[1] else max(self.size)

    def ring(self, at) -> np.ndarray:
        """Outline of a flash of this aperture at a point."""
        c = (at[0] + self.offset[0], at[1] + self.offset[1])
        w, h = self.size
        if self.shape == "circle":
            return geom.circle_ring(c, max(w, h) / 2.0, 16)
        a = math.radians(self.angle)
        ux, uy = math.cos(a), math.sin(a)
        if self.shape == "oval":
            half = (max(w, h) - min(w, h)) / 2.0
            if h > w:
                ux, uy = -uy, ux
            return capsule((c[0] - ux * half, c[1] - uy * half), (c[0] + ux * half, c[1] + uy * half), min(w, h))
        local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        return local @ np.array([[ux, uy], [-uy, ux]]) + np.asarray(c)


def capsule(p0, p1, width: float, n: int = 8) -> np.ndarray:
    """Outline of a round ended stroke from p0 to p1."""
    r = width / 2.0
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) < 1e-12:
        return geom.circle_ring(p0, r, 2 * n)
    th = math.atan2(dy, dx)
    a1 = th - math.pi / 2 + math.pi * np.arange(n + 1) / n
    a0 = a1 + math.pi
    return np.vstack([np.column_stack([p1[0] + r * np.cos(a1), p1[1] + r * np.sin(a1)]),
                      np.column_stack([p0[0] + r * np.cos(a0), p0[1] + r * np.sin(a0)])])


@dataclass
class _Obj:
    kind: str                     # flash, draw or region
    pts: np.ndarray               # flash point, draw polyline or region ring, meters
    ap: Aperture | None
    net: str
    ref: str
    pin: str
    function: str
    holes: list = field(default_factory=list)

    def rings(self) -> list[np.ndarray]:
        """Outlines covering the object, used for clearances and negative planes."""
        if self.kind == "region":
            return [self.pts]
        if self.kind == "flash":
            return [self.ap.ring(self.pts)]
        return [capsule(a, b, self.ap.width) for a, b in zip(self.pts[:-1], self.pts[1:])]


@dataclass
class GerberLayer:
    """What one Gerber file holds after parsing."""

    name: str
    function: str = ""
    polarity: str = "Positive"
    objects: list[_Obj] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    has_x2: bool = False
    has_nets: bool = False
    clears: list = field(default_factory=list)


def _coord(s: str, dec: int, digits: int, trailing: bool) -> float:
    if "." in s:
        return float(s)
    if trailing:
        neg = s.startswith("-")
        body = s.lstrip("+-").ljust(digits, "0")
        v = int(body) / 10 ** dec
        return -v if neg else v
    return int(s) / 10 ** dec


def parse_gerber(text: str, name: str = "") -> GerberLayer:
    """Parse one RS-274X file into flashes, draws and regions in meters."""
    text = text.replace("\r", "").replace("\n", "")
    lay = GerberLayer(name)
    dec, digits, trailing = 6, 10, False
    unit = None
    aps: dict[int, Aperture] = {}
    macros: dict[str, list[str]] = {}
    ta: dict[str, str] = {}
    to: dict[str, list[str]] = {}
    cur = None
    x = y = 0.0
    mode, multi, region, dark = 1, True, False, True
    last_op = 1
    contour: list = []
    rotation = 0.0
    counts = {"clear": 0, "step": 0, "block": 0, "macro": set(), "rect draws": 0}

    def scale() -> float:
        return unit if unit is not None else MM

    def attrs():
        net = to.get("N", [""])[0]
        net = "" if net == "N/C" else _x2(net)
        p = to.get("P", [])
        return net, _x2(p[0]) if p else "", _x2(p[1]) if len(p) > 1 else ""

    def close_contour():
        nonlocal contour
        if len(contour) >= 3:
            ring = np.array(contour, dtype=float)
            if np.hypot(*(ring[0] - ring[-1])) < 1e-9:
                ring = ring[:-1]
            if len(ring) >= 3:
                if dark:
                    net, ref, pin = attrs()
                    lay.objects.append(_Obj("region", ring, None, net, ref, pin, ta.get("AperFunction", "")))
                else:
                    lay.clears.append((len(lay.objects), ring))
        contour = []

    def arc(x0, y0, x1, y1, i, j):
        if multi:
            cx, cy = x0 + i, y0 + j
        else:
            best = None
            for sx in (1, -1):
                for sy in (1, -1):
                    c = (x0 + sx * abs(i), y0 + sy * abs(j))
                    r0, r1 = math.hypot(x0 - c[0], y0 - c[1]), math.hypot(x1 - c[0], y1 - c[1])
                    sw = _sweep(c, x0, y0, x1, y1, mode == 2, False)
                    if abs(sw) <= math.pi / 2 + 1e-6:
                        err = abs(r0 - r1)
                        if best is None or err < best[0]:
                            best = (err, c)
            cx, cy = best[1] if best else (x0 + i, y0 + j)
        sweep = _sweep((cx, cy), x0, y0, x1, y1, mode == 2, multi)
        r0, r1 = math.hypot(x0 - cx, y0 - cy), math.hypot(x1 - cx, y1 - cy)
        a0 = math.atan2(y0 - cy, x0 - cx)
        pts = geom.arc_points((cx, cy), 1.0, a0, sweep)
        rr = np.linspace(r0, r1, len(pts))[:, None]
        pts = np.array([cx, cy]) + (pts - np.array([cx, cy])) * rr
        pts[0], pts[-1] = (x0, y0), (x1, y1)
        return pts

    bad = 0
    for m in _CMD.finditer(text):
        ext, word = m.group(1), m.group(2)
        try:
            if ext is not None:
                stmts = [s for s in ext.split("*")]
                if stmts and stmts[0].startswith("AM"):
                    macros[stmts[0][2:]] = [s for s in stmts[1:] if s.strip()]
                    continue
                for st in stmts:
                    if not st:
                        continue
                    if st.startswith("FS"):
                        fm = re.match(r"FS([LTD]?)([AI]?)(?:N\d+)?(?:G\d+)?X(\d)(\d)Y(\d)(\d)", st)
                        if fm:
                            trailing = fm.group(1) == "T"
                            dec, digits = int(fm.group(4)), int(fm.group(3)) + int(fm.group(4))
                    elif st.startswith("MO"):
                        unit = 0.0254 if st[2:4] == "IN" else MM
                    elif st.startswith("AD"):
                        am = re.match(r"ADD(\d+)([^,]+)(?:,(.*))?$", st)
                        if am:
                            aps[int(am.group(1))] = _aperture(am.group(2), am.group(3), macros, scale(),
                                                              ta.get("AperFunction", ""), counts)
                    elif st.startswith("LP"):
                        if region:
                            close_contour()
                        dark = st[2:3] != "C"
                    elif st.startswith("LR"):
                        try:
                            rotation = float(st[2:])
                        except ValueError:
                            rotation = 0.0
                    elif st.startswith("SR"):
                        sm = re.match(r"SRX(\d+)Y(\d+)", st)
                        if sm and (int(sm.group(1)) > 1 or int(sm.group(2)) > 1):
                            counts["step"] += 1
                    elif st.startswith("AB") and len(st) > 2:
                        counts["block"] += 1
                    elif st.startswith("IPNEG"):
                        lay.polarity = "Negative"
                    elif st.startswith("TF"):
                        lay.has_x2 = True
                        key, _, val = st[2:].partition(",")
                        if key == ".FileFunction":
                            lay.function = val
                        elif key == ".FilePolarity":
                            lay.polarity = val
                    elif st.startswith("TA"):
                        key, _, val = st[2:].partition(",")
                        ta[key.lstrip(".")] = val
                    elif st.startswith("TO"):
                        key, _, val = st[2:].partition(",")
                        key = key.lstrip(".")
                        to[key] = val.split(",")
                        if key == "N":
                            lay.has_nets = True
                    elif st.startswith("TD"):
                        key = st[2:].lstrip(".")
                        if not key:
                            ta.clear()
                            to.clear()
                        else:
                            ta.pop(key, None)
                            to.pop(key, None)
                continue
            word = word.strip()
            if not word or word.startswith("G04") or (word.startswith("G4") and not word[2:3].isdigit()):
                continue
            op = None
            xs = ys = None
            i = j = 0.0
            for letter, num in _WORD.findall(word):
                if letter == "G":
                    g = int(float(num))
                    if g in (1, 2, 3):
                        mode = g
                    elif g == 36:
                        region, contour = True, []
                    elif g == 37:
                        close_contour()
                        region = False
                    elif g == 74:
                        multi = False
                    elif g == 75:
                        multi = True
                    elif g == 70:
                        unit = 0.0254
                    elif g == 71:
                        unit = MM
                elif letter == "D":
                    d = int(float(num))
                    if d >= 10:
                        cur = aps.get(d)
                    else:
                        op = d
                elif letter == "M":
                    continue
                elif letter == "X":
                    xs = num
                elif letter == "Y":
                    ys = num
                elif letter == "I":
                    i = _coord(num, dec, digits, trailing) * scale()
                elif letter == "J":
                    j = _coord(num, dec, digits, trailing) * scale()
            if op is None and (xs is not None or ys is not None):
                op = last_op
            if op is None:
                continue
            last_op = op
            nx = _coord(xs, dec, digits, trailing) * scale() if xs is not None else x
            ny = _coord(ys, dec, digits, trailing) * scale() if ys is not None else y
            if op == 1:
                pts = np.array([[x, y], [nx, ny]]) if mode == 1 else arc(x, y, nx, ny, i, j)
                if region:
                    if not contour:
                        contour = [(x, y)]
                    contour.extend(map(tuple, pts[1:]))
                elif cur is not None:
                    if dark:
                        if cur.shape not in ("circle",):
                            counts["rect draws"] += 1
                        net, ref, pin = attrs()
                        lay.objects.append(_Obj("draw", pts, cur, net, ref, pin, cur.function))
                    else:
                        lay.clears.extend((len(lay.objects), capsule(a, b, cur.width))
                                          for a, b in zip(pts[:-1], pts[1:]))
            elif op == 2:
                if region:
                    close_contour()
            elif op == 3 and cur is not None and not region:
                if dark:
                    net, ref, pin = attrs()
                    ap = cur if not rotation else Aperture(cur.shape, cur.size, (cur.angle + rotation) % 360,
                                                           tuple(_rotate(np.array([cur.offset]), rotation)[0]),
                                                           cur.function, cur.thermal)
                    lay.objects.append(_Obj("flash", np.array([nx, ny]), ap, net, ref, pin, cur.function))
                else:
                    lay.clears.append((len(lay.objects), cur.ring((nx, ny))))
            x, y = nx, ny
        except (ValueError, IndexError, ZeroDivisionError, TypeError):
            bad += 1
    if region:
        close_contour()
    if bad:
        lay.notes.append(f"{name} has {bad} commands that could not be read, which were skipped.")
    if unit is None:
        lay.notes.append(f"{name} does not state its units, so millimeters were assumed.")
    if lay.clears:
        _cut_regions(lay)
    if counts["step"]:
        lay.notes.append(f"{name} uses step and repeat, which was not expanded.")
    if counts["block"]:
        lay.notes.append(f"{name} uses block apertures, which were not expanded.")
    if counts["macro"]:
        lay.notes.append(f"{name} uses aperture macro primitives {', '.join(sorted(counts['macro']))} that were "
                         "approximated by the size of the rest of the aperture.")
    if counts["rect draws"]:
        lay.notes.append(f"{name} draws {counts['rect draws']} lines with a non round aperture. Their width is "
                         "the aperture's smaller side.")
    return lay


def _cut_regions(lay: GerberLayer) -> None:
    """Turn clear polarity objects into holes of the dark regions drawn before them."""
    idx = np.array([c[0] for c in lay.clears])
    box = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for _, r in lay.clears])
    used = np.zeros(len(idx), dtype=bool)
    for i, ob in enumerate(lay.objects):
        if ob.kind != "region":
            continue
        (x0, y0), (x1, y1) = ob.pts.min(axis=0), ob.pts.max(axis=0)
        sel = np.nonzero((idx > i) & (box[:, 0] <= x1) & (box[:, 2] >= x0) & (box[:, 1] <= y1) & (box[:, 3] >= y0))[0]
        ob.holes = [lay.clears[k][1] for k in sel]
        used[sel] = True
    if used.any():
        lay.notes.append(f"{lay.name} clears {int(used.sum())} shapes out of its pours, which became holes in them.")
    if (~used).any():
        lay.notes.append(f"{lay.name} has {int((~used).sum())} clear polarity shapes over tracks or pads, which were "
                         "ignored.")


def _sweep(c, x0, y0, x1, y1, cw: bool, multi: bool) -> float:
    a0 = math.atan2(y0 - c[1], x0 - c[0])
    a1 = math.atan2(y1 - c[1], x1 - c[0])
    tau = 2 * math.pi
    if cw:
        s = (a0 - a1) % tau
        if s < 1e-12 and multi:
            s = tau
        return -s
    s = (a1 - a0) % tau
    if s < 1e-12 and multi:
        s = tau
    return s


def _aperture(name: str, args: str | None, macros: dict, unit: float, function: str, counts: dict) -> Aperture:
    vals = []
    for v in (args or "").split("X"):
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if name == "C":
        d = (vals[0] if vals else 0.0) * unit
        return Aperture("circle", (d, d), function=function)
    if name in ("R", "O"):
        w = (vals[0] if vals else 0.0) * unit
        h = (vals[1] if len(vals) > 1 else vals[0] if vals else 0.0) * unit
        return Aperture("rect" if name == "R" else "oval", (w, h), function=function)
    if name == "P":
        d = (vals[0] if vals else 0.0) * unit
        return Aperture("circle", (d, d), vals[2] if len(vals) > 2 else 0.0, function=function)
    rings, unknown = macro_shapes(macros.get(name, []), vals)
    thermal = any(st.strip().split(",")[0].strip() == "7" for st in macros.get(name, []))
    counts["macro"].update(unknown)
    if name not in macros:
        counts["macro"].add(name)
    if not rings:
        return Aperture("rect", (0.0, 0.0), function=function)
    center, size, angle = geom.min_area_rect(np.vstack(rings))
    shape = "roundrect" if "roundrect" in name.lower() else "rect"
    if len(rings) == 1 and len(rings[0]) == 24 and abs(size[0] - size[1]) < 1e-9 * max(size[0], 1.0):
        shape = "circle"
    return Aperture(shape, (size[0] * unit, size[1] * unit), angle, (center[0] * unit, center[1] * unit),
                    function, thermal)


def parse_excellon(text: str) -> list[tuple[float, float, float]]:
    """Drill hits (x, y, diameter) in meters from an Excellon file."""
    unit, lead_zeros = MM, False
    digits = None
    tools: dict[int, float] = {}
    tool = None
    x = y = 0.0
    hits = []
    routed = False

    def num(s: str) -> float:
        if "." in s:
            return float(s) * unit
        whole, dec = digits or ((3, 3) if unit == MM else (2, 4))
        neg = s.startswith("-")
        body = s.lstrip("+-")
        if lead_zeros:
            body = body.ljust(whole + dec, "0")
        v = int(body or "0") / 10 ** dec * unit
        return -v if neg else v

    for raw in text.splitlines():
        line = raw.strip()
        fm = re.search(r"FILE_FORMAT=(\d):(\d)", line)
        if fm:
            digits = (int(fm.group(1)), int(fm.group(2)))
            continue
        if not line or line.startswith(";"):
            continue
        if line.startswith(("METRIC", "M71", "INCH", "M72")):
            unit = MM if line.startswith(("METRIC", "M71")) else 0.0254
            lead_zeros = "LZ" in line
            zm = re.search(r",(0+)\.(0+)", line)
            if zm:
                digits = (len(zm.group(1)), len(zm.group(2)))
            continue
        tm = re.match(r"T(\d+)(?:F[\d.]+)?(?:S[\d.]+)?C([\d.]+)", line)
        if tm:
            tools[int(tm.group(1))] = float(tm.group(2)) * unit
            continue
        tm = re.match(r"T(\d+)$", line)
        if tm:
            tool = int(tm.group(1))
            continue
        if line.startswith(("G00", "G01", "X", "Y")):
            parts = line.split("G85")
            x0, y0 = x, y
            for part in parts:
                xm = re.search(r"X([+-]?[\d.]+)", part)
                ym = re.search(r"Y([+-]?[\d.]+)", part)
                x = num(xm.group(1)) if xm else x
                y = num(ym.group(1)) if ym else y
                if len(parts) == 1:
                    break
                if part is parts[0]:
                    x0, y0 = x, y
            if tool not in tools:
                continue
            if len(parts) > 1 or (line.startswith("G01") and routed):
                # a slot, drilled (G85) or routed (G00 then G01), counts as a hole at its middle
                if line.startswith("G01") and routed:
                    x0, y0 = hits.pop()[:2]
                hits.append(((x0 + x) / 2.0, (y0 + y) / 2.0, tools[tool]))
                routed = False
            elif not line.startswith("G01"):
                hits.append((x, y, tools[tool]))
                routed = line.startswith("G00")
    return hits


def parse_gbrjob(text: str) -> dict:
    """Stackup, board thickness, layer count and file functions from a KiCad .gbrjob file."""
    try:
        job = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"The .gbrjob file is not valid JSON ({e.msg}). Regenerate it from KiCad's plot "
                         "dialog or leave it out.") from None
    out = {"files": {}, "thickness": None, "layers": None, "sheets": []}
    spec = job.get("GeneralSpecs", {}) or {}
    if isinstance(spec.get("BoardThickness"), (int, float)):
        out["thickness"] = float(spec["BoardThickness"]) * MM
    if isinstance(spec.get("LayerNumber"), (int, float)):
        out["layers"] = int(spec["LayerNumber"])
    for f in job.get("FilesAttributes", []) or []:
        if isinstance(f, dict) and "Path" in f:
            out["files"][str(f["Path"]).replace("\\", "/").split("/")[-1].lower()] = (
                str(f.get("FileFunction", "")), str(f.get("FilePolarity", "Positive")))
    for i, s in enumerate(job.get("MaterialStackup", []) or []):
        if not isinstance(s, dict):
            continue
        kind = str(s.get("Type", "")).lower()
        t = s.get("Thickness")
        if not isinstance(t, (int, float)) or t <= 0:
            continue
        if kind == "copper":
            out["sheets"].append(("copper", str(s.get("Name", f"L{i}")), float(t) * MM))
        elif kind == "dielectric":
            er = s.get("Epsilon", 4.5)
            tan = s.get("LossTangent", 0.02)
            out["sheets"].append(("dielectric", str(s.get("Name", f"dielectric {i}")), float(t) * MM,
                                  float(er) if isinstance(er, (int, float)) else 4.5,
                                  float(tan) if isinstance(tan, (int, float)) else 0.02,
                                  str(s.get("Material", "FR4"))))
    return out


# file roles

_EXT_ROLES = [
    (r"(^|[-_.])f[._]cu\b|\.gtl$|\.cmp$|toplayer|top[._]copper|copper[._]top", ("copper", 1)),
    (r"(^|[-_.])b[._]cu\b|\.gbl$|\.sol$|bottomlayer|bottom[._]copper|copper[._]bottom", ("copper", 99)),
    (r"(^|[-_.])in(\d+)[._]cu\b", ("inner", None)),
    (r"\.g(\d+)$|\.gp(\d+)$|\.ly(\d+)$|inner(\d+)|\.in(\d+)$", ("inner", None)),
    (r"edge[._]cuts|\.gko$|\.gm1$|\.gml$|outline|profile|\.gm$|\.oln$|boardoutline", ("profile", 0)),
    (r"\.gm\d+$", ("mechanical", 0)),
]


_TF_FUNCTION = re.compile(r"%TF\.FileFunction,([^*%]*)\*%")


def classify(name: str, func: str) -> tuple[str, int]:
    """Role of a file as (kind, copper order), kind being copper, profile, drill, mechanical or other.

    func is the X2 file function (from the file or the .gbrjob). Without it the file name decides.
    """
    if func:
        parts = func.split(",")
        if parts[0] == "Copper":
            lm = re.match(r"L(\d+)", parts[1] if len(parts) > 1 else "")
            n = int(lm.group(1)) if lm else 1
            return "copper", n
        if parts[0] == "Profile":
            return "profile", 0
        if parts[0] in ("Plated", "NonPlated"):
            return "drill", 0
        return "other", 0
    low = name.lower()
    for pat, (kind, order) in _EXT_ROLES:
        mm = re.search(pat, low)
        if mm:
            if kind == "inner":
                num = next((int(g) for g in mm.groups() if g and g.isdigit()), 1)
                return "copper", 1 + num
            return kind, order
    return "other", 0


def _side(lay: GerberLayer) -> str:
    parts = lay.function.split(",") if lay.function else []
    return parts[2] if len(parts) > 2 and parts[0] == "Copper" else ""


def looks_like_gerber(head: str) -> bool:
    return "%FS" in head or "%MO" in head or ("G04" in head and "*" in head) or "%TF." in head


def looks_like_excellon(head: str) -> bool:
    s = head.lstrip()
    return s.startswith("M48") or ("\nM48" in head[:2000]) or (s.startswith(";") and "M48" in head[:4000])


def read_gerber_texts(texts: dict[str, str], name: str = "board") -> Board:
    """Build a Board from Gerber, Excellon and .gbrjob file contents keyed by file name."""
    notes: list[str] = []
    job = None
    for fname, txt in texts.items():
        if fname.lower().endswith(".gbrjob"):
            try:
                job = parse_gbrjob(txt)
            except ValueError:
                notes.append(f"{fname} is not valid JSON, so it was ignored.")
    job_files = job["files"] if job else {}
    copper: list[tuple[int, str, GerberLayer]] = []
    profiles, mechanical, drills = [], [], []
    for fname, txt in sorted(texts.items()):
        low = fname.lower()
        if low.endswith(".gbrjob"):
            continue
        head = txt[:4000]
        if looks_like_excellon(head) and not looks_like_gerber(head):
            drills.extend(parse_excellon(txt))
            continue
        if not looks_like_gerber(head):
            continue
        fm = _TF_FUNCTION.search(head)
        func = fm.group(1) if fm else job_files.get(low, ("", ""))[0]
        kind, order = classify(fname, func)
        if kind == "other":
            continue                      # silkscreen, mask, paste and drawings are never parsed
        if kind == "mechanical":
            mechanical.append((fname, txt))
            continue
        lay = parse_gerber(txt, fname)
        if not lay.function and func:
            lay.function = func
            lay.polarity = job_files.get(low, ("", lay.polarity))[1] or lay.polarity
        if kind == "copper":
            copper.append((order, fname, lay))
        elif kind == "profile":
            profiles.append(lay)
        else:
            drills.extend((float(ob.pts[0]), float(ob.pts[1]), max(ob.ap.size)) for ob in lay.objects
                          if ob.kind == "flash")
    if not copper:
        raise ValueError("None of the files is a copper Gerber layer. Pass the folder or zip with the copper "
                         "layers (for example F_Cu.gbr and B_Cu.gbr, or .gtl and .gbl).")
    copper.sort(key=lambda c: c[0])
    last = max(c[0] for c in copper)
    layers: dict[str, GerberLayer] = {}
    for order, fname, lay in copper:
        side = _side(lay)
        if side == "Top" or (not side and order == 1):
            nm = "F.Cu"
        elif side == "Bot" or (not side and order == last and order > 1):
            nm = "B.Cu"
        else:
            nm = f"In{order - 1}.Cu"
        if nm in layers:
            notes.append(f"{fname} repeats copper layer {nm} and was left out.")
            continue
        layers[nm] = lay
        notes.extend(lay.notes)
    names = list(layers)
    order_names = sorted(names, key=lambda s: 0 if s == "F.Cu" else 1000 if s == "B.Cu" else int(s[2:-3]))
    # stackup
    stack = None
    if job and job["sheets"]:
        cu = [s[1] for s in job["sheets"] if s[0] == "copper"]
        if len(cu) == len(order_names) and any(s[0] == "dielectric" for s in job["sheets"]):
            # the job may use renamed layers, so copper sheets are matched by their order
            rename = iter(order_names)
            sheets = [(s[0], next(rename)) + tuple(s[2:]) if s[0] == "copper" else s for s in job["sheets"]]
            stack = build_stackup(sheets, ".gbrjob")
    if stack is None:
        stack = default_stackup(order_names, job["thickness"] if job else None)
        notes.append(f"No stackup came with the Gerbers. Using a {stack.source}.")
    if not profiles and mechanical:
        profiles = [parse_gerber(mechanical[0][1], mechanical[0][0])]
        notes.append(f"No board edge file, so the outline was taken from {mechanical[0][0]}.")
    b = _assemble(name, stack, layers, profiles, drills, notes)
    return b


def _outline(profiles, notes) -> list[np.ndarray]:
    outline = []
    for prof in profiles:
        lines = [ob.pts for ob in prof.objects if ob.kind == "draw"]
        rings, left = geom.chain_rings(lines, 1e-5)
        if left:
            more, left = geom.chain_rings(left, 2e-4)
            rings += more
        if left:
            notes.append(f"{len(left)} pieces of the board edge in {prof.name} do not close and were left out.")
        outline += rings + [ob.pts for ob in prof.objects if ob.kind == "region"]
    outline.sort(key=geom.ring_area, reverse=True)
    return outline


def _negative_plane(lay: GerberLayer, nm: str, outline, notes) -> Zone | None:
    """A plane drawn in negative, read as the board area with every drawn shape cut out and thermal flashes kept."""
    if not lay.objects:
        return None
    if outline:
        outer = outline[0]
    else:
        allp = np.vstack([r for ob in lay.objects for r in ob.rings()])
        (x0, y0), (x1, y1) = allp.min(axis=0) - MM, allp.max(axis=0) + MM
        outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    holes = [r for ob in lay.objects if not (ob.kind == "flash" and ob.ap.thermal) for r in ob.rings()]
    notes.append(f"Layer {nm} is a plane drawn in negative. It was read as the board area with its {len(holes)} "
                 "clearances cut out.")
    return Zone("", nm, [(outer, holes)])


def _assemble(name, stack, layers: dict[str, GerberLayer], profiles, drills, notes) -> Board:
    names = stack.names()
    tracks, zones_raw = [], {}
    pads: dict[tuple, Pad] = {}
    vias: dict[tuple, dict] = {}
    skipped = 0
    any_nets = any(lay.has_nets for lay in layers.values())
    x2_functions = any(ob.function for lay in layers.values() for ob in lay.objects if ob.kind == "flash")
    outline = _outline(profiles, notes)
    planes = []
    for nm in names:
        lay = layers.get(nm)
        if lay is None:
            continue
        if lay.polarity.lower().startswith("neg"):
            plane = _negative_plane(lay, nm, outline, notes)
            if plane is not None:
                planes.append(plane)
            continue
        for ob in lay.objects:
            func = ob.function
            if ob.kind == "flash":
                x, y = float(ob.pts[0]) + ob.ap.offset[0], float(ob.pts[1]) + ob.ap.offset[1]
                key = (round(x / 1e-6), round(y / 1e-6))
                if "ViaPad" in func:
                    v = vias.setdefault(key, {"at": (x, y), "layers": [], "d": 0.0, "net": ob.net})
                    v["layers"].append(nm)
                    v["d"] = max(v["d"], max(ob.ap.size))
                    continue
                _add_pad(pads, (ob.ref, ob.pin) + key, ob, nm, (x, y), ob.ap.size, ob.ap.shape, ob.ap.angle)
            elif ob.kind == "draw":
                if ob.ref:
                    a, z = ob.pts[0], ob.pts[-1]
                    length = float(np.hypot(*(z - a)))
                    ang = math.degrees(math.atan2(z[1] - a[1], z[0] - a[0])) % 180.0
                    c = (a + z) / 2.0
                    key = (round(c[0] / 1e-6), round(c[1] / 1e-6))
                    _add_pad(pads, (ob.ref, ob.pin) + key, ob, nm, (float(c[0]), float(c[1])),
                             (length + ob.ap.width, ob.ap.width), "oval", ang)
                    continue
                if func.startswith("NonConductor") or (any_nets and not ob.net and func and func != "Conductor"):
                    skipped += 1
                    continue
                w = ob.ap.width
                for p0, p1 in zip(ob.pts[:-1], ob.pts[1:]):
                    if p0[0] != p1[0] or p0[1] != p1[1]:
                        tracks.append(Track(ob.net, nm, (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])),
                                            w))
            else:
                if ob.ref:
                    center, size, ang = geom.min_area_rect(ob.pts)
                    key = (round(center[0] / 1e-6), round(center[1] / 1e-6))
                    _add_pad(pads, (ob.ref, ob.pin) + key, ob, nm, center, size, "custom", ang)
                    continue
                if func.startswith("NonConductor"):
                    skipped += 1
                    continue
                zones_raw.setdefault((nm, ob.net), []).append((ob.pts, list(ob.holes)))
    if skipped:
        notes.append(f"{skipped} copper drawings marked as not conductors (text, logos) were left out.")
    # drills give via and pad hole sizes
    tree = None
    if drills:
        from scipy.spatial import cKDTree
        hits = np.array(drills)
        tree = cKDTree(hits[:, :2])

    def drill_at(xy):
        if tree is None:
            return 0.0
        d, k = tree.query(xy)
        return float(drills[k][2]) if d < 0.05 * MM else 0.0

    via_list = []
    for v in vias.values():
        lays = sorted(set(v["layers"]), key=names.index)
        if len(lays) == 1:
            lays = [lays[0], lays[0]]
        via_list.append(Via(v["net"], v["at"], drill_at(v["at"]), v["d"], (lays[0], lays[-1])))
    # inner layer flashes of a through hole pad often carry no pad attribute, so fold them into the named pad
    named: dict[tuple, list[Pad]] = {}
    for p in pads.values():
        if p.ref:
            named.setdefault((round(p.at[0] / 1e-6), round(p.at[1] / 1e-6)), []).append(p)
    for key, p in list(pads.items()):
        if p.ref:
            continue
        owners = named.get((round(p.at[0] / 1e-6), round(p.at[1] / 1e-6)), [])
        owner = min(owners, key=lambda o: abs(max(o.size) - max(p.size)), default=None)
        if owner is not None:
            owner.layers.extend(lay for lay in p.layers if lay not in owner.layers)
            if not owner.net:
                owner.net = p.net
            del pads[key]
    # a pad drawn twice on the same layers (a region for its outline and a flash for its anchor) is one pad
    groups: dict[tuple, list] = {}
    for key, p in pads.items():
        if p.ref:
            groups.setdefault((p.ref, p.number), []).append(key)
    for keys in groups.values():
        for i, ki in enumerate(keys):
            a = pads.get(ki)
            if a is None:
                continue
            for kj in keys[i + 1:]:
                b = pads.get(kj)
                if b is not None and set(a.layers) == set(b.layers) and (a.contains([b.at], 1e-6)[0]
                                                                          or b.contains([a.at], 1e-6)[0]):
                    if b.size[0] * b.size[1] > a.size[0] * a.size[1]:
                        a.at, a.size, a.shape, a.angle_deg = b.at, b.size, b.shape, b.angle_deg
                    a.layers.extend(lay for lay in b.layers if lay not in a.layers)
                    del pads[kj]
    pad_list = list(pads.values())
    for p in pad_list:
        p.layers.sort(key=names.index)
        if len(p.layers) > 1 or tree is not None:
            p.drill = drill_at(p.at) if len(p.layers) > 1 else 0.0
    if not x2_functions:
        # without aperture functions a small round flash through several layers is taken for a via
        kept = []
        for p in pad_list:
            if (not p.ref and len(p.layers) > 1 and p.shape == "circle" and p.size[0] <= 1.0 * MM
                    and p.drill <= 0.6 * MM):
                via_list.append(Via(p.net, p.at, p.drill, p.size[0], (p.layers[0], p.layers[-1])))
            else:
                kept.append(p)
        if len(kept) < len(pad_list):
            notes.append(f"{len(pad_list) - len(kept)} small round flashes through several layers were taken for "
                         "vias, since the files do not mark vias.")
        pad_list = kept
    zones = [Zone(net, nm, polys) for (nm, net), polys in zones_raw.items()] + planes
    nets = sorted({t.net for t in tracks} | {p.net for p in pad_list} | {v.net for v in via_list}
                  | {z.net for z in zones} - {""})
    nets = [n for n in nets if n]
    if drills and not any(v.drill for v in via_list) and via_list:
        notes.append("Drill files did not line up with the vias, so via drill sizes are unknown (0).")
    elif via_list and not drills:
        notes.append("No drill file came with the Gerbers, so via drill sizes are unknown (0).")
    b = Board(name, "gerber", stack, tracks, via_list, pad_list, zones, nets, outline, notes)
    if not any_nets:
        from .graph import infer_nets
        infer_nets(b)
    return b


def _add_pad(pads, key, ob, layer, at, size, shape, angle):
    if ob.ref:
        # a named pad and a smaller one with the same name at the same spot (a via in a thermal pad) stay apart
        key = key + (round(min(size) / 1e-6), round(max(size) / 1e-6))
    p = pads.get(key)
    if p is None:
        pads[key] = Pad(ob.ref, ob.pin, ob.net, [layer], (float(at[0]), float(at[1])),
                        (float(size[0]), float(size[1])), shape, float(angle) % 360.0, 0.0)
    elif layer not in p.layers:
        p.layers.append(layer)
