"""Reader for KiCad .kicad_pcb files, KiCad 4 and 5 (module footprints, net table) through KiCad 10.

KiCad stores millimeters with y pointing down. The reader converts to meters
with y up. Pad positions in a footprint are local and turn with the footprint.
Pad angles are absolute in every KiCad version seen so far, and the reader
checks each file in case an old one stores them relative to the footprint.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from . import geom
from .model import MM, Board, Pad, Track, Via, Zone, build_stackup, default_stackup

_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"', re.S)
# parts of a board file the reader never looks at, dropped while parsing
_DROP = frozenset({"effects", "model", "uuid", "tstamp", "tedit", "embedded_files", "title_block", "pcbplotparams",
                   "fp_text_private", "gr_text", "fp_text_box", "gr_text_box", "dimension", "image", "teardrops",
                   "tenting", "capping", "covering", "plugging", "filling", "stroke", "font", "path", "sheetname",
                   "sheetfile", "attr", "descr", "tags"})
_ESCAPE = re.compile(r"\\(.)", re.S)
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}
_BRACES = {"dblquote": '"', "quote": "'", "lt": "<", "gt": ">", "backslash": "\\", "slash": "/", "bar": "|",
           "comma": ",", "colon": ":", "space": " ", "dollar": "$", "tab": "\t", "return": "\n", "brace": "{"}
_BRACE = re.compile(r"\{(\w+)\}")
_OLD_IDS = {0: "F.Cu", 31: "B.Cu", 44: "Edge.Cuts", **{k: f"In{k}.Cu" for k in range(1, 31)}}
_GRAPHICS = ("line", "arc", "circle", "rect", "poly", "curve")


def parse_sexpr(text: str, drop=frozenset()) -> list:
    """Parse s-expression text into nested lists of strings. Quoted strings lose their quotes.

    Lists whose head is in drop are thrown away as soon as they close, which keeps memory down
    on very large files.
    """
    root: list = []
    cur = root
    stack = []
    for i, seg in enumerate(_QUOTED.split(text)):
        if i & 1:
            cur.append(_ESCAPE.sub(lambda e: _ESCAPES.get(e.group(1), e.group(1)), seg) if "\\" in seg else seg)
            continue
        for tok in seg.replace("(", " ( ").replace(")", " ) ").split():
            if tok == "(":
                new: list = []
                cur.append(new)
                stack.append(cur)
                cur = new
            elif tok == ")":
                if not stack:
                    raise ValueError("there is a closing bracket without an opening one.")
                done = cur
                cur = stack.pop()
                if done and type(done[0]) is str and done[0] in drop:
                    cur.pop()
            else:
                cur.append(tok)
    if stack:
        raise ValueError(f"the file ends inside {len(stack)} open bracket(s), so it looks cut short.")
    return root


def unescape_name(name: str) -> str:
    """Undo KiCad's {slash} style escapes in net and pad names."""
    return _BRACE.sub(lambda m: _BRACES.get(m.group(1), m.group(0)), name) if "{" in name else name


def _kids(node) -> dict:
    out = {}
    for c in node[1:]:
        if type(c) is list and c:
            out.setdefault(c[0], c)
    return out


def _xy(node) -> tuple[float, float]:
    return float(node[1]), float(node[2])


def _first_num(node, default: float = 0.0) -> float:
    if node:
        for v in node[1:]:
            if type(v) is str:
                try:
                    return float(v)
                except ValueError:
                    continue
    return default


def _pts(node) -> np.ndarray:
    """Points of a (pts ...) list in file millimeters, arcs inside it expanded."""
    kids = [c for c in node[1:] if type(c) is list and c]
    if all(c[0] == "xy" for c in kids):
        return np.array([c[1:3] for c in kids], dtype=float).reshape(-1, 2)
    out = []
    for c in kids:
        if c[0] == "xy":
            out.append((float(c[1]), float(c[2])))
        elif c[0] == "arc":
            k = _kids(c)
            if "start" in k and "mid" in k and "end" in k:
                arc = geom.arc_through(_xy(k["start"]), _xy(k["mid"]), _xy(k["end"]))
                if out and math.hypot(out[-1][0] - arc[0][0], out[-1][1] - arc[0][1]) < 1e-9:
                    arc = arc[1:]
                out.extend(map(tuple, arc))
    return np.array(out, dtype=float).reshape(-1, 2)


def _rot(x, y, deg: float):
    """KiCad rotation in file coordinates (y down), counter clockwise on screen for positive angles."""
    if not deg:
        return x, y
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c + y * s, -x * s + y * c


def _to_m(pts) -> np.ndarray:
    """File millimeters with y down to meters with y up."""
    a = np.asarray(pts, dtype=float).reshape(-1, 2) * MM
    a[:, 1] *= -1.0
    return a


def _shape_points(node) -> tuple[np.ndarray, bool] | None:
    """Points of a graphic shape in file millimeters and whether it is closed."""
    kind = node[0].split("_", 1)[-1]
    k = _kids(node)
    try:
        if kind == "line":
            return np.array([_xy(k["start"]), _xy(k["end"])]), False
        if kind == "rect":
            (x0, y0), (x1, y1) = _xy(k["start"]), _xy(k["end"])
            return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), True
        if kind == "circle":
            c, e = _xy(k["center"]), _xy(k["end"])
            return geom.circle_ring(c, math.hypot(e[0] - c[0], e[1] - c[1])), True
        if kind == "poly":
            p = _pts(k["pts"])
            return p, len(p) > 2
        if kind == "curve":
            p = _pts(k["pts"])
            return (geom.bezier_points(*p[:4]) if len(p) >= 4 else p), False
        if kind == "arc":
            if "mid" in k:
                return geom.arc_through(_xy(k["start"]), _xy(k["mid"]), _xy(k["end"])), False
            # KiCad 5 and older keep the center in start, the first point in end and the sweep (clockwise on
            # screen) in angle
            c, p0 = np.array(_xy(k["start"])), np.array(_xy(k["end"]))
            ang = _first_num(k.get("angle"))
            n = max(1, int(math.ceil(abs(math.radians(ang)) / geom.ARC_STEP)))
            pts = [c + np.array(_rot(*(p0 - c), -ang * i / n)) for i in range(n + 1)]
            return np.array(pts), abs(ang) >= 359.999
    except (KeyError, IndexError, ValueError):
        return None
    return None


def kicad_major(version: int) -> int:
    """KiCad release that writes a given file format version."""
    if version < 20000000:
        return 4
    for limit, major in ((20200000, 5), (20211014, 6), (20221018, 7), (20240108, 8), (20241229, 9)):
        if version <= limit:
            return major
    return 10


class _Reader:
    def __init__(self, root: list, name: str):
        self.root = root
        self.name = name
        self.notes: list[str] = []
        self.version = 0
        self.layer_map: dict[str, str] = {}
        self.copper: list[str] = []
        self.net_table: dict[str, str] = {}
        self.thickness = None
        self.tracks: list[Track] = []
        self.vias: list[Via] = []
        self.pads: list[Pad] = []
        self.zones: list[Zone] = []
        self.edge_open: list[np.ndarray] = []
        self.edge_rings: list[np.ndarray] = []
        self.nets_seen: dict[str, None] = {}
        self.skipped = {"copper graphics": 0, "np_thru_hole pads": 0, "teardrops": 0, "unreadable": 0}

    def safe(self, fn, *args):
        """Run one item reader, counting the item as unreadable instead of failing the whole board."""
        try:
            return fn(*args)
        except (KeyError, IndexError, ValueError, TypeError):
            self.skipped["unreadable"] += 1
            return True

    # layers and nets
    def canon(self, name: str) -> str:
        return self.layer_map.get(name, name)

    def read_layers(self, node) -> None:
        old = self.version < 20200000
        for entry in node[1:]:
            if type(entry) is not list or len(entry) < 2:
                continue
            try:
                lid = int(entry[0])
            except ValueError:
                continue
            name = entry[1]
            canon = _OLD_IDS.get(lid, name) if old else name
            self.layer_map[name] = canon
            self.layer_map[canon] = canon
            if canon.endswith(".Cu") and canon not in self.copper:
                self.copper.append(canon)

        def order(n: str) -> int:
            if n == "F.Cu":
                return 0
            if n == "B.Cu":
                return 1000
            m = re.match(r"In(\d+)\.Cu", n)
            return int(m.group(1)) if m else 500

        self.copper.sort(key=order)

    def copper_of(self, names) -> list[str]:
        out = []
        for n in names:
            if type(n) is not str:
                continue
            if n == "*.Cu":
                group = self.copper
            elif n == "F&B.Cu":
                group = [c for c in self.copper if c in ("F.Cu", "B.Cu")]
            else:
                c = self.canon(n)
                group = [c] if c in self.copper else []
            out.extend(g for g in group if g not in out)
        return sorted(out, key=self.copper.index)

    def net(self, node) -> str:
        if not node or len(node) < 2:
            return ""
        if len(node) >= 3 and type(node[2]) is str:
            name = node[2]
        else:
            tok = node[1]
            name = self.net_table.get(tok, tok if not (self.net_table and tok.isdigit()) else "")
            if tok == "0" and self.net_table:
                name = ""
        name = unescape_name(name)
        if name:
            self.nets_seen.setdefault(name, None)
        return name

    # stackup
    def read_stackup(self, node):
        sheets = []
        for lay in node[1:]:
            if type(lay) is not list or len(lay) < 2 or lay[0] != "layer":
                continue
            name = self.canon(lay[1])
            k = _kids(lay)
            kind = k["type"][1] if "type" in k else ""
            if name in self.copper or kind == "copper":
                sheets.append(("copper", name, _first_num(k.get("thickness"), 0.035) * MM))
            elif kind in ("core", "prepreg") or lay[1].startswith("dielectric"):
                groups: list[list] = [[]]
                for c in lay[2:]:
                    if c == "addsublayer":
                        groups.append([])
                    elif type(c) is list and c:
                        groups[-1].append(c)
                for i, g in enumerate(groups):
                    d = {c[0]: c for c in g}
                    t = _first_num(d.get("thickness"), 0.0) * MM
                    if t <= 0:
                        continue
                    mat = d["material"][1] if "material" in d else (kind or "FR4")
                    sheets.append(("dielectric", lay[1] if len(groups) == 1 else f"{lay[1]}.{i + 1}", t,
                                   _first_num(d.get("epsilon_r"), 4.5), _first_num(d.get("loss_tangent"), 0.02),
                                   mat))
        cu = [s[1] for s in sheets if s[0] == "copper"]
        if cu != self.copper or not any(s[0] == "dielectric" for s in sheets):
            if sheets:
                self.notes.append("The stackup in the file does not match its copper layers, so a default "
                                  "stackup was used.")
            return None
        return build_stackup(sheets, "kicad stackup")

    # items
    def read_segment(self, node):
        k = _kids(node)
        layer = self.canon(k["layer"][1]) if "layer" in k else ""
        if layer not in self.copper:
            return
        net = self.net(k.get("net"))
        width = _first_num(k.get("width"), 0.2) * MM
        s, e = k["start"], k["end"]
        if node[0] == "arc" and "mid" in k:
            pts = _to_m(geom.arc_through(_xy(s), _xy(k["mid"]), _xy(e))).tolist()
        else:
            pts = [(float(s[1]) * MM, -float(s[2]) * MM), (float(e[1]) * MM, -float(e[2]) * MM)]
        for a, b in zip(pts[:-1], pts[1:]):
            self.tracks.append(Track(net, layer, (a[0], a[1]), (b[0], b[1]), width))

    def read_via(self, node):
        k = _kids(node)
        at = _to_m([_xy(k["at"])])[0]
        names = self.copper_of(k["layers"][1:]) if "layers" in k else []
        if not names:
            names = [self.copper[0], self.copper[-1]] if self.copper else []
        if not names:
            return
        drill = _first_num(k.get("drill"), 0.0) * MM
        size = _first_num(k.get("size"), 0.0) * MM or 2 * drill
        self.vias.append(Via(self.net(k.get("net")), (float(at[0]), float(at[1])), drill, size,
                             (names[0], names[-1])))

    def read_zone(self, node):
        k = _kids(node)
        if "name" in k and len(k["name"]) > 1 and k["name"][1].startswith("$teardrop"):
            self.skipped["teardrops"] += 1
            return
        if "net_name" in k and len(k["net_name"]) > 1:
            net = unescape_name(k["net_name"][1])
            if net:
                self.nets_seen.setdefault(net, None)
        else:
            net = self.net(k.get("net"))
        default_layers = self.copper_of(k["layers"][1:]) if "layers" in k else []
        if "layer" in k:
            default_layers = self.copper_of([k["layer"][1]])
        stroke = 0.0
        thick = "filled_areas_thickness" in k and k["filled_areas_thickness"][1] == "yes"
        if self.version < 20200000 or thick:
            stroke = _first_num(k.get("min_thickness"), 0.0) * MM
        by_layer: dict[str, list] = {}
        for c in node[1:]:
            if type(c) is not list or not c or c[0] != "filled_polygon":
                continue
            ck = _kids(c)
            layer = self.copper_of([ck["layer"][1]]) if "layer" in ck else default_layers[:1]
            if not layer or "pts" not in ck:
                continue
            ring = _to_m(_pts(ck["pts"]))
            if len(ring) >= 3:
                by_layer.setdefault(layer[0], []).append((ring, []))
        for layer, polys in by_layer.items():
            self.zones.append(Zone(net, layer, polys, stroke))

    def read_edge(self, node, transform=None):
        k = _kids(node)
        if "layer" not in k or self.canon(k["layer"][1]) != "Edge.Cuts":
            return False
        got = _shape_points(node)
        if got is None:
            return True
        pts, closed = got
        if transform is not None:
            pts = transform(pts)
        pts = _to_m(pts)
        (self.edge_rings if closed else self.edge_open).append(pts)
        return True

    def read_footprints(self, fps):
        raw = []
        for fp in fps:
            k = _kids(fp)
            at = k.get("at", ["at", "0", "0"])
            try:
                fx, fy = float(at[1]), float(at[2])
                fa = float(at[3]) if len(at) > 3 and type(at[3]) is str else 0.0
            except (ValueError, IndexError, TypeError):
                self.skipped["unreadable"] += 1
                continue
            ref = ""
            for c in fp[1:]:
                if type(c) is not list or len(c) < 3:
                    continue
                if (c[0] == "fp_text" and c[1] == "reference") or (c[0] == "property" and c[1] == "Reference"):
                    ref = unescape_name(c[2])
                    break
            pads = [c for c in fp[1:] if type(c) is list and c and c[0] == "pad"]
            raw.append((fp, fx, fy, fa, ref, pads))
        # decide whether pad angles are absolute (KiCad writes them so) or relative to the footprint
        absolute = relative = 0
        for _, _, _, fa, _, pads in raw:
            if fa % 360 == 0 or not pads:
                continue
            angs = []
            for p in pads:
                pat = _kids(p).get("at", [])
                angs.append(_first_num(["at"] + pat[3:4], 0.0))
            if all(abs((a - fa + 180) % 360 - 180) < 1e-6 for a in angs):
                absolute += 1
            elif all(a == 0 for a in angs):
                relative += 1
        rel = relative > absolute
        if rel:
            self.notes.append("Pad angles in this file are relative to their footprints, so the footprint angle "
                              "was added to each.")

        for fp, fx, fy, fa, ref, pads in raw:
            def transform(pts, fx=fx, fy=fy, fa=fa):
                x, y = _rot(pts[:, 0], pts[:, 1], fa)
                return np.column_stack([fx + x, fy + y])

            for c in fp[1:]:
                if type(c) is list and c and type(c[0]) is str and c[0].startswith("fp_") and c[0][3:] in _GRAPHICS:
                    self.safe(self.read_edge, c, transform)
                elif type(c) is list and c and c[0] == "zone":
                    self.safe(self.read_zone, c)
            for p in pads:
                self.safe(self.read_pad, p, fx, fy, fa, ref, rel)

    def read_pad(self, p, fx, fy, fa, ref, rel):
        if len(p) < 4:
            return
        number, ptype, shape = unescape_name(str(p[1])), p[2], p[3]
        k = _kids(p)
        if ptype == "np_thru_hole":
            self.skipped["np_thru_hole pads"] += 1
            return
        layers = self.copper_of(k["layers"][1:]) if "layers" in k else []
        if not layers:
            return
        at = k.get("at", ["at", "0", "0"])
        lx, ly = float(at[1]), float(at[2])
        pa = float(at[3]) if len(at) > 3 and type(at[3]) is str else 0.0
        if rel:
            pa += fa
        dx, dy = _rot(lx, ly, fa)
        ax, ay = fx + dx, fy + dy
        drill_node = k.get("drill")
        drill = 0.0
        if drill_node:
            nums = [float(v) for v in drill_node[1:] if type(v) is str and v != "oval"]
            drill = min(nums) * MM if nums else 0.0
            off = next((c for c in drill_node[1:] if type(c) is list and c and c[0] == "offset"), None)
            if off:
                ox, oy = _rot(float(off[1]), float(off[2]), pa)
                ax, ay = ax + ox, ay + oy
        size = k.get("size", ["size", "0", "0"])
        w = float(size[1])
        h = float(size[2]) if len(size) > 2 else w
        if shape == "custom" and "primitives" in k:
            w, h = self.custom_extent(k["primitives"], w, h)
        self.pads.append(Pad(ref, number, self.net(k.get("net")), layers, (ax * MM, -ay * MM), (w * MM, h * MM),
                             shape, pa % 360.0, drill))

    def custom_extent(self, prims, w, h):
        """Half extents of a custom pad's primitives, kept centred on the pad."""
        ex, ey = w / 2.0, h / 2.0
        for c in prims[1:]:
            if type(c) is not list or not c or not c[0].startswith("gr_"):
                continue
            got = _shape_points(c)
            if got is None:
                continue
            pts = got[0]
            half = _first_num(_kids(c).get("width"), 0.0) / 2.0
            if len(pts):
                ex = max(ex, float(np.abs(pts[:, 0]).max()) + half)
                ey = max(ey, float(np.abs(pts[:, 1]).max()) + half)
        return 2 * ex, 2 * ey

    def outline(self) -> list[np.ndarray]:
        rings, left = geom.chain_rings(self.edge_open, 1e-5)
        if left:
            more, left = geom.chain_rings(left, 2e-4)
            rings += more
        for piece in left:
            span = float(np.ptp(piece, axis=0).max()) if len(piece) > 1 else 0.0
            if len(piece) > 2 and span > 0 and math.hypot(*(piece[-1] - piece[0])) < 0.05 * span:
                rings.append(piece)
            else:
                self.notes.append(f"Part of the board edge ({len(piece)} points) does not close and was left out.")
        rings = self.edge_rings + rings
        rings.sort(key=geom.ring_area, reverse=True)
        return rings

    def board(self) -> Board:
        groups: dict[str, list] = {}
        for c in self.root[1:]:
            if type(c) is list and c:
                groups.setdefault(c[0], []).append(c)
        if "version" in groups:
            self.version = int(_first_num(groups["version"][0], 0))
        major = kicad_major(self.version)
        self.notes.append(f"KiCad {major} board, file format version {self.version}.")
        if "layers" in groups:
            self.read_layers(groups["layers"][0])
        if not self.copper:
            self.copper = ["F.Cu", "B.Cu"]
            self.layer_map.update({"F.Cu": "F.Cu", "B.Cu": "B.Cu", "Edge.Cuts": "Edge.Cuts"})
            self.notes.append("The file has no layer table, so two copper layers were assumed.")
        for g in groups.get("general", []):
            t = _kids(g).get("thickness")
            if t:
                self.thickness = _first_num(t) * MM
        for n in groups.get("net", []):
            if len(n) >= 3:
                self.net_table[n[1]] = n[2]
        stack = None
        for s in groups.get("setup", []):
            st = _kids(s).get("stackup")
            if st:
                stack = self.read_stackup(st)
        if stack is None:
            stack = default_stackup(self.copper, self.thickness)
            self.notes.append(f"No usable stackup in the file. Using a {stack.source}.")
        for kind in ("segment", "arc"):
            for c in groups.get(kind, []):
                self.safe(self.read_segment, c)
        for c in groups.get("via", []):
            self.safe(self.read_via, c)
        self.read_footprints(groups.get("footprint", []) + groups.get("module", []))
        for c in groups.get("zone", []):
            self.safe(self.read_zone, c)
        for g in _GRAPHICS:
            for c in groups.get("gr_" + g, []):
                if not self.safe(self.read_edge, c):
                    k = _kids(c)
                    if "layer" in k and self.canon(k["layer"][1]) in self.copper:
                        self.skipped["copper graphics"] += 1
        if self.skipped["unreadable"]:
            self.notes.append(f"{self.skipped['unreadable']} items could not be read and were left out.")
        if self.skipped["copper graphics"]:
            self.notes.append(f"{self.skipped['copper graphics']} copper drawings without a net were left out.")
        if self.skipped["teardrops"]:
            self.notes.append(f"{self.skipped['teardrops']} teardrop fillets were left out of the zones.")
        if self.skipped["np_thru_hole pads"]:
            self.notes.append(f"{self.skipped['np_thru_hole pads']} non plated holes were left out.")
        table = [unescape_name(v) for k, v in sorted(self.net_table.items(), key=lambda kv: _int(kv[0]))]
        nets = [n for n in table if n]
        known = set(nets)
        nets += sorted(n for n in self.nets_seen if n not in known)
        b = Board(self.name, "kicad", stack, self.tracks, self.vias, self.pads, self.zones, nets, self.outline(),
                  self.notes)
        return b


def _int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 1 << 30


def read_kicad_pcb(path) -> Board:
    """Read a KiCad .kicad_pcb file into a Board."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"Could not open {p} ({e.strerror}). Check the path and that the file is readable.") from None
    if not text.lstrip().startswith("(kicad_pcb"):
        raise ValueError(f"{p.name} is not a KiCad board file because it does not start with '(kicad_pcb'. Save the "
                         "board from KiCad's PCB editor, or pass the folder of Gerber files instead.")
    try:
        tree = parse_sexpr(text, _DROP)
    except ValueError as e:
        raise ValueError(f"{p.name} could not be parsed because {e} Open and save it in KiCad, then try "
                         "again.") from None
    if not tree or type(tree[0]) is not list:
        raise ValueError(f"{p.name} holds no board. Save the board from KiCad's PCB editor and try again.")
    return _Reader(tree[0], p.stem).board()
