"""Bench parts for the circuit drawings: a small SVG writer and the things on a bench, seen from above.

Breadboards, banded resistors, LEDs, capacitors, instruments with their displays, leads, scope probes, a
wound toroid, a TO-220 on its heatsink. Sizes are to scale where the eye checks them: hole pitch,
resistor bodies, LED domes, the TO-220's tab. Text names IBM Plex Mono and falls back to the system's
monospace, and every box is sized for the 0.6 em advance they all share, so nothing needs a font file.

A drawing is a still of the run's final state, so it reads anywhere (report.html, GitHub, an image
viewer). Parts that change while the circuit runs carry data-* attributes, and the website plays them
from power on to steady state:

    data-live="2"         on <svg>: the seconds from power on to steady state
    data-series="a b c"   values over that ramp, written into the text (data-fmt="decimals unit")
                          or into the attribute named by data-attr
    data-flow="80"        dots moving along a wire at 80 units per second, scaled by data-k over the
                          ramp and gated by data-pwm
    data-pwm="d T"        on for the first d of every T seconds ("d T inverted" for the rest)
    data-wave="a b c"     an opacity following these values once every data-period seconds
    data-sweep="3"        a trace drawn in over 3 seconds, again and again ("ramp": once)
    data-heat="#a #b"     a fill warming from one color to the other over the ramp
    data-after            hidden until steady state, such as the verdict
    data-drift="1.6"      heat shimmer rising, one cycle every 1.6 seconds
"""
from __future__ import annotations

import html
import math

from ..units import fmt

W, H = 640, 480
BG, GRID = "#0e1011", "#15181b"
FG, FG2, MUTED, LINE2 = "#f2f2ee", "#c9ccc5", "#9ca099", "#3a3e45"
YEL, INK = "#ffd400", "#0e0f11"


VERDICT = {"ok": "#46d17f", "warm": "#ffb020", "marginal": "#ffb020", "fails": "#ff5a4a", "hot": "#ff5a4a"}
RED_LEAD, BLACK_LEAD = "#c8352b", "#2b2e33"


FONT = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# resistor color code: digits 0 to 9, then gold and silver


BAND = ["#1b1b1b", "#7a4420", "#d0312d", "#f07d1a", "#f2cf1d", "#2e9e4c", "#2f62cf", "#8443c9", "#8d8d8d", "#f4f4f4"]
GOLD, SILVER = "#c9a645", "#c0c4c9"


LED_COLOR = {"red": "#ff3b2f", "orange": "#ff8a1f", "yellow": "#ffd21f", "green": "#36d26a", "blue": "#3d7bff",
             "white": "#eef3ff", "uv": "#9b5cff", "ir": "#a33a5a", "power-white": "#eef3ff", "power-red": "#ff3b2f"}


def num(v) -> str:
    s = f"{float(v):.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _attrs(a: dict) -> str:
    out = []
    for k, v in a.items():
        if v is None or v is False:
            continue
        k = "class" if k == "cls" else k.replace("_", "-")
        if v is True:
            v = ""
        out.append(f' {k}="{num(v) if isinstance(v, (int, float)) else html.escape(str(v))}"')
    return "".join(out)


def mix(a: str, b: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    ca, cb = (int(a[i:i + 2], 16) for i in (1, 3, 5)), (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def nums(vals, digits: int = 4) -> str:
    return " ".join(f"{float(v):.{digits}g}" for v in vals)


def cubic(p0, p1, p2, p3) -> str:
    return "M{} {}C{} {} {} {} {} {}".format(*map(num, (*p0, *p1, *p2, *p3)))


def poly(pts) -> str:
    return "M" + "L".join(f"{num(x)} {num(y)}" for x, y in pts)


def _sig(x: float) -> int:
    """Decimals that show x to three significant digits, the way Wobbl prints numbers."""
    return max(0, 2 - math.floor(math.log10(abs(x)))) if x else 0


class Svg:
    """A small SVG writer. Ids carry a namespace so two drawings can share one page."""

    def __init__(self, ns: str, w: int = W, h: int = H):
        self.ns, self.w, self.h = ns, w, h
        self.defs: list[str] = []
        self.body: list[str] = []
        self.k = 0

    def uid(self, name: str) -> str:
        self.k += 1
        return f"{self.ns}-{name}{self.k}"

    def add(self, tag: str, text: str | None = None, **a):
        o = f"<{tag}{_attrs(a)}"
        self.body.append(f"{o}/>" if text is None else f"{o}>{text}</{tag}>")

    def open(self, **a):
        self.body.append(f"<g{_attrs(a)}>")

    def close(self):
        self.body.append("</g>")

    def text(self, x, y, s, size=12, fill=FG2, anchor=None, weight=None, ls=None, **a):
        self.add("text", html.escape(s), x=x, y=y, font_size=size, fill=fill, text_anchor=anchor, font_weight=weight,
                 letter_spacing=ls, **a)

    def grad(self, kind: str, stops, **a) -> str:
        gid = self.uid(kind[0])
        st = "".join(f'<stop offset="{num(o)}" stop-color="{c}"' + (f' stop-opacity="{op:.2g}"' if op < 1 else "")
                     + "/>" for o, c, op in stops)
        self.defs.append(f'<{kind}Gradient id="{gid}"{_attrs(a)}>{st}</{kind}Gradient>')
        return f"url(#{gid})"

    def svg(self, label: str, live: float | None = None) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{html.escape(label)}" font-family="{FONT}"'
                + (f' data-live="{live:g}"' if live else "") + ">")
        return "\n".join([head, f"<title>{html.escape(label)}</title>", "<defs>", *self.defs, "</defs>", *self.body,
                          "</svg>"]) + "\n"


# ---- bench furniture -------------------------------------------------------------------


def mat(s: Svg):
    """The bench: near black with a faint mat grid."""
    gid = s.uid("grid")
    s.defs.append(f'<pattern id="{gid}" width="20" height="20" patternUnits="userSpaceOnUse">'
                  f'<path d="M20 0H0V20" fill="none" stroke="{GRID}"/></pattern>')
    s.add("rect", width=s.w, height=s.h, fill=BG)
    s.add("rect", width=s.w, height=s.h, fill=f"url(#{gid})")


def knob(s: Svg, x, y, turn=40):
    s.add("circle", cx=x, cy=y, r=11, fill="#2b2f35", stroke="#4b5159")
    s.add("circle", cx=x, cy=y, r=7.5, fill="#353a40")
    a = math.radians(turn - 90)
    s.add("path", d=f"M{num(x)} {num(y)}l{num(9 * math.cos(a))} {num(9 * math.sin(a))}", stroke=FG2, stroke_width=1.6,
          stroke_linecap="round")


def post(s: Svg, x, y, color):
    s.add("circle", cx=x, cy=y, r=8.5, fill=color, stroke=mix(color, "#000000", 0.45))
    s.add("circle", cx=x, cy=y, r=4.2, fill=mix(color, "#000000", 0.3))
    s.add("circle", cx=x, cy=y, r=1.6, fill="#0b0b0b")


def instrument(s: Svg, x, y, w, h, name, rows, lamp="CV", posts=("-", "+")) -> dict:
    """A bench instrument's front panel: its display rows (value, unit[, series, format]), two knobs and posts.

    posts are named left to right; "+" is red and anything else black. Returns where each one is.
    """
    s.add("rect", x=x + 3, y=y + 5, width=w, height=h, rx=7, fill="#000", opacity=0.5)
    s.add("rect", x=x, y=y, width=w, height=h, rx=7, fill="#1b1e22", stroke="#3b4047")
    s.add("rect", x=x + 1, y=y + 1, width=w - 2, height=7, rx=6, fill="#24282d")
    s.text(x + 12, y + 21, name, size=8.5, fill=MUTED, ls=1.2)
    dx, dy, dw, dh = x + 12, y + 29, w - 66, 14 + 26 * len(rows)
    s.add("rect", x=dx, y=dy, width=dw, height=dh, rx=3, fill="#07080a", stroke="#30353b")
    for i, (val, unit, *live) in enumerate(rows):
        by = dy + 30 + 26 * i
        a = {"data_series": live[0], "data_fmt": live[1]} if live else {}
        s.text(dx + dw - 26, by, val, size=20, fill=YEL, anchor="end", weight=600, **a)
        s.text(dx + dw - 21, by, unit, size=10, fill=MUTED)
    knob(s, x + w - 29, y + 36, 30)
    knob(s, x + w - 29, y + 64, 130)
    if lamp:
        s.add("circle", cx=dx + 4, cy=y + h - 14, r=3, fill="#46d17f")
        s.text(dx + 11, y + h - 10.5, lamp, size=9, fill=MUTED)
    at = {}
    for i, name in enumerate(posts):
        px = x + w - 24 - 28 * (len(posts) - 1 - i)
        at[name] = (px, y + h - 15)
        post(s, px, y + h - 15, "#c8342b" if name == "+" else "#2a2d31")
    return at


def lead(s: Svg, d: str, color: str, w: float = 5):
    """A flexible insulated lead: a darker edge, the insulation and a soft highlight."""
    edge = "#5d636b" if color == BLACK_LEAD else mix(color, "#000000", 0.4)
    s.add("path", d=d, fill="none", stroke=edge, stroke_width=w + 2, stroke_linecap="round", stroke_linejoin="round")
    s.add("path", d=d, fill="none", stroke=color, stroke_width=w, stroke_linecap="round", stroke_linejoin="round")
    s.add("path", d=d, fill="none", stroke="#fff", stroke_opacity=0.25, stroke_width=max(1.0, w * 0.3),
          stroke_linecap="round", stroke_linejoin="round")


def plug(s: Svg, x, y, color):
    s.add("circle", cx=x, cy=y, r=6.5, fill=color, stroke=mix(color, "#000000", 0.5) if color != BLACK_LEAD else
          "#5d636b")


def pin(s: Svg, x, y, ang):
    """A Dupont male pin pushed into a breadboard hole, its housing along the lead."""
    s.open(transform=f"translate({num(x)} {num(y)}) rotate({num(ang)})")
    s.add("rect", x=-17, y=-3.8, width=14, height=7.6, rx=1, fill="#1c1e21", stroke="#4a4f57", stroke_width=0.8)
    s.add("rect", x=-3, y=-1, width=4, height=2, fill="#b8bdc3")
    s.close()


def flow(s: Svg, d: str, speed: float, **a):
    """Moving charge: dots along a conduction path, dark ringed so they read on the light board too."""
    s.open(fill="none", stroke_linecap="round", stroke_dasharray="0 12", data_flow=f"{speed:.3g}", **a)
    s.add("path", d=d, stroke=INK, stroke_width=5.5, stroke_opacity=0.6)
    s.add("path", d=d, stroke=YEL, stroke_width=3)
    s.close()


def readout(s: Svg, label: str, value: float, unit: str, series, limit: float, note: str, verdict: str):
    """The one live number: a value, a bar against its limit and the verdict, top right."""
    x, y, w, h = W - 216, 18, 196, 100
    s.add("rect", x=x, y=y, width=w, height=h, rx=3, fill="#0a0b0c", fill_opacity=0.92, stroke=LINE2)
    s.text(x + 14, y + 22, label, size=10.5, fill=MUTED, ls=1.3, weight=500)
    d = _sig(value)
    s.text(x + 14, y + 55, f"{value:.{d}f} {unit}", size=26, fill=FG, weight=600, data_series=nums(series),
           data_fmt=f"{d} {unit}")
    word = verdict.upper()
    cw = len(word) * 6.6 + 14
    s.add("rect", x=x + w - 14 - cw, y=y + 9, width=cw, height=19, fill=VERDICT.get(verdict, YEL), data_after=True)
    s.text(x + w - 14 - cw / 2, y + 22.5, word, size=11, fill=INK, anchor="middle", weight=700, data_after=True)
    gx, gy, gw = x + 14, y + 68, w - 28
    top = max(limit, max(series)) * 1.2
    s.add("rect", x=gx, y=gy, width=gw, height=6, fill="#23272c")
    lim_x = gw * limit / top
    s.add("rect", x=gx, y=gy, width=gw * min(value, limit) / top, height=6, fill=YEL,
          data_series=nums(gw * min(v, limit) / top for v in series), data_attr="width")
    if max(series) > limit:
        s.add("rect", x=gx + lim_x, y=gy, width=gw * max(value - limit, 0) / top, height=6, fill=VERDICT["fails"],
              data_series=nums(gw * max(v - limit, 0) / top for v in series), data_attr="width")
    s.add("path", d=f"M{num(gx + lim_x)} {num(gy - 4)}v14", stroke=FG, stroke_width=1.5)
    s.text(gx + lim_x, y + 92, note, size=10, fill=MUTED, anchor="end" if lim_x > gw * 0.55 else "start")


def shimmer(s: Svg, x, y, color, n=3, gap=9.0):
    """Heat rising off a part: thin wavy lines, drifting upward on the site."""
    s.open(stroke=color, stroke_width=1.6, fill="none", stroke_linecap="round", opacity=0.85, data_after=True,
           data_drift="1.6")
    for i in range(n):
        x0 = x + (i - (n - 1) / 2) * gap
        s.add("path", d=f"M{num(x0)} {num(y)}c4 -4 -4 -8 0 -12s-4 -8 0 -12")
    s.close()


# ---- breadboard parts -----------------------------------------------------------------


class Breadboard:
    """A half size solderless breadboard: 30 columns, rows a to j, and a pair of rails at each edge."""

    ROWS = {"t+": 1.0, "t-": 2.0, **{r: 3.6 + i for i, r in enumerate("abcde")},
            **{r: 10.6 + i for i, r in enumerate("fghij")}, "b-": 16.2, "b+": 17.2}

    def __init__(self, s: Svg, x: float, y: float, p: float = 16.5, cols: int = 24):
        self.x, self.y, self.p, self.cols = x, y, p, cols
        self.w, self.h = (cols - 1) * p + 4.4 * p, 18.2 * p
        self.x0 = x + 2.2 * p
        w, h = self.w, self.h
        s.add("rect", x=x + 4, y=y + 6, width=w, height=h, rx=6, fill="#000", opacity=0.55)
        s.add("rect", x=x, y=y, width=w, height=h, rx=6, fill="#e4e0d5", stroke="#c3bdae")
        for fy in (0.3, 0.62):  # the dovetails that join boards side by side
            s.add("rect", x=x - 1, y=y + fy * h, width=4, height=0.9 * p, fill=BG)
        s.add("rect", x=x + 0.6 * p, y=y + 8.65 * p, width=w - 1.2 * p, height=0.9 * p, fill="#d6d1c5")
        for fy, col in ((0.42, "#d0392e"), (2.58, "#2f62bd"), (15.62, "#2f62bd"), (17.78, "#d0392e")):
            s.add("path", d=f"M{num(x + 1.3 * p)} {num(y + fy * p)}H{num(x + w - 1.3 * p)}", stroke=col,
                  stroke_width=1.6)
        hs = 0.3 * p
        hole = f'<rect x="{{}}" y="{num((p - hs) / 2)}" width="{num(hs)}" height="{num(hs)}" rx="0.8" fill="#77736a"/>'
        holes = {"main": (1, hole.format(num((p - hs) / 2)), 3.1),
                 "rail": (6, "".join(hole.format(num((p - hs) / 2 + i * p)) for i in range(5)), 0.5),
                 "rail2": (6, "".join(hole.format(num((p - hs) / 2 + i * p)) for i in range(5)), 15.7)}
        fills = {}
        for name, (span, cells, top) in holes.items():  # one pattern per block, so holes land on the row centers
            fills[name] = s.uid(name)
            s.defs.append(f'<pattern id="{fills[name]}" width="{num(span * p)}" height="{num(p)}" '
                          f'patternUnits="userSpaceOnUse" x="{num(self.x0 - p / 2)}" y="{num(y + top * p)}">{cells}'
                          f'</pattern>')
        for name, top, rows in (("main", 3.1, 5), ("main", 10.1, 5), ("rail", 0.5, 2), ("rail2", 15.7, 2)):
            s.add("rect", x=self.x0 - p / 2, y=y + top * p, width=cols * p, height=rows * p,
                  fill=f"url(#{fills[name]})")
        s.open(font_size=7, fill="#9a9586", text_anchor="middle")
        for r in "abcdefghij":
            for lx in (x + 1.15 * p, x + w - 1.15 * p):
                s.add("text", r, x=lx, y=y + self.ROWS[r] * p + 2.5)
        for c in [1, *range(5, cols + 1, 5)]:
            s.add("text", str(c), x=self.x0 + (c - 1) * p, y=y + 3.12 * p)
        s.close()

    def hole(self, col: int, row: str) -> tuple[float, float]:
        return self.x0 + (col - 1) * self.p, self.y + self.ROWS[row] * self.p


def bands(ohms: float) -> list[str] | None:
    """Color bands for a resistance: four bands at 5% when two digits are exact, else five at 1%."""
    if not ohms > 0:
        return None
    for n in (2, 3):
        e = math.floor(math.log10(ohms)) - (n - 1)
        sig = round(ohms / 10 ** e)
        if sig >= 10 ** n:
            sig, e = sig // 10, e + 1
        if abs(sig * 10.0 ** e - ohms) <= 1e-3 * ohms or n == 3:
            if not -2 <= e <= 9:
                return None
            mult = SILVER if e == -2 else GOLD if e == -1 else BAND[e]
            return [BAND[int(c)] for c in str(sig)] + [mult, GOLD if n == 2 else BAND[1]]
    return None


def _dogbone(cx, bl, bh) -> str:
    """A through hole resistor's outline: round end caps and a slimmer middle."""
    r, m, x0, x1 = bh / 2, 0.4 * bh, cx - bl / 2, cx + bl / 2
    c1, n1, c2, n2, k = x0 + 0.27 * bl, x0 + 0.36 * bl, x1 - 0.27 * bl, x1 - 0.36 * bl, 0.04 * bl
    f = num
    top = (f"M{f(x0 + r)} {f(-r)}H{f(c1)}C{f(c1 + k)} {f(-r)} {f(n1 - k)} {f(-m)} {f(n1)} {f(-m)}H{f(n2)}"
           f"C{f(n2 + k)} {f(-m)} {f(c2 - k)} {f(-r)} {f(c2)} {f(-r)}H{f(x1 - r)}"
           f"A{f(r)} {f(r)} 0 0 1 {f(x1 - r)} {f(r)}")
    bottom = (f"H{f(c2)}C{f(c2 - k)} {f(r)} {f(n2 + k)} {f(m)} {f(n2)} {f(m)}H{f(n1)}"
              f"C{f(n1 - k)} {f(m)} {f(c1 + k)} {f(r)} {f(c1)} {f(r)}H{f(x0 + r)}"
              f"A{f(r)} {f(r)} 0 0 1 {f(x0 + r)} {f(-r)}Z")
    return top + bottom


def resistor(s: Svg, a, b, ohms: float, p: float):
    """An axial resistor lying between holes a and b, banded with its value."""
    (ax, ay), (bx, by) = a, b
    length, ang = math.hypot(bx - ax, by - ay), math.degrees(math.atan2(by - ay, bx - ax))
    bl, bh, cx = 2.6 * p, 0.95 * p, length / 2
    code = bands(ohms)
    s.open(transform=f"translate({num(ax)} {num(ay)}) rotate({num(ang)})")
    s.add("path", d=f"M0 0H{num(length)}", stroke="#80858c", stroke_width=2.8, stroke_linecap="round")
    s.add("path", d=f"M0 0H{num(length)}", stroke="#d7dade", stroke_width=1.4, stroke_linecap="round")
    body = _dogbone(cx, bl, bh)
    cid = s.uid("rb")
    s.defs.append(f'<clipPath id="{cid}"><path d="{body}"/></clipPath>')
    s.add("path", d=body, fill="#7aa6d6" if code and len(code) == 5 else "#d9bf8f")
    if code:
        at = [0.2, 0.36, 0.48, 0.6, 0.8] if len(code) == 5 else [0.2, 0.38, 0.5, 0.78]
        s.open(clip_path=f"url(#{cid})")
        for f, c in zip(at, code):
            s.add("rect", x=cx - bl / 2 + f * bl - 0.035 * bl, y=-bh, width=0.075 * bl, height=2 * bh, fill=c)
        s.close()
    s.add("path", d=body, fill=s.grad("linear", [(0, "#ffffff", 0.45), (0.35, "#ffffff", 0), (0.6, "#000000", 0),
                                                  (1, "#000000", 0.35)], x2="0", y2="1"))
    s.add("path", d=body, fill="none", stroke="#000", stroke_opacity=0.35, stroke_width=0.8)
    s.close()


def led(s: Svg, an, ca, color: str, p: float, glow: list[float], size: float):
    """A 5 mm LED seen from above: the flange flat on the cathode side, the dome lit by its current.

    glow is the brightness from 0 to 1 over the ramp, ending at steady state; size grows the halo.
    """
    (ax, ay), (kx, ky) = an, ca
    cx, cy = (ax + kx) / 2, (ay + ky) / 2
    ang = math.degrees(math.atan2(ky - ay, kx - ax))
    c = LED_COLOR.get(color, LED_COLOR["red"])
    rd, rf = 0.98 * p, 1.14 * p
    rg = rd * (2.0 + 2.4 * min(size, 1.5))
    s.add("circle", cx=cx, cy=cy, r=rg, fill=s.grad("radial", [(0, c, 0.95), (0.3, c, 0.42), (1, c, 0)]),
          opacity=glow[-1], data_series=nums(glow), data_attr="opacity")
    s.open(transform=f"translate({num(cx)} {num(cy)}) rotate({num(ang)})")
    for x0, x1 in ((-rf - 0.6 * p, -rf + 2), (rf - 2, rf + 0.6 * p)):  # the legs, bent out to their holes
        s.add("path", d=f"M{num(x0)} 0H{num(x1)}", stroke="#80858c", stroke_width=2.8, stroke_linecap="round")
        s.add("path", d=f"M{num(x0)} 0H{num(x1)}", stroke="#d7dade", stroke_width=1.4, stroke_linecap="round")
    fa, fb = 0.86 * rf, math.sqrt(1 - 0.86 ** 2) * rf
    s.add("path", d=f"M{num(fa)} {num(fb)}A{num(rf)} {num(rf)} 0 1 1 {num(fa)} {num(-fb)}Z",
          fill=mix(c, "#000000", 0.25),
          stroke=mix(c, "#000000", 0.5), stroke_width=0.8)
    s.add("circle", r=rd, fill=s.grad("radial", [(0, mix(c, "#ffffff", 0.5), 1), (0.55, c, 1),
                                                  (1, mix(c, "#000000", 0.45), 1)], cx="0.4", cy="0.38", r="0.7"))
    s.add("circle", r=rd * 0.34, fill="#fff", opacity=0.6 * glow[-1], data_series=nums(0.6 * g for g in glow),
          data_attr="opacity")
    s.add("ellipse", cx=-0.35 * rd, cy=-0.4 * rd, rx=0.3 * rd, ry=0.17 * rd, fill="#fff", opacity=0.55,
          transform=f"rotate(-35 {num(-0.35 * rd)} {num(-0.4 * rd)})")
    s.close()
    return cx, cy, rd


def capacitor_top(s: Svg, a, b, farads: float, p: float):
    """A radial capacitor from above: an electrolytic can from 1 uF up, a ceramic disc below that."""
    (ax, ay), (bx, by) = a, b
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    ang = math.degrees(math.atan2(by - ay, bx - ax))
    s.open(transform=f"translate({num(cx)} {num(cy)}) rotate({num(ang)})")
    half = math.hypot(bx - ax, by - ay) / 2
    for x0, x1 in ((-half, -0.5 * p), (0.5 * p, half)):
        s.add("path", d=f"M{num(x0)} 0H{num(x1)}", stroke="#80858c", stroke_width=2.8, stroke_linecap="round")
        s.add("path", d=f"M{num(x0)} 0H{num(x1)}", stroke="#d7dade", stroke_width=1.4, stroke_linecap="round")
    if farads >= 1e-6:
        r = 1.05 * p
        s.add("circle", r=r, fill="#1f2328", stroke="#000", stroke_opacity=0.5)
        s.add("path", d=f"M{num(r * 0.5)} {num(-r * 0.866)}A{num(r)} {num(r)} 0 0 1 {num(r * 0.5)} {num(r * 0.866)}"
                        f"L{num(r * 0.42)} {num(r * 0.7)}A{num(r * 0.82)} {num(r * 0.82)} 0 0 0 {num(r * 0.42)} "
                        f"{num(-r * 0.7)}Z", fill="#9aa1a8")  # the stripe that marks the negative lead
        s.add("circle", r=r * 0.78, fill=s.grad("radial", [(0, "#dfe2e5", 1), (1, "#9ba2a9", 1)], cx="0.4", cy="0.4"))
        s.add("path", d=f"M{num(-r * 0.45)} 0H{num(r * 0.45)}M0 {num(-r * 0.45)}V{num(r * 0.45)}", stroke="#7d848b",
              stroke_width=1.2)
    else:
        r = 0.85 * p
        s.add("ellipse", rx=r * 1.05, ry=r * 0.8, fill="#c4743a", stroke="#7d4520")
        exp = math.floor(math.log10(farads * 1e12)) - 1
        code = f"{round(farads * 1e12 / 10 ** exp)}{exp}" if exp >= 0 else f"{farads * 1e12:.2g}"
        s.text(0, 2.5, code, size=6.5, fill="#3b2410", anchor="middle", transform=f"rotate({num(-ang)})")
    s.close()


def jumper(s: Svg, pts, color: str):
    """A pre cut jumper wire: straight runs, square bends and bare tips in the holes."""
    d = poly(pts)
    s.add("path", d=d, fill="none", stroke=mix(color, "#000000", 0.45), stroke_width=7, stroke_linejoin="round")
    s.add("path", d=d, fill="none", stroke=color, stroke_width=5, stroke_linejoin="round")
    s.add("path", d=d, fill="none", stroke="#fff", stroke_opacity=0.28, stroke_width=1.4, stroke_linejoin="round")
    for x, y in (pts[0], pts[-1]):
        s.add("circle", cx=x, cy=y, r=2.2, fill="#b8bdc3", stroke="#5b574d", stroke_width=0.8)


def probe(s: Svg, cable: str, tip, ang: float, ring: str = YEL):
    """A scope probe: its cable, a slim body pointing at the tip and a colored channel ring."""
    lead(s, cable, "#4b5057", 3)
    s.open(transform=f"translate({num(tip[0])} {num(tip[1])}) rotate({num(ang)})")
    s.add("path", d="M0 0L-8 0", stroke="#c3c8cd", stroke_width=1.6, stroke_linecap="round")
    s.add("rect", x=-44, y=-4.5, width=36, height=9, rx=4.5, fill="#2a2d31", stroke="#4a4f57", stroke_width=0.8)
    s.add("rect", x=-36, y=-4.5, width=4, height=9, fill=ring)
    s.close()


def scope(s: Svg, x, y, w, h, traces, sweep, marks=(), inputs=("1", "2")) -> dict:
    """An oscilloscope with its traces (x and y from 0 to 1) on the graticule; the site draws them in."""
    s.add("rect", x=x + 3, y=y + 5, width=w, height=h, rx=7, fill="#000", opacity=0.5)
    s.add("rect", x=x, y=y, width=w, height=h, rx=7, fill="#1b1e22", stroke="#3b4047")
    s.add("rect", x=x + 1, y=y + 1, width=w - 2, height=7, rx=6, fill="#24282d")
    s.text(x + 12, y + 21, "OSCILLOSCOPE", size=8.5, fill=MUTED, ls=1.2)
    sx, sy, sw, sh = x + 12, y + 28, w - 62, h - 44
    s.add("rect", x=sx, y=sy, width=sw, height=sh, rx=3, fill="#070a0a", stroke="#30353b")
    grid = "".join(f"M{num(sx + sw * i / 10)} {num(sy)}v{num(sh)}" for i in range(1, 10))
    grid += "".join(f"M{num(sx)} {num(sy + sh * j / 8)}h{num(sw)}" for j in range(1, 8))
    s.add("path", d=grid, stroke="#172021", stroke_width=0.8)
    s.add("path", d=f"M{num(sx + sw / 2)} {num(sy)}v{num(sh)}M{num(sx)} {num(sy + sh / 2)}h{num(sw)}", stroke="#27302f",
          stroke_width=0.8)
    for mx in marks:
        s.add("path", d=f"M{num(sx + 3 + mx * (sw - 6))} {num(sy + 2)}v{num(sh - 4)}", stroke=MUTED, stroke_width=1,
              stroke_dasharray="2 3")
    for xs, ys, color in traces:
        d = poly([(sx + 3 + a * (sw - 6), sy + sh - 3 - b * (sh - 6)) for a, b in zip(xs, ys)])
        s.open(fill="none", stroke=color, stroke_linejoin="round", data_sweep=sweep)
        s.add("path", d=d, stroke_width=4, stroke_opacity=0.22, pathLength=1)
        s.add("path", d=d, stroke_width=1.6, pathLength=1)
        s.close()
    knob(s, x + w - 29, y + 40, 200)
    knob(s, x + w - 29, y + 70, 60)
    at = {}
    for i, ch in enumerate(inputs):
        bx, by = x + w - 42 + 22 * i, y + h - 14
        s.add("circle", cx=bx, cy=by, r=7, fill="#aab0b6", stroke="#5d636b")
        s.add("circle", cx=bx, cy=by, r=3.4, fill="#16181b")
        s.add("circle", cx=bx, cy=by - 10.5, r=1.6, fill=YEL if ch == "1" else "#5aa9ff")
        at[ch] = (bx, by)
    return at


def eng(x: float, unit: str) -> str:
    """A value with an SI prefix and three significant digits: 10 µF, 500 kHz, 100 ms."""
    for scale, pre in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if abs(x) >= scale * 0.9999:
            return f"{fmt(x / scale)} {pre}{unit}"
    return f"{fmt(x * 1e12)} p{unit}"
