"""Bench parts in 3D for circuit scenes: resistors with their color bands, LEDs, capacitors, ICs, transistors,
wires as real round leads, a bench supply and the bench it all sits on. Sizes are in mm and follow the real parts.

Every function takes the scene and returns the bodies it made, so the adapter can color, light or flag them.
"""
from __future__ import annotations

import math

import numpy as np

from . import props as P

BAND = ["#1a1a1a", "#7a4a1e", "#d0312d", "#f28c28", "#f2d22e", "#2f9e44", "#2f6fd6", "#8e44ad", "#8a8a8a",
        "#f4f4f4"]
GOLD, SILVER = "#c9a33a", "#b8bcc4"
LEAD = "#c3c7cd"
BODY_TAN = "#d9c49a"


def bands(ohms: float | None) -> list[str]:
    """The four band colors of a resistor of this value: two digits, the multiplier and gold for 5%."""
    if not ohms or ohms <= 0 or not math.isfinite(ohms):
        return []
    e = int(math.floor(math.log10(ohms))) - 1
    digits = int(round(ohms / 10 ** e))
    if digits >= 100:
        digits //= 10
        e += 1
    d1, d2 = divmod(digits, 10)
    mult = BAND[e] if 0 <= e <= 9 else GOLD if e == -1 else SILVER if e == -2 else BAND[0]
    return [BAND[d1], BAND[d2], mult, GOLD]


def wire(s, pts, radius: float = 0.32, color="#2b2e33", name="wire", prop=True) -> list:
    """A round wire through the points, a capsule per straight piece so the bends close up."""
    out = []
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        out.append(s.body("capsule", [radius, L / 2], pos=(a + b) / 2, quat=P.z_to(d), color=color, name=name,
                          prop=prop))
    return out


def arch(a, b, height: float, n: int = 7) -> np.ndarray:
    """Points of a lead bent up and over from a to b, as a jumper or a test lead hangs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = np.linspace(0, 1, n)
    pts = a + np.outer(u, b - a)
    pts[:, 2] += height * np.sin(np.pi * u) ** 0.7
    return pts


def resistor(s, a, b, ohms, name: str, lift: float = 3.0):
    """An axial resistor between two holes: leads bent down into them and a tan body with its color bands.
    Returns the body, which the caller may color by heat."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d)) or 2.54
    u = d / L
    mid = (a + b) / 2 + [0, 0, lift]
    bl = min(L * 0.6, 6.3)
    q = P.z_to(u)
    wire(s, [a, a + [0, 0, lift], mid - u * bl / 2], 0.25, LEAD, f"{name} lead")
    wire(s, [mid + u * bl / 2, b + [0, 0, lift], b], 0.25, LEAD, f"{name} lead")
    body = s.body("capsule", [1.1, bl / 2 - 1.1], pos=mid, quat=q, color=BODY_TAN, name=name)
    for i, c in enumerate(bands(ohms)):
        at = -bl * 0.3 + i * bl * 0.17 + (bl * 0.08 if i == 3 else 0)
        s.body("cylinder", [1.16, bl * 0.035], pos=mid + u * at, quat=q, color=c, name=f"{name} band")
    return body


def diode(s, a, b, name: str, lift: float = 3.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d)) or 2.54
    u = d / L
    mid = (a + b) / 2 + [0, 0, lift]
    bl = min(L * 0.5, 4.2)
    wire(s, [a, a + [0, 0, lift], mid - u * bl / 2], 0.25, LEAD, f"{name} lead")
    wire(s, [mid + u * bl / 2, b + [0, 0, lift], b], 0.25, LEAD, f"{name} lead")
    body = s.body("cylinder", [1.0, bl / 2], pos=mid, quat=P.z_to(u), color="#202226", name=name)
    s.body("cylinder", [1.04, bl * 0.06], pos=mid + u * bl * 0.36, quat=P.z_to(u), color=SILVER, name=f"{name} band")
    return body


def legs(s, pins, height: float, name: str, radius: float = 0.25):
    for q in pins:
        wire(s, [q, np.asarray(q) + [0, 0, height]], radius, LEAD, f"{name} lead")


def led(s, pins, color: str, name: str, lift: float = 3.0):
    """A 5 mm LED standing on its two legs: its tinted body and dome. Returns the parts that glow."""
    mid = np.mean(pins, axis=0)
    legs(s, pins, lift, name)
    rim = s.body("cylinder", [2.9, 0.5], pos=mid + [0, 0, lift + 0.5], color=color, name=f"{name} rim")
    cyl = s.body("cylinder", [2.5, 2.2], pos=mid + [0, 0, lift + 3.2], color=color, name=name)
    dome = s.body("sphere", [2.5], pos=mid + [0, 0, lift + 5.4], color=color, name=f"{name} dome")
    return [rim, cyl, dome]


def capacitor(s, pins, farads: float | None, name: str, lift: float = 1.0):
    """An electrolytic can for 1 uF and up, a small ceramic disc below. Returns the body."""
    mid = np.mean(pins, axis=0)
    big = (farads or 0) >= 1e-6
    if big:
        r = float(np.clip(2.5 + 1.2 * math.log10(max(farads, 1e-6) / 1e-6) / 2, 2.5, 5.0))
        h = r * 2.2
        legs(s, pins, lift, name)
        body = s.body("cylinder", [r, h / 2], pos=mid + [0, 0, lift + h / 2], color="#1d3a63", name=name)
        s.body("cylinder", [r * 0.92, 0.12], pos=mid + [0, 0, lift + h + 0.05], color=SILVER, name=f"{name} top")
        side = np.array([r * 0.97, 0, 0])
        s.body("box", [0.25, r * 0.3, h / 2 * 0.96], pos=mid + side + [0, 0, lift + h / 2], color="#c8ccd2",
               name=f"{name} stripe")
        return body
    legs(s, pins, lift + 1.5, name)
    return s.body("ellipsoid", [2.4, 1.0, 2.4], pos=mid + [0, 0, lift + 3.6], color="#d98a2b", name=name)


def inductor(s, pins, name: str):
    mid = np.mean(pins, axis=0)
    legs(s, pins, 1.5, name)
    body = s.body("torus", [3.4, 1.4], pos=mid + [0, 0, 4.9], quat=P.z_to([0, 1, 0]), color="#3a4a3a", name=name)
    s.body("torus", [3.4, 1.55], pos=mid + [0, 0, 4.9], quat=P.z_to([0, 1, 0]), color="#b87333", name=f"{name} coil",
           opacity=0.55)
    return body


def to220(s, pins, name: str):
    """A TO-220 package standing on its three legs with its metal tab behind."""
    mid = np.mean(pins, axis=0)
    legs(s, pins, 3.0, name, 0.35)
    body = s.body("box", [5.0, 2.25, 4.6], pos=mid + [0, 0, 3.0 + 4.6], color="#1c1d20", name=name)
    s.body("box", [5.0, 0.65, 3.4], pos=mid + [0, 2.9, 3.0 + 9.2 + 1.6], color="#c9ccd1", name=f"{name} tab")
    s.body("cylinder", [1.8, 0.7], pos=mid + [0, 2.9, 3.0 + 10.8], quat=P.z_to([0, 1, 0]), color="#2a2c30",
           name=f"{name} tab hole")
    return body


def to92(s, pins, name: str):
    mid = np.mean(pins, axis=0)
    legs(s, pins, 3.0, name, 0.22)
    return s.body("cylinder", [2.3, 2.4], pos=mid + [0, 0, 5.4], color="#1c1d20", name=name)


def dip(s, pins, name: str):
    """A DIP package across the channel: its pins in two rows and a notch at pin 1. Returns the body."""
    pins = np.asarray(pins, float)
    lo, hi = pins.min(0), pins.max(0)
    mid = (lo + hi) / 2
    half = np.array([max((hi[0] - lo[0]) / 2 + 1.3, 2.4), max((hi[1] - lo[1]) / 2 - 0.9, 2.0)])
    for q in pins:
        inward = np.sign(mid[1] - q[1]) or 1
        wire(s, [q, q + [0, 0, 2.2], q + [0, inward * 0.9, 3.2]], 0.28, LEAD, f"{name} pin")
    body = s.body("box", [half[0], half[1], 1.7], pos=mid + [0, 0, 4.6], color="#1c1d20", name=name)
    s.body("cylinder", [0.8, 0.1], pos=[lo[0] - 0.4, mid[1], 6.3], color="#3a3c40", name=f"{name} notch")
    return body


def module(s, pins, name: str):
    pins = np.asarray(pins, float)
    lo, hi = pins.min(0), pins.max(0)
    mid = (lo + hi) / 2
    half = np.maximum((hi - lo) / 2 + 1.6, 2.2)
    legs(s, pins, 2.0, name)
    s.body("box", [half[0], half[1], 0.8], pos=mid + [0, 0, 2.8], color="#1f5e3a", name=f"{name} board")
    return s.body("box", [half[0] * 0.7, half[1] * 0.7, 1.6], pos=mid + [0, 0, 5.2], color="#2a2c30", name=name)


def breadboard(s, W: float, H: float, holes: np.ndarray):
    """The white board, its holes as dark sockets and the bench under it."""
    s.body("box", [W / 2, H / 2, 4.25], pos=[W / 2, -H / 2, -4.25], color="#ece9e0", name="breadboard", prop=True)
    s.points(holes, color="#3c3c3a", size=0.42, name="holes")


def bench(s, center, half_u: float, half_v: float):
    """The wooden bench top everything stands on."""
    c = np.asarray(center, float)
    s.body("box", [half_u, half_v, 9.0], pos=c - [0, 0, 9.0], color="#6b543b", name="bench", prop=True)


def supply(s, at, name: str, posts: int = 2, display: str = "#7cf29a") -> list[np.ndarray]:
    """A bench power supply (or generator or meter, by its display color) with its display and terminal posts.
    Returns the posts' tips, + first."""
    at = np.asarray(at, float)
    W, D, H = 34.0, 26.0, 17.0
    s.body("box", [W / 2, D / 2, H / 2], pos=at + [0, 0, H / 2], color="#2b2e33", name=name, prop=True)
    s.body("box", [W / 2 + 0.3, 0.6, H / 2 - 1.5], pos=at + [0, -D / 2, H / 2], color="#3a3e45",
           name=f"{name} panel", prop=True)
    disp = s.body("box", [W * 0.36, 0.4, 3.0], pos=at + [0, -D / 2 - 0.6, H * 0.68], color=display,
                  name=f"{name} display")
    disp.light(np.full(s.n, 0.55))
    tips = []
    for i, col in enumerate(("#c8352b", "#1c1d20", "#2f9e44")[:posts]):
        x = (i - (posts - 1) / 2) * 8.0
        base = at + [x, -D / 2 - 0.6, H * 0.28]
        s.body("cylinder", [1.8, 2.0], pos=base + [0, -2.0, 0], quat=P.z_to([0, -1, 0]), color=col,
               name=f"{name} post", prop=True)
        tips.append(base + [0, -4.0, 0])
    return tips
