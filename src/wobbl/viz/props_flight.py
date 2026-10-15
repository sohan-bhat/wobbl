"""Shapes for flying things: a rocket's airframe from its spec, a parachute canopy, an exhaust plume, a rough rock,
and the scale marks that make a flight read at its real size. Every function returns plain vertices and triangles
in the body's own frame, nose up along +z, so the adapters move them with the run's own positions.
"""
from __future__ import annotations

import numpy as np

from .adapters.common import join, lathe


def nose_profile(shape: str, R: float, L: float, n: int = 14) -> list[tuple[float, float]]:
    """(radius, distance from the tip) down a nose cone of the named shape, from the tip to its shoulder."""
    x = np.linspace(0, 1, n)
    shape = (shape or "ogive").lower()
    if shape == "cone":
        f = x
    elif shape in ("elliptical", "ellipse"):
        f = np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1))
    elif shape == "parabolic":
        f = 2 * x - x * x
    elif shape in ("haack", "von karman", "vonkarman"):
        th = np.arccos(1 - 2 * x)
        f = np.sqrt(np.clip(th - np.sin(2 * th) / 2, 0, None) / np.pi)
    else:                                               # tangent ogive
        rho = (R * R + L * L) / (2 * R)
        f = (np.sqrt(np.clip(rho * rho - (L - x * L) ** 2, 0, None)) + R - rho) / R
    return [(float(R * max(fi, 0.0)), float(xi * L)) for fi, xi in zip(f, x)]


def airframe(length: float, diameter: float, nose_length: float, nose_shape: str = "ogive", fins=None,
             seg: int = 28) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The rocket as three meshes, nose, body tube and fins, with the aft end at z = 0 and the tip at z = length.
    fins carries count, root, tip, span, sweep and position (root leading edge from the tip), in metres."""
    R, L = diameter / 2, length
    nl = min(max(nose_length, diameter), L * 0.6)
    prof = nose_profile(nose_shape, R, nl)
    nose = lathe([(r, L - x) for r, x in prof][::-1] + [(0.0, L - nl)], seg)
    tube = lathe([(0.0, 0.0), (R, 0.0), (R, L - nl), (0.0, L - nl)], seg)
    out = {"nose": nose, "body tube": tube}
    if fins is not None and getattr(fins, "count", 0):
        th = max(diameter * 0.035, 0.002)
        z_le = L - (fins.position if fins.position else L - fins.root)
        pts = np.array([[R * 0.98, z_le], [R + fins.span, z_le - fins.sweep],
                        [R + fins.span, z_le - fins.sweep - fins.tip], [R * 0.98, z_le - fins.root]])
        plates = []
        for i in range(int(fins.count)):
            a = 2 * np.pi * i / fins.count + np.pi / 2
            ca, sa = np.cos(a), np.sin(a)
            V = []
            for side in (-1, 1):
                for r, z in pts:
                    V.append([r * ca - side * th / 2 * sa, r * sa + side * th / 2 * ca, z])
            # two quads for the faces and four for the edges, corners 0..3 on one side and 4..7 on the other
            T = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
            for j in range(4):
                k = (j + 1) % 4
                T += [(j, j + 4, k + 4), (j, k + 4, k)]
            plates.append((np.array(V), np.array(T)))
        out["fins"] = join(*plates)
    return out


def canopy(diameter: float, lines: float, seg: int = 20, gores: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A round parachute: a shallow dome of the given diameter whose shroud lines meet at the origin, the canopy
    above it at height lines."""
    R = diameter / 2
    ph = np.linspace(0, 1, 7)
    dome = [(R * np.sin(p * np.pi / 2.4) / np.sin(np.pi / 2.4), lines + R * 0.45 * np.cos(p * np.pi / 2.4))
            for p in ph]
    dome = [(0.0, dome[0][1])] + dome[1:] + [(R * 0.97, dome[-1][1] - R * 0.02), (0.0, dome[-1][1] - R * 0.02)]
    parts = [lathe(dome, seg)]
    w = max(diameter * 0.006, 0.004)
    rim = dome[-3][1]
    for i in range(gores):
        a = 2 * np.pi * i / gores
        top = np.array([R * np.cos(a), R * np.sin(a), rim])
        side = np.array([-np.sin(a), np.cos(a), 0.0]) * w
        parts.append((np.array([[0, 0, 0] - side, [0, 0, 0] + side, top + side, top - side]),
                      np.array([(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)])))
    return join(*parts)


def plume(radius: float, length: float, seg: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """An exhaust plume from the nozzle exit at z = 0 down to z = -length, widest a little way out."""
    k = np.linspace(0, 1, 9)
    prof = [(radius * (1 + 0.9 * np.sin(np.pi * min(x * 1.4, 1.0)) ** 0.7) * (1 - x) ** 0.6 + 1e-6, -length * x)
            for x in k]
    return lathe([(0.0, 0.0)] + prof + [(0.0, -length)], seg)


def rock(diameter: float, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A rough stone about diameter across: an icosphere pushed in and out by a few smooth bumps."""
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(2):
        mid, F2 = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = V[a] + V[b]
                V.append(p / np.linalg.norm(p))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    X = np.array(V)
    rng = np.random.default_rng(seed)
    bumps = rng.normal(size=(7, 3))
    bumps /= np.linalg.norm(bumps, axis=1, keepdims=True)
    amp = rng.uniform(-0.12, 0.12, 7)
    r = 1 + (np.maximum(X @ bumps.T, 0) ** 3 * amp).sum(1)
    X = X * r[:, None] * [1.0, 0.86, 0.78]
    return X * diameter / 2, np.array(F)


def nice_step(span: float, ticks: int = 6) -> float:
    """A round spacing that puts about ticks marks over span."""
    raw = max(span, 1e-9) / ticks
    p = 10 ** np.floor(np.log10(raw))
    return float(min((m * p for m in (1, 2, 2.5, 5, 10)), key=lambda v: abs(v - raw)))


def mast(s, at, top: float, step: float, name: str = "height scale", unit: str = "m"):
    """A thin pole standing at at with a tick and a label every step up to top, so heights read at a glance."""
    at = np.asarray(at, float)
    r = max(top * 0.0012, 0.02)
    s.body("cylinder", [r, top / 2], pos=at + [0, 0, top / 2], color="#5b6068", name=name, prop=True)
    ticks = [np.array([at + [-r * 8, 0, z], at + [r * 8, 0, z]]) for z in np.arange(step, top + 1e-9, step)]
    if ticks:
        s.lines(ticks, color="#9aa0a6", width=1.5, name=f"{name} ticks", prop=True)
    for z in np.arange(step, top + 1e-9, step):
        s.label(f"{z:g} {unit}", at + [r * 10, 0, z], kind="note")


def wind(s, center, speed: float, along, area: float, heights, rate: float = 1.0, name: str = "wind"):
    """Streaks across the pad that glow along the way the wind blows, and one arrow naming its speed. rate turns
    real seconds into playback seconds, so the glows move at the wind's own speed."""
    if speed <= 0:
        return
    d = np.asarray(along, float) / np.linalg.norm(along)
    side = np.array([-d[1], d[0], 0.0])
    c = np.asarray(center, float)
    paths = []
    for i, h in enumerate(heights):
        for j in (-1, 0, 1):
            o = c + side * j * area * 0.35 + [0, 0, h] + side * (i % 2) * area * 0.1
            paths.append(np.array([o - d * area / 2, o + d * area / 2]))
    s.lines(paths, color="#7fb7ff", width=1.2, flow=speed * rate, spacing=area / 6, name=name, prop=True)
    top = max(heights)
    s.arrows([c + [0, 0, top * 1.15] - d * area * 0.15], [d * area * 0.3], color="#7fb7ff",
             labels=[f"wind {speed:.2g} m/s"], name="wind arrow", radius=area * 0.006)
