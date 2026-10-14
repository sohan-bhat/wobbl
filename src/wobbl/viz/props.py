"""Props: the things around a part that make a scene read as the user's situation, a wall, bolts, a shelf, a motor,
a launch pad. Each sits where the run's own supports and loads are, and every one is marked as a prop, so the viewer
draws it with faint stripes and lists it as illustrative. None of them takes part in any analysis.
"""
from __future__ import annotations

import numpy as np

WALL, WOOD, STEEL, DARK, BOLT, WEIGHT, YELLOW = "#6d7178", "#8b6b47", "#8d939b", "#2c2f34", "#c3c7cd", "#50555d", "#ffd400"


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-30)


def basis(n, hint=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two directions square to n and to each other, the first level when n is not vertical."""
    n = unit(n)
    u = np.cross(hint, n)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross((1.0, 0.0, 0.0), n)
    u = unit(u)
    return u, np.cross(n, u), n


def quat(u, v, w) -> list[float]:
    """The quaternion (w, x, y, z) turning the x, y and z axes onto u, v and w."""
    m = np.column_stack([unit(u), unit(v), unit(w)])
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    return list(np.asarray(q) / np.linalg.norm(q))


def z_to(d) -> list[float]:
    """The quaternion turning +z onto direction d."""
    u, v, w = basis(d)
    return quat(u, v, w)


def slab(s, center, normal, half_u: float, half_v: float, thick: float, color=WALL, name="wall", hint=(0, 0, 1)):
    """A flat block whose face lies at center, facing along normal, extending behind it."""
    u, v, n = basis(normal, hint)
    c = np.asarray(center, float) - n * thick / 2
    return s.body("box", [half_u, half_v, thick / 2], pos=c, quat=quat(u, v, n), color=color, name=name, prop=True)


def bolt(s, center, axis, radius: float, length: float, head: float = 1.0, color=BOLT, name="bolt"):
    """A bolt through a hole: the shank along axis, centered at center, and a hex head at the end head points to."""
    a = unit(axis) * np.sign(head or 1)
    c = np.asarray(center, float)
    s.body("cylinder", [radius * 0.92, length / 2], pos=c, quat=z_to(a), color=color, name=name, prop=True)
    hh = radius * 0.75
    s.body("hex", [radius * 1.75, hh / 2], pos=c + a * (length / 2 + hh / 2), quat=z_to(a), color=color,
           name=f"{name} head", prop=True)


def block(s, base, up, half_u: float, half_v: float, height: float, color=WEIGHT, name="weight", hint=(1, 0, 0),
          opacity: float = 1.0):
    """A block standing on base, rising along up."""
    u, v, n = basis(up, hint)
    c = np.asarray(base, float) + n * height / 2
    return s.body("box", [half_u, half_v, height / 2], pos=c, quat=quat(u, v, n), color=color, name=name, prop=True,
                  opacity=opacity)


def motor(s, face, normal, shaft_end, size: float, name="motor"):
    """A stepper motor bolted against a face, its body out along normal, its shaft running to shaft_end where a
    pulley sits."""
    n = unit(normal)
    f = np.asarray(face, float)
    u, v, _ = basis(n)
    h = size / 2
    s.body("box", [h, h, size * 0.45], pos=f + n * size * 0.45, quat=quat(u, v, n), color=DARK, name=name, prop=True)
    s.body("box", [h * 1.01, h * 1.01, size * 0.06], pos=f + n * size * 0.06, quat=quat(u, v, n), color=STEEL,
           name=f"{name} flange", prop=True)
    end = np.asarray(shaft_end, float)
    d = end - f
    L = float(np.linalg.norm(d))
    if L > 1e-9:
        s.body("cylinder", [size * 0.06, L / 2 + size * 0.05], pos=f + d / 2, quat=z_to(d), color=BOLT,
               name=f"{name} shaft", prop=True)
        s.body("cylinder", [size * 0.2, size * 0.1], pos=end, quat=z_to(d), color=STEEL, name="pulley", prop=True)


def arrow(s, at, direction, length: float, scale=None, label: str = "", color=YELLOW, name="load", radius=None):
    """A force arrow whose tip touches at, pointing along direction, growing with scale per frame."""
    d = unit(direction)
    tip = np.asarray(at, float)
    if scale is None:
        return s.arrows([tip - d * length], [d * length], color=color, labels=[label] if label else None, name=name,
                        radius=radius)
    sc = np.asarray(scale, float)
    o = tip[None, None, :] - sc[:, None, None] * d * length
    return s.arrows(o, np.broadcast_to(d * length, o.shape), scale=sc, color=color,
                    labels=[label] if label else None, name=name, radius=radius)


def pad(s, at, size: float, rail: float = 0.0, name="launch pad"):
    """A launch pad: a base plate on the ground with a rail standing on it."""
    p = np.asarray(at, float)
    s.body("box", [size, size, size * 0.06], pos=p + [0, 0, size * 0.06], color=STEEL, name=name, prop=True)
    if rail > 0:
        s.body("box", [size * 0.05, size * 0.05, rail / 2], pos=p + [size * 0.2, 0, rail / 2 + size * 0.12],
               color=BOLT, name="launch rail", prop=True)


def table(s, top_center, half_u: float, half_v: float, height: float, name="table"):
    c = np.asarray(top_center, float)
    t = max(half_u, half_v) * 0.05
    s.body("box", [half_u, half_v, t], pos=c - [0, 0, t], color=WOOD, name=name, prop=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.body("box", [t, t, height / 2], pos=c + [sx * (half_u - 2 * t), sy * (half_v - 2 * t), -height / 2 - t],
                   color=WOOD, name=f"{name} leg", prop=True)


def wire_box(s, lo, hi, color="#5a5f66", name="oven", width=1.5):
    """The edges of a box, as an oven or an enclosure seen through."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = [np.where([i & 1, i & 2, i & 4], hi, lo) for i in range(8)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return s.lines([np.array([c[a], c[b]]) for a, b in edges], color=color, width=width, name=name, prop=True)
