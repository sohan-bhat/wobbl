"""Wires and cables as ready triangle meshes: a round tube swept along a smooth path with parallel transported
frames, which costs a few hundred triangles where a CAD sweep along a spline costs tens of thousands.

    from mech import cable
    part = cable.tube([(0, 0, 0), (10, 0, -2), (14, 0, -12)], 0.6)      # (('mesh', P, N, T))
    parts = cable.ribbon(path, 0.55, ['wire_brown', 'wire_red', 'wire_orange'], side=(0, 1, 0))
"""
import numpy as np


def smooth(points, n=48):
    """A Catmull-Rom curve through the points, sampled at n points along it."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 2:
        return np.linspace(P[0], P[1], n)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    seg = len(P) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * seg
        k = min(int(t), seg - 1)
        u = t - k
        p0, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def _frames(C, up=None):
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.asarray(up if up is not None else (0, 0, 1), dtype=np.float64)
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0])
    Nn = [np.cross(T[0], np.cross(ref, T[0]))]
    Nn[0] /= np.linalg.norm(Nn[0])
    for i in range(1, len(C)):
        v = Nn[-1] - np.dot(Nn[-1], T[i]) * T[i]
        Nn.append(v / np.linalg.norm(v))
    Nn = np.array(Nn)
    B = np.cross(T, Nn)
    return T, Nn, B


def tube(points, r, seg=10, n=None, caps=True, up=None, smooth_path=True):
    """A round tube of radius r along a path through points: (('mesh', P, N, T))."""
    C = smooth(points, n or max(8, len(points) * 12)) if smooth_path else np.asarray(points, dtype=np.float64)
    T_, N_, B_ = _frames(C, up)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.cos(a)[None, :, None] * N_[:, None, :] + np.sin(a)[None, :, None] * B_[:, None, :]
    P = (C[:, None, :] + r * ring).reshape(-1, 3)
    Nrm = ring.reshape(-1, 3)
    tris = []
    m = len(C)
    for i in range(m - 1):
        for j in range(seg):
            a0 = i * seg + j
            a1 = i * seg + (j + 1) % seg
            b0 = a0 + seg
            b1 = a1 + seg
            tris += [(a0, b0, a1), (a1, b0, b1)]
    P, Nrm = list(P), list(Nrm)
    if caps:
        for end, sgn in ((0, -1), (m - 1, 1)):
            base = len(P)
            P.append(C[end])
            Nrm.append(sgn * T_[end])
            for j in range(seg):
                P.append(C[end] + r * ring[end, j])
                Nrm.append(sgn * T_[end])
            for j in range(seg):
                a0, a1 = base + 1 + j, base + 1 + (j + 1) % seg
                tris.append((base, a1, a0) if sgn < 0 else (base, a0, a1))
    return ('mesh', np.array(P), np.array(Nrm), np.array(tris, dtype=np.int64))


def ribbon(points, r, materials, side=(0, 1, 0), seg=8, n=None):
    """Side by side wires of radius r (a servo lead) along a path, laid across the direction side:
    [(mesh, material)]."""
    C = smooth(points, n or max(8, len(points) * 12))
    T_, _, _ = _frames(C)
    s = np.asarray(side, dtype=np.float64)
    out = []
    k0 = (len(materials) - 1) / 2
    for k, m in enumerate(materials):
        off = []
        for t in T_:
            v = s - np.dot(s, t) * t
            off.append(v / np.linalg.norm(v))
        off = np.array(off) * (k - k0) * 2 * r * 0.98
        out.append((tube(C + off, r, seg, smooth_path=False), m))
    return out
