"""Stress intensity factors along a crack front from the opening of the crack faces behind it.

Near a crack front the faces open as the square root of the distance r behind it, in proportion to K. In a frame
with x ahead of the front, y normal to the crack and z along it, the jump across the faces is

    du_y = 8 K_I sqrt(r / 2 pi) / E',  du_x = 8 K_II sqrt(r / 2 pi) / E',  du_z = 4 K_III sqrt(r / 2 pi) / mu

with E' = E / (1 - nu^2) along the front and mu the shear modulus (displacement correlation, Chan, Tuba and Wilson,
1970). Each node pair on the faces behind a front point gives an apparent K, and a straight line through them in
r is read at r = 0, which takes out the next term of the field. The quarter point elements make the apparent K
nearly flat over the first element, so the line is short and steady. Values along the front are then smoothed
with a local quadratic, which also carries them to the two ends, where a front meets a free surface and the
square root field gives way to a different singularity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .crackmesh import CrackedMesh
from .handbook import k_equivalent, mts_angle

SQRT_MM = math.sqrt(1e-3)     # K in MPa sqrt(mm) times this is MPa sqrt(m)


@dataclass
class FrontK:
    """Stress intensity factors at the corner nodes along one front, in MPa sqrt(m)."""
    nodes: np.ndarray
    xyz: np.ndarray
    s: np.ndarray                 # distance along the front, mm
    forward: np.ndarray           # unit vectors: ahead of the front, normal to the crack (up), along the front
    normal: np.ndarray
    tangent: np.ndarray
    raw: np.ndarray               # (k, 3) K_I, K_II, K_III at each node before smoothing
    k: np.ndarray                 # (k, 3) smoothed
    nu: float
    fit_points: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def length(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    def k_eq(self) -> np.ndarray:
        return k_equivalent(self.k[:, 0], self.k[:, 1], self.k[:, 2], self.nu)

    def kink(self) -> np.ndarray:
        return mts_angle(self.k[:, 0], self.k[:, 1])

    def at(self, frac: float) -> np.ndarray:
        """Smoothed K at a fraction of the way along the front."""
        x = frac * self.length
        return np.array([np.interp(x, self.s, self.k[:, i]) for i in range(3)])

    def to_dict(self) -> dict:
        return {"length_mm": self.length, "points": len(self.s),
                "k1_max": float(self.k[:, 0].max()), "k2_absmax": float(np.abs(self.k[:, 1]).max()),
                "k3_absmax": float(np.abs(self.k[:, 2]).max()), "k_eq_max": float(self.k_eq().max())}


def extract(cm: CrackedMesh, U: np.ndarray, E: float, nu: float, reach: float = 4.0) -> list[FrontK]:
    """K_I, K_II and K_III along every front from the displacements of a solve on the cracked mesh."""
    X = cm.mesh.X
    lo, up = cm.pairs[:, 0], cm.pairs[:, 1]
    du = U[up] - U[lo]
    pts = X[lo]
    tree = cKDTree(pts)
    tri = cm.crack_tris
    nrm = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    cen = X[tri[:, :3]].mean(axis=1)
    tri_tree = cKDTree(cen)
    e_prime = E / (1 - nu ** 2)
    mu = E / (2 * (1 + nu))
    hf = cm.sizing.h_front
    out = []
    for chain in cm.fronts:
        P = X[chain]
        if len(P) < 3:
            continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        tan = np.gradient(P, s, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1), 1e-300)[:, None]
        raw = np.full((len(P), 3), np.nan)
        fwd, up_n = np.zeros_like(P), np.zeros_like(P)
        npts = np.zeros(len(P), int)
        for i, p in enumerate(P):
            near = tri_tree.query_ball_point(p, 2.0 * hf)
            if not near:
                near = [int(tri_tree.query(p)[1])]
            n = nrm[near].sum(0)
            c = cen[near].mean(0)
            e1 = np.cross(tan[i], n)
            e1 /= max(np.linalg.norm(e1), 1e-300)
            if (p - c) @ e1 < 0:
                e1 = -e1
            e2 = n - (n @ e1) * e1 - (n @ tan[i]) * tan[i]
            e2 /= max(np.linalg.norm(e2), 1e-300)
            e3 = np.cross(e1, e2)
            fwd[i], up_n[i] = e1, e2
            half = 0.6 * max(seg[max(i - 1, 0)], seg[min(i, len(seg) - 1)])
            cand = np.asarray(tree.query_ball_point(p, reach * hf + half), int)
            if not len(cand):
                continue
            d = pts[cand] - p
            r = -(d @ e1)
            ok = (r > 1e-3 * hf) & (r <= reach * hf) & (np.abs(d @ tan[i]) <= half) & (np.abs(d @ e2) <= 0.5 * r + 1e-3 * hf)
            if ok.sum() < 1:
                continue
            r, dl = r[ok], du[cand[ok]]
            f = np.sqrt(2 * math.pi / r)
            app = np.column_stack([e_prime / 8 * f * (dl @ e2), e_prime / 8 * f * (dl @ e1), mu / 4 * f * (dl @ e3)])
            # the quarter point nodes right at the front scatter with the shape of each element, so the line goes
            # through the nodes from half an element back, where the opening is already a clean square root
            far = r >= 0.5 * hf
            if far.sum() >= 3:
                r, app = r[far], app[far]
            npts[i] = len(r)
            if len(np.unique(np.round(r / hf, 3))) >= 2 and len(r) >= 3:
                A = np.column_stack([np.ones_like(r), r])
                coef, *_ = np.linalg.lstsq(A, app, rcond=None)
                raw[i] = coef[0]
            else:
                raw[i] = app.mean(0)
        raw *= SQRT_MM
        out.append(FrontK(chain, P, s, fwd, up_n, tan, raw, smooth(s, raw), nu, npts))
    return out


def smooth(s: np.ndarray, vals: np.ndarray, band: float = 0.12, end: float = 0.06) -> np.ndarray:
    """Local quadratic fit along the front with Gaussian weights over about a tenth of its length. Points within a
    few percent of either end, where the front meets a free surface, are left out of the fits, and the fits carry
    the values to them."""
    L = float(s[-1]) if len(s) else 0.0
    out = np.full_like(vals, np.nan)
    if L <= 0:
        return vals.copy()
    good = np.isfinite(vals).all(axis=1) & (s >= end * L) & (s <= (1 - end) * L)
    if good.sum() < 4:
        good = np.isfinite(vals).all(axis=1)
    if good.sum() < 2:
        return np.where(np.isfinite(vals), vals, 0.0)
    w_band = max(band * L, 1e-9)
    xs, ys = s[good], vals[good]
    for i, x in enumerate(s):
        w = np.exp(-0.5 * ((xs - x) / w_band) ** 2)
        if w.sum() < 2.5:   # near an end the band holds few points: widen it
            w = np.exp(-0.5 * ((xs - x) / (2 * w_band)) ** 2)
        deg = 2 if len(xs) >= 5 else 1
        A = np.vander(xs - x, deg + 1)
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], ys * sw[:, None], rcond=None)
        out[i] = coef[-1]
    return out
