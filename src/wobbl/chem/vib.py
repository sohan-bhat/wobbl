"""Vibrations: the Hessian by finite differences of forces, normal modes, frequencies, IR intensities and names.

Translations and rotations are projected out of the mass weighted Hessian (Eckart conditions), so a molecule has
3N-6 modes (3N-5 when linear). A minimum has none imaginary, which is the check that an optimization ended at a
structure that exists rather than a saddle point. IR intensities come from how the dipole moves along each mode,
in km/mol, and each mode is named by the bond or angle that changes most along it, like 'C=O stretch'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AMU = 1.66053906660e-27
EV = 1.602176634e-19
C_CM = 2.99792458e10
TO_CM = np.sqrt(EV / 1e-20 / AMU) / (2 * np.pi * C_CM)      # sqrt(eV/A^2/amu) to cm-1
IR_KM_MOL = 42.2561                                         # (D/A)^2/amu to km/mol


@dataclass
class Modes:
    freqs: np.ndarray            # cm-1, imaginary ones negative, translations and rotations removed
    vectors: np.ndarray          # (nmodes, n, 3) Cartesian displacements, each normalized
    intensities: np.ndarray      # km/mol
    labels: list[str]
    linear: bool
    reduced_masses: np.ndarray

    def imaginary(self, below: float = -20.0) -> list[int]:
        return [i for i, f in enumerate(self.freqs) if f < below]


def hessian(engine, positions: np.ndarray, step: float = 0.005):
    """(H in eV/A^2, dipole derivatives in D/A as (3n, 3)), by central differences of forces."""
    x0 = np.asarray(positions, float)
    n = len(x0)
    H = np.zeros((3 * n, 3 * n))
    dmu = np.zeros((3 * n, 3))
    for i in range(n):
        for k in range(3):
            res = []
            for s in (+1, -1):
                x = x0.copy()
                x[i, k] += s * step
                res.append(engine.compute(x))
            H[3 * i + k] = -(res[0].forces - res[1].forces).ravel() / (2 * step)
            dmu[3 * i + k] = (res[0].dipole - res[1].dipole) / (2 * step)
    engine.compute(x0)
    return 0.5 * (H + H.T), dmu


def is_linear(positions: np.ndarray, tol: float = 1e-3) -> bool:
    x = positions - positions.mean(axis=0)
    if len(x) < 3:
        return True
    s = np.linalg.svd(x, compute_uv=False)
    return s[1] < tol * max(s[0], 1e-12) * 10


def _tr_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Orthonormal mass weighted vectors of the three translations and the rotations."""
    n = len(masses)
    sm = np.sqrt(masses)
    com = (masses[:, None] * positions).sum(0) / masses.sum()
    x = positions - com
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = sm
        vecs.append(v.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        v = np.cross(axis, x) * sm[:, None]
        vecs.append(v.ravel())
    B = np.array(vecs).T
    u, s, _ = np.linalg.svd(B, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def normal_modes(H: np.ndarray, dmu: np.ndarray, positions: np.ndarray, masses: np.ndarray, symbols: list[str],
                 bonds: list[tuple[int, int, float]] | None = None) -> Modes:
    n = len(masses)
    mw = np.repeat(masses, 3)
    Hm = H / np.sqrt(np.outer(mw, mw))
    if n == 1:
        return Modes(np.zeros(0), np.zeros((0, 1, 3)), np.zeros(0), [], False, np.zeros(0))
    T = _tr_basis(positions, masses)
    P = np.eye(3 * n) - T @ T.T
    Hp = P @ Hm @ P
    w, v = np.linalg.eigh(Hp)
    # drop the eigenvectors that live in the translation and rotation space
    overlap = np.linalg.norm(T.T @ v, axis=0)
    keep = np.argsort(overlap)[: 3 * n - T.shape[1]]
    keep = keep[np.argsort(w[keep])]
    w, v = w[keep], v[:, keep]
    freqs = np.sign(w) * np.sqrt(np.abs(w)) * TO_CM
    cart = v / np.sqrt(mw)[:, None]                       # mass weighted to Cartesian displacements
    red = 1.0 / (cart ** 2).sum(0)
    norm = cart / np.linalg.norm(cart, axis=0)
    dmu_dq = dmu.T @ (v / np.sqrt(mw)[:, None])           # (3, nmodes) D/(A amu^1/2)
    inten = (dmu_dq ** 2).sum(0) * IR_KM_MOL
    vecs = norm.T.reshape(-1, n, 3)
    labels = [label(vecs[k], positions, symbols, bonds) for k in range(len(freqs))]
    return Modes(freqs, vecs, inten, labels, T.shape[1] == 5, red)


def label(disp: np.ndarray, positions: np.ndarray, symbols: list[str],
          bonds: list[tuple[int, int, float]] | None) -> str:
    """The bond stretch or angle bend that changes most along a mode."""
    if not bonds:
        return ""
    best, what = 0.0, ""
    for i, j, order in bonds:
        u = positions[j] - positions[i]
        u /= np.linalg.norm(u)
        d = abs(np.dot(disp[j] - disp[i], u))
        if d > best:
            a, b = sorted((symbols[i], symbols[j]), key=_writing_order)
            best, what = d, f"{a}{_bond(order)}{b} stretch"
    neighbours: dict[int, list[int]] = {}
    for i, j, _ in bonds:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    best_b, bend = 0.0, ""
    for c, ns in neighbours.items():
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                i, j = ns[a], ns[b]
                r1, r2 = positions[i] - positions[c], positions[j] - positions[c]
                l1, l2 = np.linalg.norm(r1), np.linalg.norm(r2)
                cos = np.dot(r1, r2) / (l1 * l2)
                # change of the angle per unit of mode displacement, scaled by bond length to compare with stretches
                g1 = (r2 / (l1 * l2) - cos * r1 / l1 ** 2)
                g2 = (r1 / (l1 * l2) - cos * r2 / l2 ** 2)
                sin = max(np.sqrt(max(1 - cos * cos, 0.0)), 1e-6)
                dth = abs(np.dot(g1, disp[i] - disp[c]) + np.dot(g2, disp[j] - disp[c])) / sin
                score = dth * 0.5 * (l1 + l2) * 0.6
                if score > best_b:
                    best_b, bend = score, f"{symbols[i]}-{symbols[c]}-{symbols[j]} bend"
    if best_b > best:
        return bend
    return what


def _writing_order(sym: str) -> tuple:
    """C before O, metals before nonmetals and hydrogen last, the way bonds are written."""
    from .formula import element
    if sym == "H":
        return (2, 0.0)
    en = element(sym).get("electronegativity") or 2.0
    return (0 if sym == "C" else 1, en)


def _bond(order: float) -> str:
    return "#" if order > 2.5 else "=" if order > 1.6 else ":" if order > 1.3 else "-"


def spectrum(freqs: np.ndarray, inten: np.ndarray, fwhm: float = 20.0, lo: float = 400.0, hi: float = 4000.0,
             points: int = 1801) -> tuple[np.ndarray, np.ndarray]:
    """A Lorentzian broadened IR spectrum, absorbance in km/mol per cm-1."""
    x = np.linspace(lo, hi, points)
    y = np.zeros_like(x)
    g = fwhm / 2
    for f, a in zip(freqs, inten):
        if f > 0:
            y += a * g / np.pi / ((x - f) ** 2 + g * g)
    return x, y
