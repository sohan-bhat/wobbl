"""Thermochemistry of an ideal gas molecule from its structure and frequencies.

Translation, rotation and vibration in the rigid rotor and harmonic oscillator picture give the enthalpy,
entropy, Gibbs energy and heat capacity at a temperature and pressure. Vibrations softer than about 100 cm-1 are
mostly hindered rotations, so their entropy blends into a free rotor's as Grimme's quasi rigid rotor harmonic
oscillator does (Chem. Eur. J. 2012, 18, 9955), which keeps floppy molecules from getting absurd entropies. The
rotational symmetry number is found from the structure itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KB = 1.380649e-23
H = 6.62607015e-34
C = 2.99792458e10          # cm/s
NA = 6.02214076e23
R = KB * NA                # J/(mol K)
AMU = 1.66053906660e-27
EV_TO_KJ = 96.48533212
OMEGA0 = 100.0             # cm-1, where Grimme's blend is half rotor, half oscillator


@dataclass
class Thermo:
    T: float
    P: float
    zpe: float               # kJ/mol
    H: float                 # kJ/mol, thermal enthalpy correction including ZPE, relative to the electronic energy
    S: float                 # J/(mol K)
    G: float                 # kJ/mol, H - TS correction
    Cp: float                # J/(mol K)
    parts: dict              # S and Cp by translation, rotation, vibration, electronic
    sigma: int
    linear: bool

    def to_dict(self) -> dict:
        return {"T_K": self.T, "P_Pa": self.P, "zpe_kJ_mol": self.zpe, "H_corr_kJ_mol": self.H,
                "S_J_molK": self.S, "G_corr_kJ_mol": self.G, "Cp_J_molK": self.Cp, "symmetry_number": self.sigma,
                "parts": self.parts}


def moments(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Principal moments of inertia, amu A^2."""
    com = (masses[:, None] * positions).sum(0) / masses.sum()
    x = positions - com
    inertia = np.zeros((3, 3))
    for m, r in zip(masses, x):
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return np.sort(np.linalg.eigvalsh(inertia))


def symmetry_number(positions: np.ndarray, symbols: list[str], tol: float = 0.08) -> int:
    """How many proper rotations map the molecule onto itself, found by trying the rotations that take three
    reference atoms onto atoms of the same elements at the same distances."""
    x = np.asarray(positions, float)
    n = len(x)
    if n == 1:
        return 1
    masses = np.array([_mass(s) for s in symbols])
    com = (masses[:, None] * x).sum(0) / masses.sum()
    x = x - com
    if n == 2 or np.linalg.svd(x, compute_uv=False)[1] < 1e-3:       # linear
        flipped = -x
        return 2 if _maps(flipped, x, symbols, tol) else 1
    radius = np.linalg.norm(x, axis=1)
    classes: dict = {}
    for i, (s, r) in enumerate(zip(symbols, radius)):
        key = (s, round(r / tol))
        classes.setdefault(key, []).append(i)
    order = sorted(range(n), key=lambda i: (len(_same(i, x, symbols, radius, tol)), -radius[i]))
    ref = []
    for i in order:
        if radius[i] < tol:
            continue
        if len(ref) == 0 or (len(ref) == 1 and np.linalg.norm(np.cross(x[ref[0]], x[i])) > 0.1) or \
                (len(ref) == 2 and abs(np.dot(np.cross(x[ref[0]], x[ref[1]]), x[i])) > 0.05):
            ref.append(i)
        if len(ref) == 3:
            break
    if len(ref) < 3:       # planar about the center of mass: the normal makes a third reference
        ref = ref[:2]
        if len(ref) < 2:
            return 1
    found = []
    cands = [_same(i, x, symbols, radius, tol) for i in ref]
    A = x[ref]
    for a in cands[0]:
        for b in cands[1]:
            if b == a or abs(np.linalg.norm(x[a] - x[b]) - np.linalg.norm(A[0] - A[1])) > tol:
                continue
            third = cands[2] if len(ref) == 3 else [None]
            for c in third:
                if c is not None and (c in (a, b) or abs(np.linalg.norm(x[a] - x[c]) - np.linalg.norm(A[0] - A[2]))
                                      > tol or abs(np.linalg.norm(x[b] - x[c]) - np.linalg.norm(A[1] - A[2])) > tol):
                    continue
                src = A if c is not None else np.vstack([A, np.cross(A[0], A[1])])
                dst = x[[a, b, c]] if c is not None else np.vstack([x[[a, b]], np.cross(x[a], x[b])])
                Rm = _kabsch(src, dst)
                if Rm is None or np.linalg.det(Rm) < 0:
                    continue
                y = x @ Rm.T
                if _maps(y, x, symbols, tol) and not any(np.allclose(Rm, f, atol=1e-2) for f in found):
                    found.append(Rm)
    return max(len(found), 1)


def _same(i, x, symbols, radius, tol):
    return [j for j in range(len(x)) if symbols[j] == symbols[i] and abs(radius[j] - radius[i]) < tol]


def _kabsch(src, dst):
    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    if not np.allclose(src @ rot.T, dst, atol=0.15):
        return None
    return rot


def _maps(y, x, symbols, tol) -> bool:
    for i in range(len(y)):
        d = np.linalg.norm(x - y[i], axis=1)
        j = int(np.argmin(d))
        if d[j] > tol or symbols[j] != symbols[i]:
            return False
    return True


def _mass(s: str) -> float:
    from .formula import mass
    return mass(s)


def rrho(freqs_cm: np.ndarray, positions: np.ndarray, symbols: list[str], T: float = 298.15, P: float = 101325.0,
         multiplicity: int = 1, sigma: int | None = None) -> Thermo:
    """Ideal gas thermochemistry. Imaginary and near zero frequencies are left out of the vibrations."""
    masses = np.array([_mass(s) for s in symbols])
    M = masses.sum() * AMU
    n = len(symbols)
    beta = 1.0 / (KB * T)
    # translation
    q_tr = (2 * np.pi * M * KB * T / H ** 2) ** 1.5 * KB * T / P
    S_tr = R * (np.log(q_tr) + 2.5)
    Cp_tr = 2.5 * R
    H_tr = 2.5 * R * T
    # rotation
    linear = n == 2 or (n > 2 and np.linalg.svd(positions - positions.mean(0), compute_uv=False)[1] < 1e-3)
    sigma = sigma if sigma is not None else symmetry_number(positions, symbols)
    if n == 1:
        S_rot = Cp_rot = H_rot = 0.0
    else:
        Imom = moments(positions, masses) * AMU * 1e-20
        theta = H ** 2 / (8 * np.pi ** 2 * Imom * KB)
        if linear:
            S_rot = R * (np.log(T / (sigma * theta[-1])) + 1)
            Cp_rot, H_rot = R, R * T
        else:
            q_rot = np.sqrt(np.pi) / sigma * T ** 1.5 / np.sqrt(np.prod(theta))
            S_rot = R * (np.log(q_rot) + 1.5)
            Cp_rot, H_rot = 1.5 * R, 1.5 * R * T
    # vibration, harmonic with the soft modes blended into free rotors for the entropy
    f = np.asarray([v for v in freqs_cm if v > 1.0], float)
    x = H * C * f * beta
    zpe = (0.5 * H * C * f).sum() * NA
    H_vib = zpe + (H * C * f / np.expm1(x)).sum() * NA
    Cp_vib = (R * x ** 2 * np.exp(x) / np.expm1(x) ** 2).sum()
    S_harm = R * (x / np.expm1(x) - np.log1p(-np.exp(-x)))
    B_av = 1e-44     # kg m^2, Grimme's average moment for the free rotor limit
    mu = H / (8 * np.pi ** 2 * C * np.maximum(f, 1e-6))
    mu_eff = mu * B_av / (mu + B_av)
    S_rotor = R * (0.5 + np.log(np.sqrt(8 * np.pi ** 3 * mu_eff * KB * T / H ** 2)))
    w = 1.0 / (1.0 + (OMEGA0 / np.maximum(f, 1e-6)) ** 4)
    S_vib = float((w * S_harm + (1 - w) * S_rotor).sum())
    S_el = R * np.log(max(multiplicity, 1))
    S = S_tr + S_rot + S_vib + S_el
    Hc = (H_tr + H_rot + H_vib) / 1000.0
    Cp = Cp_tr + Cp_rot + Cp_vib
    return Thermo(T, P, zpe / 1000.0, Hc, S, Hc - T * S / 1000.0, Cp,
                  {"S_translation": S_tr, "S_rotation": S_rot, "S_vibration": S_vib, "S_electronic": S_el,
                   "Cp_translation": Cp_tr, "Cp_rotation": Cp_rot, "Cp_vibration": Cp_vib}, sigma, bool(linear))
