"""Laminates: the ply stack, classical lamination theory (CLT) and ply failure.

A layup is written the way the trade writes it, plies listed from the inner face out:

    [0/90]s                      0, 90, 90, 0
    [0/±45/90]s                  0, 45, -45, 90, 90, -45, 45, 0
    [0_2/90]2s                   two 0 plies and a 90, twice, then mirrored
    0, 45, -45, 90               plain angles
    [0:im7-8552/45:e-glass-epoxy:0.2]s   a ply's own material and thickness after colons

Angles are in degrees from the part's reference direction, turning toward the second in plane axis.
Strains are engineering strains (the shear is gamma, twice the tensor component) and stresses are MPa.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..materials import PlyMaterial, get_ply

R = np.diag([1.0, 1.0, 2.0])       # engineering shear strain from tensor shear strain
R_INV = np.diag([1.0, 1.0, 0.5])


@dataclass
class Ply:
    angle: float
    material: PlyMaterial
    thickness: float


class LayupError(ValueError):
    pass


@dataclass
class Laminate:
    plies: list[Ply]
    text: str = ""

    @property
    def thickness(self) -> float:
        return float(sum(p.thickness for p in self.plies))

    def z(self) -> np.ndarray:
        """Ply boundaries through the thickness, from the inner face (-h/2) out, mm."""
        return np.concatenate([[0.0], np.cumsum([p.thickness for p in self.plies])]) - 0.5 * self.thickness

    def abd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The A (N/mm), B (N) and D (N mm) stiffness matrices of CLT."""
        z = self.z()
        A, B, D = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        for k, p in enumerate(self.plies):
            Qb = qbar(reduced_stiffness(p.material), p.angle)
            A += Qb * (z[k + 1] - z[k])
            B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2.0
            D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
        return A, B, D

    def compliance(self) -> np.ndarray:
        A, B, D = self.abd()
        return np.linalg.inv(np.block([[A, B], [B, D]]))

    def effective(self) -> dict[str, float]:
        """Engineering constants of the laminate in its own axes: membrane Ex, Ey, Gxy, nuxy and the flexural
        moduli of a beam cut along x or y that is free to twist."""
        a, h = self.compliance(), self.thickness
        return {"Ex": 1.0 / (h * a[0, 0]), "Ey": 1.0 / (h * a[1, 1]), "Gxy": 1.0 / (h * a[2, 2]),
                "nuxy": -a[0, 1] / a[0, 0], "Efx": 12.0 / (h ** 3 * a[3, 3]), "Efy": 12.0 / (h ** 3 * a[4, 4])}

    def symmetric(self) -> bool:
        n = len(self.plies)
        return all(abs(self.plies[k].angle - self.plies[n - 1 - k].angle) < 1e-9 and
                   self.plies[k].material.key == self.plies[n - 1 - k].material.key and
                   abs(self.plies[k].thickness - self.plies[n - 1 - k].thickness) < 1e-12 for k in range(n // 2))

    def balanced(self) -> bool:
        """Every off axis ply has a partner at the opposite angle, so pulling it does not shear it."""
        A, _, _ = self.abd()
        return bool(abs(A[0, 2]) + abs(A[1, 2]) <= 1e-6 * (abs(A[0, 0]) + abs(A[1, 1])))

    def share(self, angle: float) -> float:
        """Fraction of the thickness in plies at this angle (either sign for off axis angles)."""
        t = sum(p.thickness for p in self.plies if abs(abs(_wrap(p.angle)) - abs(_wrap(angle))) < 1e-6)
        return t / self.thickness

    def materials(self) -> list[PlyMaterial]:
        out = []
        for p in self.plies:
            if p.material not in out:
                out.append(p.material)
        return out

    def describe(self) -> str:
        mats = self.materials()
        what = mats[0].name if len(mats) == 1 else ", ".join(m.name for m in mats)
        return f"{self.text or self.angles()}, {len(self.plies)} plies of {what}, {self.thickness:.4g} mm"

    def angles(self) -> str:
        return "/".join(f"{_wrap(p.angle):g}" for p in self.plies)


def _wrap(a: float) -> float:
    """Angle folded into (-90, 90]."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if abs(a + 90.0) < 1e-9 else a


_TOKEN = re.compile(r"^(±|\+-|\+/-|∓|-\+)?\s*([-+]?\d+(?:\.\d+)?)(?:_(\d+))?(?::([a-z0-9][a-z0-9_.-]*?))?"
                    r"(?::(\d*\.?\d+))?$", re.I)


def parse_layup(text: str, ply: PlyMaterial | str, thickness: float | None = None) -> Laminate:
    """Read a layup. ``ply`` is the material of plies that do not name their own, ``thickness`` overrides the
    material's cured ply thickness."""
    default = get_ply(ply) if isinstance(ply, str) else ply
    s = text.strip().replace(" ", "").replace("°", "")
    if not s:
        raise LayupError("the layup is empty; write it like '[0/±45/90]s'")
    reps, sym = 1, False
    m = re.fullmatch(r"\[(.+)\](\d*)(s?)", s, re.I)
    if m:
        body, reps, sym = m.group(1), int(m.group(2) or 1), bool(m.group(3))
    else:
        body = s.strip("[]")
    plies = []
    for tok in re.split(r"[/,;]", body):
        if not tok:
            continue
        mt = _TOKEN.match(tok)
        if not mt:
            raise LayupError(f"could not read the ply '{tok}' in '{text}'. Write plies like 0, 45, ±45, 90, 0_2 or "
                             "0:im7-8552:0.131")
        sign, ang, count, mat, th = mt.groups()
        if mat and th is None and re.fullmatch(r"\d*\.?\d+", mat):   # '0:0.2' gives a thickness, not a material
            mat, th = None, mat
        a = float(ang)
        angs = [a, -a] if sign in ("±", "+-", "+/-") else [-a, a] if sign in ("∓", "-+") else [a]
        material = get_ply(mat) if mat else default
        t = float(th) if th else (thickness or material.thickness)
        if t <= 0:
            raise LayupError(f"ply thickness must be positive in '{tok}'")
        for _ in range(int(count or 1)):
            plies += [Ply(x, material, t) for x in angs]
    plies = plies * reps
    if sym:
        plies = plies + plies[::-1]
    if not plies:
        raise LayupError(f"no plies in '{text}'")
    return Laminate(plies, text.strip())


def reduced_stiffness(m: PlyMaterial) -> np.ndarray:
    """Plane stress stiffness of a ply in its own axes (1 along the fibers), engineering shear."""
    nu21 = m.nu12 * m.E2 / m.E1
    d = 1.0 - m.nu12 * nu21
    return np.array([[m.E1 / d, m.nu12 * m.E2 / d, 0.0], [m.nu12 * m.E2 / d, m.E2 / d, 0.0], [0.0, 0.0, m.G12]])


def _T(angle: float) -> np.ndarray:
    """Stress transformation from laminate axes to ply axes for a ply at this angle."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.array([[c * c, s * s, 2 * c * s], [s * s, c * c, -2 * c * s], [-c * s, c * s, c * c - s * s]])


def qbar(Q: np.ndarray, angle: float) -> np.ndarray:
    """Ply stiffness in laminate axes."""
    T = _T(angle)
    return np.linalg.inv(T) @ Q @ R @ T @ R_INV


def ply_axes_strain(strain_xy: np.ndarray, angle: float) -> np.ndarray:
    """(k,3) engineering strains in laminate axes to ply axes (eps1, eps2, gamma12)."""
    return np.asarray(strain_xy, float) @ (R @ _T(angle) @ R_INV).T


def ply_stress(strain_xy: np.ndarray, ply: Ply) -> np.ndarray:
    """(k,3) stresses in ply axes (sigma1, sigma2, tau12) from laminate axis strains."""
    return ply_axes_strain(strain_xy, ply.angle) @ reduced_stiffness(ply.material).T


def tsai_wu_ratio(sig: np.ndarray, m: PlyMaterial) -> np.ndarray:
    """Strength ratio by Tsai-Wu: the factor on the load at which the ply fails, with F12 = -sqrt(F11 F22)/2."""
    s1, s2, t = np.asarray(sig, float).T
    F1, F2 = 1 / m.Xt - 1 / m.Xc, 1 / m.Yt - 1 / m.Yc
    F11, F22, F66 = 1 / (m.Xt * m.Xc), 1 / (m.Yt * m.Yc), 1 / m.S12 ** 2
    F12 = -0.5 * np.sqrt(F11 * F22)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t ** 2 + 2 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    return _ratio(a, b)


def _ratio(a, b) -> np.ndarray:
    """Positive root of a R^2 + b R = 1, inf where neither term grows with the load."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    disc = np.sqrt(np.maximum(b * b + 4 * a, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(a > 1e-300, (-b + disc) / (2 * np.maximum(a, 1e-300)), np.where(b > 0, 1 / b, np.inf))
    return np.where(np.isfinite(r) & (r > 0), r, np.inf)


HASHIN_MODES = ("fiber tension", "fiber compression", "matrix tension", "matrix compression")


def hashin_ratios(sig: np.ndarray, m: PlyMaterial) -> dict[str, np.ndarray]:
    """Strength ratio of each Hashin (1980) plane stress mode. Matrix compression takes the transverse shear
    strength equal to the in plane one, the usual choice when it is not measured."""
    s1, s2, t = np.asarray(sig, float).T
    inf = np.full(len(s1), np.inf)
    st = m.S12
    shear = (t / m.S12) ** 2
    out = {"fiber tension": np.where(s1 >= 0, _ratio((s1 / m.Xt) ** 2 + shear, 0.0), inf),
           "fiber compression": np.where(s1 < 0, _ratio((s1 / m.Xc) ** 2, 0.0), inf),
           "matrix tension": np.where(s2 >= 0, _ratio((s2 / m.Yt) ** 2 + shear, 0.0), inf),
           "matrix compression": np.where(s2 < 0, _ratio((s2 / (2 * st)) ** 2 + shear,
                                                         ((m.Yc / (2 * st)) ** 2 - 1) * s2 / m.Yc), inf)}
    return out


def max_stress_ratio(sig: np.ndarray, m: PlyMaterial) -> np.ndarray:
    """Strength ratio by the maximum stress criterion, each stress against its own strength."""
    s1, s2, t = np.abs(np.asarray(sig, float)).T
    sg1, sg2 = np.asarray(sig, float)[:, 0], np.asarray(sig, float)[:, 1]
    with np.errstate(divide="ignore"):
        r1 = np.where(sg1 >= 0, m.Xt, m.Xc) / np.maximum(s1, 1e-300)
        r2 = np.where(sg2 >= 0, m.Yt, m.Yc) / np.maximum(s2, 1e-300)
        r3 = m.S12 / np.maximum(t, 1e-300)
    return np.minimum(np.minimum(r1, r2), r3)


def ply_failure(lam: Laminate, k: int, strain_xy: np.ndarray) -> dict:
    """Stresses and strength ratios of ply k at the given laminate axis strains, (n,3) each."""
    p = lam.plies[k]
    sig = ply_stress(strain_xy, p)
    tw = tsai_wu_ratio(sig, p.material)
    hs = hashin_ratios(sig, p.material)
    hmin = np.min(np.vstack([hs[x] for x in HASHIN_MODES]), axis=0)
    mode = np.array(HASHIN_MODES)[np.argmin(np.vstack([hs[x] for x in HASHIN_MODES]), axis=0)]
    return {"stress": sig, "tsai_wu": tw, "hashin": hmin, "hashin_mode": mode}


def clt_first_ply_failure(lam: Laminate, N: np.ndarray, M: np.ndarray | None = None, cylindrical: bool = False
                          ) -> dict:
    """First ply failure of a laminate under force resultants N (N/mm) and moments M (N mm/mm), by CLT.

    Ply strains are taken at each ply's inner and outer face. Returns the governing strength ratio by Tsai-Wu
    and by Hashin, which ply and mode set each, and the ply by ply table. With cylindrical, the laminate bends
    only about its y axis, as a plate held along an edge does (kappa_y = kappa_xy = 0), instead of curling freely
    across the way a narrow beam does.
    """
    M = np.zeros(3) if M is None else np.asarray(M, float)
    if cylindrical:
        ABD = np.linalg.inv(lam.compliance())
        free = [0, 1, 2, 3]                                   # midplane strains and kappa_x; the rest held at zero
        e = np.zeros(6)
        e[free] = np.linalg.solve(ABD[np.ix_(free, free)], np.concatenate([np.asarray(N, float), M])[free])
    else:
        e = lam.compliance() @ np.concatenate([np.asarray(N, float), M])
    z = lam.z()
    rows = []
    for k, p in enumerate(lam.plies):
        strains = np.array([e[:3] + z[k] * e[3:], e[:3] + z[k + 1] * e[3:]])
        f = ply_failure(lam, k, strains)
        j = int(np.argmin(f["hashin"]))
        rows.append({"ply": k + 1, "angle": _wrap(p.angle), "material": p.material.key,
                     "stress": f["stress"][int(np.argmin(f["tsai_wu"]))].tolist(),
                     "tsai_wu": float(f["tsai_wu"].min()), "hashin": float(f["hashin"].min()),
                     "hashin_mode": str(f["hashin_mode"][j])})
    tw = min(rows, key=lambda r: r["tsai_wu"])
    hs = min(rows, key=lambda r: r["hashin"])
    return {"tsai_wu": tw["tsai_wu"], "tsai_wu_ply": tw["ply"], "hashin": hs["hashin"], "hashin_ply": hs["ply"],
            "hashin_mode": hs["hashin_mode"], "plies": rows, "midplane": e.tolist()}


__all__ = ["Ply", "Laminate", "LayupError", "parse_layup", "reduced_stiffness", "qbar", "ply_axes_strain",
           "ply_stress", "tsai_wu_ratio", "hashin_ratios", "max_stress_ratio", "ply_failure",
           "clt_first_ply_failure", "HASHIN_MODES"]
