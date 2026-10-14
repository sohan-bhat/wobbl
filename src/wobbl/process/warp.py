"""3D print warping: does a part printed in FDM come off the bed flat?

Each layer of a printed part locks in part of its shrinkage from the glass
transition to the surroundings. The first layers stay warm and held on the bed
and lock in little, and the locked in shrinkage grows with height until it
reaches the full thermal shrinkage about a centimeter up (data.RAMP_HEIGHT,
fitted to measured ABS plates). Wobbl gives every node of a second order
tetrahedral mesh that shrinkage as a thermal strain (shrinking across the
layers too, which bends nothing) and solves the part twice with CalculiX in one
run, with the layers flat in xy and the bed at the lowest z: held on the bed, which
gives the stress the print carries and how hard the corners pull up on the bed,
and then released, which gives the warp after it comes off. A stack of layers is
checked against Timoshenko's bimetal formula, and a plate shaped part against
the same formula for its own shrinkage profile.
"""
from __future__ import annotations

import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from .data import FLATNESS_OFFSET, RAMP_HEIGHT, Filament, get_filament

VERIFY = ("mesh convergence", "hand calculation", "balance on the bed", "units and sanity")
# ISO 2768-2 general flatness tolerance, class L (coarse), by the longest dimension
ISO_2768_L = [(10, 0.1), (30, 0.2), (100, 0.4), (300, 0.8), (1000, 1.2), (3000, 1.6)]
AXES = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


@dataclass
class WarpStudy:
    part: str
    filament: str = "abs"
    build: str = "z"                   # axis or vector the layers stack along, or 'auto'
    bed_C: float | None = None
    chamber_C: float | None = None     # None: an open printer in ambient air
    ambient_C: float = 25.0
    layer_height: float = 0.2          # mm
    infill: float = 1.0                # fraction; the sparse core gets that share of the stiffness
    shell_mm: float = 1.2              # walls, top and bottom skins that stay solid
    brim_mm: float = 0.0
    flatness: float | None = None      # mm, the tolerance; None: ISO 2768-2 class L for the size
    question: str = ""
    quality: str = "normal"
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    mesh_size: float | None = None
    profile: dict = field(default_factory=dict)   # what a slicer file gave, for the assumptions


def iso_flatness(size: float) -> float:
    for top, tol in ISO_2768_L:
        if size <= top:
            return tol
    return ISO_2768_L[-1][1]


def build_direction(text) -> np.ndarray:
    t = str(text).strip().lower().replace(" ", "")
    if t in AXES:
        return np.asarray(AXES[t], float)
    nums = re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", t)
    if len(nums) == 3:
        v = np.asarray([float(x) for x in nums])
        if np.linalg.norm(v) > 0:
            return v / np.linalg.norm(v)
    raise ValueError(f"could not read the print direction '{text}'; give an axis (z, -y, x) or a vector "
                     "like 0,0.6,0.8, or auto")


def to_build_frame(d: np.ndarray) -> np.ndarray:
    """Rotation R with R @ d = +z, so a part printed along d stands on the bed with its layers flat."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(d, z)
    c = float(d @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def shrinkage_profile(z_rel: np.ndarray, eps_max: float, ramp: float = RAMP_HEIGHT) -> np.ndarray:
    """Locked in in plane shrinkage at height z above the bed: nothing at the bed, the full eps_max from ramp up."""
    return eps_max * np.clip(np.asarray(z_rel, float) / ramp, 0.0, 1.0)


def plate_curvature(height: float, profile, n: int = 400) -> float:
    """Curvature of a free plate of this height with in plane shrinkage profile(z): 12 / h^3 times the integral of
    the shrinkage weighted by its distance above the middle (Timoshenko's bimetal argument for any profile)."""
    z = np.linspace(0.0, height, n)
    e = profile(z)
    f = e * (z - height / 2.0)
    return 12.0 / height ** 3 * float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(z)))


def timoshenko_bimetal(t1: float, t2: float, E1: float, E2: float, d_eps: float) -> float:
    """Curvature of two bonded layers whose free shrinkages differ by d_eps (Timoshenko, J. Opt. Soc. Am. 11, 1925):
    kappa = 6 d_eps (1 + m)^2 / (h (3 (1 + m)^2 + (1 + m n) (m^2 + 1 / (m n)))), m = t1 / t2, n = E1 / E2."""
    m, n, h = t1 / t2, E1 / E2, t1 + t2
    return 6 * d_eps * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m * m + 1 / (m * n))))


# --------------------------------------------------------------------------- the solve
@dataclass
class WarpSolve:
    U_bed: np.ndarray                  # (n,3) mm, held on the bed
    U_free: np.ndarray                 # (n,3) mm, off the bed
    S_bed: np.ndarray                  # (n,6) MPa
    S_free: np.ndarray
    RF_bed: np.ndarray                 # (n,3) N, what the bed does to the part
    bed: np.ndarray                    # nodes on the bed
    t_solve: float = 0.0
    solver: str = ""


def bed_nodes(X: np.ndarray, tol: float) -> np.ndarray:
    return np.nonzero(X[:, 2] <= X[:, 2].min() + tol)[0]


def three_points(X: np.ndarray, bed: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    """Three bed nodes that hold a free part still without stopping it from shrinking: the one nearest the middle,
    the one farthest from it, held only square to the line between them and vertically, and the one farthest off
    that line, held vertically. Returns the three and the unit vector along the line."""
    P = X[bed]
    a = bed[int(np.argmin(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1)))]
    d = P[:, :2] - X[a, :2]
    b = bed[int(np.argmax(np.linalg.norm(d, axis=1)))]
    ab = X[b, :2] - X[a, :2]
    if np.linalg.norm(ab) < 1e-9:
        raise ValueError("the face on the bed is too small to rest the part on")
    ab /= np.linalg.norm(ab)
    c = bed[int(np.argmax(np.abs(d @ np.array([-ab[1], ab[0]]))))]
    return int(a), int(b), int(c), np.array([ab[0], ab[1], 0.0])


def solve(X: np.ndarray, tets: np.ndarray, E: np.ndarray, nu: float, shrink: np.ndarray, work: Path,
          bed_tol: float = 1e-3, progress=None, alpha: np.ndarray | None = None) -> WarpSolve:
    """Held on the bed, then released, in one CalculiX run. X is in the build frame (layers in xy, the bed at the
    lowest z). E is per element (MPa) and shrink per node (the strain the layer locked in). The shrinkage acts as a
    thermal strain through an expansion coefficient of 1, scaled per element by alpha (1 unless given), so a sharp
    step between two layers can sit on element faces instead of spreading over nodes."""
    from ..solvers import base as solvers
    from ..solvers.calculix import _nset, _run, _write_mesh, read_frd

    say = progress or (lambda a, b: None)
    ccx = solvers.for_domain("structural").require()
    work.mkdir(parents=True, exist_ok=True)
    bed = bed_nodes(X, bed_tol)
    if len(bed) < 3:
        raise ValueError("fewer than three nodes sit on the bed; the part needs a flat face down")
    a, b, c, line = three_points(X, bed)

    class _M:
        pass

    m = _M()
    m.X, m.tets = X, tets
    alpha = np.ones(len(tets)) if alpha is None else np.asarray(alpha, float)
    groups = {}
    for key in sorted(set(zip(np.asarray(E, float).tolist(), alpha.tolist()))):
        groups[key] = np.nonzero((E == key[0]) & (alpha == key[1]))[0]
    temps = -np.asarray(shrink, float)          # thermal strain = alpha (T - 0)
    lines = "".join(f"{i + 1},{t:.9e}\n" for i, t in enumerate(temps))
    with open(work / "warp.inp", "w") as f:
        f.write("*HEADING\nWobbl print warp\n")
        _write_mesh(f, m)
        for k, ((val, al), el) in enumerate(groups.items()):
            f.write(f"*ELSET, ELSET=E{k + 1}\n")
            ids = el + 1
            for i in range(0, len(ids), 12):
                f.write(", ".join(map(str, ids[i:i + 12])) + "\n")
            f.write(f"*MATERIAL, NAME=M{k + 1}\n*ELASTIC\n{val:.10g},{nu:.10g}\n"
                    f"*EXPANSION, ZERO=0.\n{al:.10g}\n")
            f.write(f"*SOLID SECTION, ELSET=E{k + 1}, MATERIAL=M{k + 1}\n")
        _nset(f, "BED", bed)
        _nset(f, "HOLDB", [b])
        # the second holding point gets axes along the line from the first and square to it, so holding it square
        # to the line leaves it free to slide as the part shrinks toward the first
        f.write("*TRANSFORM, NSET=HOLDB, TYPE=R\n" + ",".join(f"{v:.10g}" for v in line) + "," +
                ",".join(f"{v:.10g}" for v in (-line[1], line[0], 0.0)) + "\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL,0.\n")
        f.write("*STEP\n*STATIC\n*BOUNDARY\nBED,1,3\n*TEMPERATURE\n" + lines +
                "*NODE FILE\nU,RF\n*EL FILE\nS\n*END STEP\n")
        # off the bed: held at three points only, which leaves it free to shrink and curl
        f.write("*STEP\n*STATIC\n*BOUNDARY, OP=NEW\n" + f"{a + 1},1,3\nHOLDB,2,3\n{c + 1},3,3\n")
        f.write("*TEMPERATURE\n" + lines + "*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")
    say("solve", f"CalculiX, {len(X):,} nodes, on the bed and off it")
    log, t = _run(ccx, work, "warp", 3600)
    blocks = read_frd(work / "warp.frd", len(X))
    disp, stress = blocks.get("DISP", []), blocks.get("STRESS", [])
    if len(disp) < 2 or len(stress) < 2:
        raise RuntimeError("CalculiX finished without both steps of the warp solve")
    RF = blocks["FORC"][0][1][:, :3] if "FORC" in blocks else np.zeros((len(X), 3))
    return WarpSolve(disp[0][1][:, :3], disp[-1][1][:, :3], stress[0][1][:, :6], stress[-1][1][:, :6], RF, bed, t,
                     f"CalculiX {solvers.get('calculix').version()}")


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = S.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))


def resting(X: np.ndarray, ws: WarpSolve) -> np.ndarray:
    """The released displacements with the tilt the three holding points leave taken out (the best plane through
    the face that sat on the bed), and lifted so its lowest point touches z = 0, as the part lies on a table."""
    P = X[ws.bed]
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    c, *_ = np.linalg.lstsq(A, ws.U_free[ws.bed, 2], rcond=None)
    U = ws.U_free.copy()
    U[:, 2] -= X[:, 0] * c[0] + X[:, 1] * c[1] + c[2]
    U[:, 2] -= U[ws.bed, 2].min()
    return U


def bottom_curvature(X: np.ndarray, ws: WarpSolve) -> tuple[float, float, float]:
    """Curvatures kxx, kyy and the twist kxy of the face that sat on the bed, from a quadratic surface fitted to its
    vertical displacement after release (a fit over the whole face, so no line of nodes picks up a tilt)."""
    P = X[ws.bed]
    A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1], P[:, 0] ** 2, P[:, 0] * P[:, 1], P[:, 1] ** 2])
    c, *_ = np.linalg.lstsq(A, ws.U_free[ws.bed, 2], rcond=None)
    return float(2 * c[3]), float(2 * c[5]), float(c[4])


def measure(X: np.ndarray, ws: WarpSolve) -> dict:
    """Flatness of the face that sat on the bed after release, from its best fit plane, and how far its corners
    stand above its lowest point."""
    P = X[ws.bed]
    w = P[:, 2] + ws.U_free[ws.bed, 2]
    A = np.column_stack([P[:, 0] + ws.U_free[ws.bed, 0], P[:, 1] + ws.U_free[ws.bed, 1], np.ones(len(P))])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    r = w - A @ coef
    flat = float(r.max() - r.min())
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    corners = []
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            i = int(np.argmin(np.linalg.norm(P[:, :2] - [cx, cy], axis=1)))
            corners.append((float(r[i] - r.min()), X[ws.bed[i]].tolist()))
    lift, where = max(corners)
    return {"flatness_mm": flat, "corner_lift_mm": lift, "lift_xyz": where,
            "low_xyz": X[ws.bed[int(np.argmin(r))]].tolist(), "high_xyz": X[ws.bed[int(np.argmax(r))]].tolist(),
            "residual": r}


def peel(X: np.ndarray, tris: np.ndarray, ws: WarpSolve) -> dict:
    """How hard the part pulls up on the bed while it prints: the bed holds it down where its reaction points down.
    The pull is summed over a patch at each corner of the footprint and spread over that patch's area."""
    pull = np.maximum(-ws.RF_bed[ws.bed, 2], 0.0)
    P = X[ws.bed]
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    r = max(0.15 * float(np.linalg.norm(hi - lo)), 5.0)
    on_bed = np.zeros(len(X), bool)
    on_bed[ws.bed] = True
    bt = tris[on_bed[tris[:, :3]].all(axis=1)][:, :3]
    p = X[bt]
    a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    cen = p.mean(axis=1)[:, :2]
    best = (0.0, 0.0, [float(lo[0]), float(lo[1])])
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            force = float(pull[np.linalg.norm(P[:, :2] - [cx, cy], axis=1) <= r].sum())
            area = float(a[np.linalg.norm(cen - [cx, cy], axis=1) <= r].sum())
            if force > best[0]:
                best = (force, force / max(area, 1e-9), [float(cx), float(cy)])
    return {"peel_total_N": float(pull.sum()), "peel_corner_N": best[0], "peel_corner_MPa": best[1],
            "peel_corner_xy": best[2], "corner_radius_mm": r, "bed_area_mm2": float(a.sum())}
