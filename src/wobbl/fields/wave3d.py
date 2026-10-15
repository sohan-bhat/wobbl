"""Full wave antennas on any CAD: impedance, S11, radiated power and the gain pattern from a 3D solve.

Metal bodies are perfect conductors cut out of the mesh, dielectric bodies are meshed with their permittivity
and loss, and planar copper (a PCB's traces and ground) comes in as zero thickness sheets. A lumped port is a
small strip across the feed gap that carries 1 A of surface current. GetDP solves the time harmonic electric
field with edge elements in a sphere of air whose surface absorbs outgoing waves (a second order boundary
condition), frequency by frequency, on one mesh sized for the top of the sweep.

The voltage across the port gives the input impedance and S11. The fields on a sphere around the antenna give
the power flowing out, which must match the power fed in less the dielectric loss, and, turned into equivalent
currents, the far field, the directivity and the gain. A normal run repeats the working frequency on a finer
mesh and with the boundary farther out, and a straight wire dipole or a monopole over a plate is compared with
the induced EMF theory.
"""
from __future__ import annotations

import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..mesh.isolate import IsolatedTimeout
from ..units import fmt
from ..verify.checks import Check, confidence
from . import plots, pro3d, solve
from .forms import C0, EPS0, MU0, rname
from .pro3d import num
from .scene import Domain, Scene, SceneError, mesh

ETA0 = MU0 * C0
QUALITY = {"fast": 1.3, "normal": 1.0, "fine": 0.8}   # element size factor
GROWTH = 0.35         # how fast the air mesh coarsens away from a surface
FACES = 3000          # surface elements a body or a sheet gets before its element size grows
LINE_POINTS = 40      # samples of E across the port gap


@dataclass
class Port:
    """A lumped feed across a gap: 1 A flows from a to b (points in m) on a strip of the given width (m)."""
    a: tuple
    b: tuple
    width: float = 0.0
    z0: float = 50.0


@dataclass
class WaveSetup:
    metals: list = field(default_factory=list)          # body names, perfect conductors cut out of the mesh
    dielectrics: dict = field(default_factory=dict)     # body: (er, loss tangent, material name)
    sheets: dict = field(default_factory=dict)          # name: list of planar polygons (N, 3) in m, copper
    port: Port | None = None
    f0: float = 0.0                                     # working frequency, Hz
    span: float = 0.25                                  # the sweep runs f0 * (1 - span) to f0 * (1 + span)
    points: int = 9
    link: dict = field(default_factory=dict)            # tx_power, sensitivity, environment, distance and the like
    tolerance: float = 0.02                             # resonance within this fraction of f0 is on target
    sizes: dict = field(default_factory=dict)           # body or sheet name: element size at its surface, m
    refine: int = 4                                     # extra solves that home in on the resonance and any sharp bend
    hints: list = field(default_factory=list)           # frequencies a simple model expects structure at, sampled too


# --- port geometry -------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def port_width(a, b, width: float = 0.0) -> float:
    """The strip's width: as given, or a tenth of the gap and at least 0.2 mm."""
    if width and width > 0:
        return float(width)
    return max(0.1 * float(np.linalg.norm(np.subtract(b, a))), 0.2e-3)


def port_polygon(a, b, width: float, normal=None) -> np.ndarray:
    """The port strip as a rectangle (4, 3) holding the line a to b. Its width runs perpendicular to the line
    and to the given normal (the nearest metal's), or along any perpendicular when there is none."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length <= 0:
        raise ValueError("the port's two ends are the same point. Give a and b on the two conductors across the "
                         "feed gap")
    t = (b - a) / length
    u = None
    if normal is not None and np.linalg.norm(normal) > 0:
        c = np.cross(t, _unit(normal))
        if np.linalg.norm(c) > 0.1:
            u = _unit(c)
    if u is None:
        u = _unit(np.cross(t, np.eye(3)[int(np.argmin(np.abs(t)))]))
    h = 0.5 * width * u
    return np.array([a - h, a + h, b + h, b - h])


def polygon_normal(poly) -> np.ndarray:
    """A planar polygon's unit normal (Newell's method)."""
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    n = np.array([np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
                  np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
                  np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))])
    return _unit(n)


def _polygon_area(poly) -> float:
    p = np.asarray(poly, float)
    return 0.5 * float(np.linalg.norm(np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)))


def closest_on_polygon(poly, p) -> tuple[float, np.ndarray]:
    """Distance from a point to a planar polygon and the nearest point on it."""
    P = np.asarray(poly, float)
    p = np.asarray(p, float)
    n = polygon_normal(P)
    q = p - float(np.dot(p - P[0], n)) * n
    e1 = _unit(P[1] - P[0])
    e2 = np.cross(n, e1)
    uv = np.stack([(P - P[0]) @ e1, (P - P[0]) @ e2], axis=1)
    x, y = float((q - P[0]) @ e1), float((q - P[0]) @ e2)
    inside = False
    for i in range(len(uv)):
        (x1, y1), (x2, y2) = uv[i], uv[(i + 1) % len(uv)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    if inside:
        return float(np.linalg.norm(p - q)), q
    best = (math.inf, q)
    for i in range(len(P)):
        s0, s1 = P[i], P[(i + 1) % len(P)]
        d = s1 - s0
        t = float(np.clip(np.dot(p - s0, d) / max(np.dot(d, d), 1e-300), 0.0, 1.0))
        c = s0 + t * d
        dist = float(np.linalg.norm(p - c))
        if dist < best[0]:
            best = (dist, c)
    return best


def _closest_on_bodies(brep: str, indices: list[int], points: list) -> list:
    """For each point, the nearest point on the surfaces of the given bodies: (distance, point, normal, index)."""
    from .scene import _session

    gmsh = _session()
    try:
        ents = gmsh.model.occ.importShapes(brep, highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        surfs = {}
        for i in indices:
            for _, s in gmsh.model.getBoundary([(3, vols[i - 1])], oriented=False):
                surfs[abs(s)] = i
        out = []
        for p in points:
            best = None
            for s, i in surfs.items():
                c, par = gmsh.model.getClosestPoint(2, s, [float(x) for x in p])
                d = math.dist(c, p)
                if best is None or d < best[0]:
                    best = (d, tuple(float(x) for x in c), tuple(float(x) for x in gmsh.model.getNormal(s, par)), i)
            out.append(best)
        return out
    finally:
        gmsh.finalize()


def _nearest_metal(scene: Scene, s: WaveSetup, points: list) -> list:
    """For each point the nearest conductor: (distance, point, normal, (name, polygon index or -1))."""
    from ..mesh.isolate import run_isolated

    best: list = [None] * len(points)
    for name, polys in s.sheets.items():
        for j, poly in enumerate(polys):
            n = polygon_normal(poly)
            for k, p in enumerate(points):
                d, q = closest_on_polygon(poly, p)
                if best[k] is None or d < best[k][0]:
                    best[k] = (d, q, n, (name, j))
    idx = [b.index for b in scene.bodies if b.name in s.metals]
    if idx:
        names = {b.index: b.name for b in scene.bodies}
        for k, r in enumerate(run_isolated(_closest_on_bodies, scene.brep, idx, [list(p) for p in points],
                                           timeout=300)):
            if r and (best[k] is None or r[0] < best[k][0]):
                best[k] = (r[0], np.array(r[1]), np.array(r[2]), (names[r[3]], -1))
    return best


def feed(scene: Scene, s: WaveSetup) -> dict:
    """The port on the metal: its ends (moved onto the nearest conductor when a little off), its strip and the
    direction its current flows."""
    a, b = np.asarray(s.port.a, float), np.asarray(s.port.b, float)
    gap = float(np.linalg.norm(b - a))
    if gap <= 0:
        raise ValueError("the port's two ends are the same point. Give a and b on the two conductors across the "
                         "feed gap")
    near = _nearest_metal(scene, s, [a, b, 0.5 * (a + b)])
    ends, notes = [], []
    for p, nm, label in zip((a, b), near[:2], ("a", "b")):
        if nm is None:
            raise ValueError("the antenna has no metal. Mark its conductors as metals or give copper sheets")
        d, q, _, (name, _) = nm
        if d > 0.3 * gap + 1e-6:
            raise ValueError(f"the port end {label} at {_mm(p)} mm is {fmt(d * 1e3)} mm from the nearest metal "
                             f"('{name}'). Put both ends of the port on the conductors across the feed gap")
        if d > 1e-3 * gap:
            notes.append(f"The port end {label} moved {fmt(d * 1e3)} mm onto '{name}'.")
            p = np.asarray(q, float)
        ends.append(p)
    if near[0][3] == near[1][3] and (near[0][3][1] >= 0 or near[2][0] < 0.3 * gap):
        raise ValueError(f"both ends of the port touch '{near[0][3][0]}'. The port must bridge the gap between two "
                         "separate conductors")
    a, b = ends
    width = port_width(a, b, s.port.width)
    normal = near[2][2] if near[2] is not None else None
    return {"a": a, "b": b, "t": _unit(b - a), "gap": float(np.linalg.norm(b - a)), "width": width,
            "poly": port_polygon(a, b, width, normal), "I0": 1.0, "notes": notes}


def _mm(p) -> str:
    return ", ".join(f"{1e3 * x:.4g}" for x in p)


# --- sizes and the air ------------------------------------------------------------------------------
def _box_touch(b0, b1, tol: float) -> bool:
    return all(b0[i] <= b1[i + 3] + tol and b1[i] <= b0[i + 3] + tol for i in range(3))


def element_sizes(scene: Scene, s: WaveSetup, lam_min: float, sheets: dict, gap: float) -> dict:
    """Element size at each conductor and dielectric surface: fine enough for the shape (a third of a wire's
    diameter, half a slab's thickness), never coarser than a twentieth of the wavelength there, and no finer
    than its area spread over a few thousand elements."""
    out = {}
    diel_boxes = []
    for b in scene.bodies:
        side = sorted(b.bbox[i + 3] - b.bbox[i] for i in range(3))
        if b.name in s.dielectrics:
            er = float(s.dielectrics[b.name][0])
            out[b.name] = min(lam_min / (10 * math.sqrt(er)), max(side[0] / 2, math.sqrt(b.area / FACES)))
            diel_boxes.append((b.bbox, er, side[0]))
        elif b.name in s.metals:
            out[b.name] = min(lam_min / 20, max(side[0] / 3, math.sqrt(b.area / FACES)))
    for name, polys in sheets.items():
        pts = np.concatenate([np.asarray(p, float) for p in polys])
        box = tuple(pts.min(0)) + tuple(pts.max(0))
        area = sum(_polygon_area(p) for p in polys)
        perim = sum(float(np.sum(np.linalg.norm(np.asarray(p) - np.roll(p, -1, axis=0), axis=1))) for p in polys)
        width = 2 * area / max(perim, 1e-30)
        near = [d for d in diel_boxes if _box_touch(d[0], box, 1e-6)]
        er = max([1.0] + [d[1] for d in near])
        h = min(lam_min / (20 * math.sqrt(er)), max(width / 4, math.sqrt(area / FACES)))
        if near:
            h = min(h, min(d[2] for d in near))
        out[name] = h
    out["port"] = min(gap / 3, lam_min / 20)
    out.update(s.sizes)
    return out


def _extent(scene: Scene, s: WaveSetup, sheets: dict) -> tuple[np.ndarray, float]:
    """The scene's center (where the air sphere sits) and the radius around it that holds the antenna."""
    c = np.array(scene.center, float)
    pts = []
    for b in scene.bodies:
        if b.name in s.metals or b.name in s.dielectrics:
            lo, hi = b.bbox[:3], b.bbox[3:]
            pts += [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    for polys in sheets.values():
        for p in polys:
            pts += [tuple(x) for x in np.asarray(p, float)]
    return c, float(max(np.linalg.norm(np.asarray(pts) - c, axis=1)))


def sphere_grid(n_theta: int) -> dict:
    """Sample points on a sphere: Gauss Legendre in cos theta and evenly spaced in phi, with their weights."""
    x, w = np.polynomial.legendre.leggauss(int(n_theta))
    order = np.argsort(-x)
    theta = np.arccos(x[order])
    n_phi = 2 * int(n_theta)
    return {"theta": theta, "phi": np.arange(n_phi) * 2 * math.pi / n_phi, "w": w[order],
            "dphi": 2 * math.pi / n_phi}


# --- the GetDP problem -----------------------------------------------------------------------------
def problem(msh, s: WaveSetup, f: float, port: dict, line: tuple, sphere: dict, plane: dict | None,
            abc: int = 2) -> str:
    """The time harmonic electric field (e^(j w t)) with edge elements, perfect conductors as tangential E = 0, an
    absorbing sphere and the port's surface current as the source."""
    vols = [n for n, t in msh.tags.items() if t < 100]
    pec = [n for n in list(s.metals) + list(s.sheets) if n in msh.tags]
    out = [pro3d.groups(msh.tags, {"Dom": vols, "Pec": pec, "Port": ["port"], "Out": ["outer"],
                                   "All_": vols + ["port", "outer"]})]
    w = 2 * math.pi * f
    js = np.asarray(port["t"]) * port["I0"] / port["width"]
    fn = [f"  mu0 = {num(MU0)};", f"  eps0 = {num(EPS0)};", f"  w = {num(w)};", f"  I0 = {num(port['I0'])};",
          f"  Js[] = Vector[{num(js[0])}, {num(js[1])}, {num(js[2])}];",
          f"  beta[] = 1 / Complex[{num(2 / msh.outer)}, {num(2 * w / C0)}];"]
    for n in vols:
        er, tand = (s.dielectrics[n][0], s.dielectrics[n][1]) if n in s.dielectrics else (1.0, 0.0)
        fn.append(f"  epsr[{rname(n)}] = Complex[{num(er)}, {num(-er * tand)}];")
    out.append("Function {\n" + "\n".join(fn) + "\n}\n")
    out.append(pro3d.jacobian(msh, False))
    out.append(pro3d.INTEGRATION)
    out.append("FunctionSpace {\n  { Name Hcurl; Type Form1; BasisFunction {\n"
               "      { Name se; NameOfCoef ee; Function BF_Edge; Support All_; Entity EdgesOf[All, Not Pec]; } }\n"
               "  }\n" + ("  { Name Hs; Type Form0; BasisFunction {\n"
                         "      { Name sn; NameOfCoef pn; Function BF_Node; Support Out; Entity NodesOf[All]; } }\n"
                         "  }\n" if abc == 2 else "") + "}\n")
    # n x curl E = jk E_t + beta curl(n (n . curl E)) + beta grad_s(div_s E_t), beta = 1 / (2 j k + 2 / R), the
    # second order boundary of Webb and Kanellopoulos. The surface divergence enters through an auxiliary nodal
    # unknown p on the sphere, which keeps the system symmetric.
    abc2 = ("      Integral { [ beta[] / mu0 * Dof{d e}, {d e} ]; In Out; Jacobian Sur; Integration I1; }\n"
            "      Integral { [ beta[] / mu0 * Dof{d p}, {e} ]; In Out; Jacobian Sur; Integration I1; }\n"
            "      Integral { [ beta[] / mu0 * Dof{e}, {d p} ]; In Out; Jacobian Sur; Integration I1; }\n"
            "      Integral { [ beta[] / mu0 * Dof{p}, {p} ]; In Out; Jacobian Sur; Integration I1; }\n"
            if abc == 2 else "")
    out.append("Formulation {\n  { Name W; Type FemEquation;\n    Quantity { "
               "{ Name e; Type Local; NameOfSpace Hcurl; } "
               + ("{ Name p; Type Local; NameOfSpace Hs; } " if abc == 2 else "") + "}\n"
               "    Equation {\n"
               "      Integral { [ 1 / mu0 * Dof{d e}, {d e} ]; In Dom; Jacobian Vol; Integration I1; }\n"
               "      Integral { [ -w^2 * eps0 * epsr[] * Dof{e}, {e} ]; In Dom; Jacobian Vol; Integration I1; }\n"
               "      Integral { [ -Complex[0, w * Sqrt[eps0 / mu0]] * (Normal[] /\\ (Normal[] /\\ Dof{e})), {e} ]; "
               "In Out; Jacobian Sur; Integration I1; }\n"
               "      Integral { [ Complex[0, w] * Js[], {e} ]; In Port; Jacobian Sur; Integration I1; }\n"
               + abc2 + "    }\n  }\n}\n")
    out.append(f"Resolution {{\n  {{ Name W; System {{ {{ Name A; NameOfFormulation W; Type Complex; "
               f"Frequency {num(f)}; }} }}\n    Operation {{ Generate[A]; Solve[A]; SaveSolution[A]; }}\n  }}\n}}\n")
    out.append("PostProcessing {\n  { Name W; NameOfFormulation W; Quantity {\n"
               "      { Name e; Value { Local { [ {e} ]; In All_; Jacobian Vol; } } }\n"
               "      { Name h; Value { Local { [ Complex[0, 1 / (w * mu0)] * {d e} ]; In Dom; Jacobian Vol; } } }\n"
               "      { Name Vavg; Value { Integral { [ -({e} * Js[]) / I0 ]; In Port; Jacobian Sur; "
               "Integration I1; } } }\n"
               "      { Name Pdiel; Value { Integral { [ -0.5 * w * eps0 * Im[epsr[]] * SquNorm[{e}] ]; In Dom; "
               "Jacobian Vol; Integration I1; } } }\n"
               "  } }\n}\n")
    pa, pb, n = line
    po = ['Print[ Vavg[Port], OnGlobal, Format Table, File "vavg.txt" ];',
          'Print[ Pdiel[Dom], OnGlobal, Format Table, File "pdiel.txt" ];',
          f'Print[ e, OnLine {{{_pt(pa)}{_pt(pb)}}} {{{int(n) - 1}}}, Format Table, File "vline.txt" ];']
    c, rs = sphere["center"], sphere["radius"]
    grid = (f"OnGrid {{{num(c[0])} + {num(rs)} * Sin[$A] * Cos[$B], {num(c[1])} + {num(rs)} * Sin[$A] * Sin[$B], "
            f"{num(c[2])} + {num(rs)} * Cos[$A]}} {{ {{{', '.join(num(x) for x in sphere['theta'])}}}, "
            f"{{{', '.join(num(x) for x in sphere['phi'])}}}, {{0}} }}")
    po.append(f'Print[ e, {grid}, Format Table, File "sph_e.txt" ];')
    po.append(f'Print[ h, {grid}, Format Table, File "sph_h.txt" ];')
    if plane:
        po.append(pro3d.print_plane("e", plane, "slice_e.txt"))
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing W; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)


def _pt(p) -> str:
    return "{" + ", ".join(num(x) for x in p) + "}"


# --- reading results -------------------------------------------------------------------------------
def _rows(path: Path, min_cols: int) -> list[list[float]]:
    out = []
    for ln in Path(path).read_text().splitlines():
        t = ln.split()
        if len(t) >= min_cols:
            out.append([float(v) for v in t])
    return out


def read_line(path: Path) -> np.ndarray:
    """A complex vector printed OnLine in Table format: (n, 3), the values are the last six columns."""
    r = np.array(_rows(path, 8))
    return r[:, -6:-3] + 1j * r[:, -3:]


def read_grid(path: Path, sph: dict) -> np.ndarray:
    """A complex vector printed OnGrid on the sphere: (n_theta, n_phi, 3), NaN where a point was not found."""
    th, ph = sph["theta"], sph["phi"]
    out = np.full((len(th), len(ph), 3), np.nan + 0j)
    for r in _rows(path, 14):
        i = int(np.argmin(np.abs(th - r[5])))
        j = int(np.argmin(np.abs(ph - r[6])))
        out[i, j] = np.array(r[-6:-3]) + 1j * np.array(r[-3:])
    return out


# --- far field ------------------------------------------------------------------------------------
def _dirs(theta, phi) -> np.ndarray:
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)


def sphere_flux(E: np.ndarray, H: np.ndarray, sph: dict, rs: float) -> float:
    """Time average power out through the sampling sphere, 0.5 Re of the integral of (E x conj H) . n."""
    n = _dirs(sph["theta"], sph["phi"])
    s = 0.5 * np.real(np.sum(np.cross(E, np.conj(H)) * n, axis=-1))
    return float(np.nansum(s * sph["w"][:, None]) * sph["dphi"] * rs ** 2)


def far_field(E: np.ndarray, H: np.ndarray, sph: dict, rs: float, k: float, dirs: np.ndarray) -> np.ndarray:
    """The far field vector F(r) = eta N_t + L x r of the equivalent currents J = n x H and M = -n x E on the
    sampling sphere, for unit directions dirs (K, 3). The radiation intensity is k^2 |F|^2 / (32 pi^2 eta)."""
    n = _dirs(sph["theta"], sph["phi"]).reshape(-1, 3)
    ok = np.all(np.isfinite(E.reshape(-1, 3)), axis=1) & np.all(np.isfinite(H.reshape(-1, 3)), axis=1)
    J = np.cross(n, H.reshape(-1, 3))[ok]
    M = -np.cross(n, E.reshape(-1, 3))[ok]
    dS = np.repeat(sph["w"], len(sph["phi"]))[ok] * sph["dphi"] * rs ** 2
    pts = rs * n[ok]
    dirs = np.asarray(dirs, float).reshape(-1, 3)
    out = np.zeros((len(dirs), 3), complex)
    step = max(1, 2_000_000 // max(len(pts), 1))
    for i in range(0, len(dirs), step):
        d = dirs[i:i + step]
        ph = np.exp(1j * k * (d @ pts.T)) * dS
        N, L = ph @ J, ph @ M
        out[i:i + step] = ETA0 * (N - np.sum(N * d, axis=1)[:, None] * d) + np.cross(L, d)
    return out


def intensity(F: np.ndarray, k: float) -> np.ndarray:
    return k ** 2 * np.sum(np.abs(F) ** 2, axis=-1) / (32 * math.pi ** 2 * ETA0)


def pattern(E: np.ndarray, H: np.ndarray, sph: dict, rs: float, k: float, axis=None, ref=None,
            cuts: bool = True) -> dict:
    """Radiated power of the far field, peak directivity and its direction, and the elevation and azimuth cuts
    about axis (the feed direction) as directivity (not yet scaled by efficiency)."""
    g = sphere_grid(max(24, len(sph["theta"])))
    Ug = intensity(far_field(E, H, sph, rs, k, _dirs(g["theta"], g["phi"])), k).reshape(len(g["theta"]), -1)
    p_ff = float(np.sum(Ug * g["w"][:, None]) * g["dphi"])
    th = np.radians(np.arange(0, 181, 2.5))
    phs = np.radians(np.arange(0, 360, 5.0))
    dirs = _dirs(th, phs).reshape(-1, 3)
    U = intensity(far_field(E, H, sph, rs, k, dirs), k)
    i = int(np.argmax(U))
    peak = dirs[i]
    out = {"p_ff": p_ff, "d_max": 4 * math.pi * float(U[i]) / max(p_ff, 1e-300), "peak": peak,
           "theta_deg": float(np.degrees(np.arccos(np.clip(peak[2], -1, 1)))),
           "phi_deg": float(np.degrees(np.arctan2(peak[1], peak[0])) % 360)}
    if cuts:
        out["d_sphere"] = (4 * math.pi * U / max(p_ff, 1e-300)).reshape(len(th), len(phs))   # theta by phi
        out["sphere_theta"], out["sphere_phi"] = th, phs
        t = _unit(axis if axis is not None else (0, 0, 1))
        p = peak - np.dot(peak, t) * t
        if np.linalg.norm(p) < 0.1:
            p = np.asarray(ref, float) if ref is not None else np.cross(t, np.eye(3)[int(np.argmin(np.abs(t)))])
            p = p - np.dot(p, t) * t
        p = _unit(p)
        q = np.cross(t, p)
        ang = np.radians(np.arange(0, 361, 1.0))
        el = np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * p
        az = np.cos(ang)[:, None] * p + np.sin(ang)[:, None] * q
        out["angles"] = ang
        out["d_el"] = 4 * math.pi * intensity(far_field(E, H, sph, rs, k, el), k) / p_ff
        out["d_az"] = 4 * math.pi * intensity(far_field(E, H, sph, rs, k, az), k) / p_ff
        out["axes"] = (t, p, q)
    return out


# --- one frequency -----------------------------------------------------------------------------------
def _solve(msh, s: WaveSetup, f: float, fd: dict, sph: dict, d: Path, plane: dict | None = None,
           cuts: bool = False) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    t_wall, t_cpu = time.time(), _cpu()
    n = LINE_POINTS
    dl = fd["gap"] / n
    line = (fd["a"] + 0.5 * dl * fd["t"], fd["b"] - 0.5 * dl * fd["t"], n)
    lg = solve.run(problem(msh, s, f, fd, line, sph, plane), msh.path, d, resolution="W", verbosity=4,
                   timeout=7200)
    hits = re.findall(r"(\d+) Dofs", lg)
    e = read_line(d / "vline.txt")
    if len(e) < n // 2:
        raise ValueError("the port gap could not be sampled, which is a Wobbl bug. Report it with the model")
    V = -complex(np.sum(e @ fd["t"])) * fd["gap"] / len(e)
    I0 = fd["I0"]
    Z = V / I0
    va = solve.read_global(d, "vavg.txt")
    E, H = read_grid(d / "sph_e.txt", sph), read_grid(d / "sph_h.txt", sph)
    k = 2 * math.pi * f / C0
    pat = pattern(E, H, sph, sph["radius"], k, fd["t"], fd["poly"][1] - fd["poly"][0], cuts=cuts)
    out = {"f": f, "Z": Z, "Z_avg": complex(va[0], va[1] if len(va) > 1 else 0.0) / I0,
           "p_in": 0.5 * (V * np.conj(I0)).real, "p_diel": float(solve.read_global(d, "pdiel.txt")[0]),
           "p_rad": sphere_flux(E, H, sph, sph["radius"]), "dofs": int(hits[-1]) if hits else 0,
           "missing": int(np.isnan(E[..., 0].real).sum()), **pat}
    out["wall"], out["cpu"] = time.time() - t_wall, _cpu() - t_cpu
    if plane:
        out["slice"] = pro3d.read_plane(d / "slice_e.txt", plane, comps=6)
    return out


def mesh_inputs(scene: Scene, s: WaveSetup, quality: str = "normal") -> dict:
    """What sets the size of the base mesh, from the geometry alone: the air sphere's radius and far element size
    and the area and element size of each conductor and dielectric surface. wavecost turns it into a node count
    without meshing, which is how a study is priced before its mesh is worth building."""
    k = QUALITY[quality]
    freqs = _sweep(s, s.points)
    lam_min, lam_max = C0 / max(freqs), C0 / min(freqs)
    fd = feed(scene, s)
    sheets = {**{n: [np.asarray(p, float) for p in ps] for n, ps in s.sheets.items()}, "port": [fd["poly"]]}
    _, r_ext = _extent(scene, s, sheets)
    R = r_ext + lam_max / 2
    sizes = element_sizes(scene, s, lam_min, sheets, fd["gap"])
    surf, wires = [], []
    for b in scene.bodies:
        if b.name in sizes and (b.name in s.metals or b.name in s.dielectrics):
            surf.append((b.area, sizes[b.name] * k))
            side = sorted(b.bbox[i + 3] - b.bbox[i] for i in range(3))
            if b.name in s.metals and side[2] > 8 * side[0]:
                wires.append((b.area / (math.pi * max(side[0], 1e-9)), sizes[b.name] * k))   # length of a round wire
    for n, polys in sheets.items():
        surf.append((sum(_polygon_area(p) for p in polys), sizes.get(n, lam_min / 20) * k))
    return {"R": R, "far": min(lam_min / 10, R / 5) * k, "surfaces": surf, "wires": wires}


FINE_STEP = 1 / (1 - 0.75 ** 2)       # coarse error over the change to a mesh 0.75 as large, for an error of h squared


def mesh_shift(freqs, Z, f0: float, z_coarse: complex, z_fine: complex) -> float | None:
    """How far, as a fraction of f0, a resonance moves up in frequency when the mesh gets finer. The finer mesh's
    impedance at f0 is the coarse mesh's at a slightly different frequency, found from the impedance's slope there
    along the sweep. None when the sweep has no samples to take a slope from."""
    fs = list(freqs)
    if len(fs) < 2:
        return None
    i = min(max((j for j in range(len(fs)) if fs[j] <= f0 * (1 + 1e-9)), default=0), len(fs) - 2)
    slope = (Z[i + 1] - Z[i]) / (fs[i + 1] - fs[i])
    if abs(slope) < 1e-30:
        return None
    d = complex(z_fine - z_coarse)
    delta = (d * slope.conjugate()).real / abs(slope) ** 2       # the coarse mesh's frequency that looks like the fine
    return float(-delta / f0)


def _peak_memory() -> float:
    """The largest resident size of a solver this process has run, in bytes."""
    import resource
    import sys

    m = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return float(m if sys.platform == "darwin" else m * 1024)


def _cpu() -> float:
    """CPU seconds this process and its children have used, which does not grow when the machine is busy."""
    import resource

    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _gain(r: dict) -> float:
    """Peak gain: directivity times the share of the input power that is radiated rather than lost."""
    eff = min(max((r["p_in"] - r["p_diel"]) / r["p_in"], 1e-6), 1.0) if r["p_in"] > 0 else 1e-6
    return r["d_max"] * eff


# --- closed forms ------------------------------------------------------------------------------------
def _cylinder(b) -> dict | None:
    """A straight round rod: its axis (0, 1, 2 for x, y, z), radius, ends and center. Measured from the volume, the
    area and the length along the box's longest side, since a curved surface's box can be loose."""
    side = [b.bbox[i + 3] - b.bbox[i] for i in range(3)]
    ax = int(np.argmax(side))
    length = side[ax]
    a = math.sqrt(max(b.volume, 0.0) / (math.pi * length)) if length > 0 else 0.0
    if a <= 0 or length < 10 * a or abs(b.area - 2 * math.pi * a * (length + a)) > 0.02 * b.area:
        return None
    if any(not 1.9 * a <= side[i] <= 4 * a for i in range(3) if i != ax):
        return None
    return {"axis": ax, "a": a, "lo": b.bbox[ax], "hi": b.bbox[ax + 3], "c": list(b.center)}


def wire_geometry(scene: Scene, s: WaveSetup, t, f0: float) -> dict | None:
    """A straight thin wire dipole (two collinear cylinders fed across a gap), or a monopole on a plate (a thin box
    or one copper polygon) fed from the plate, when the antenna is one. None otherwise."""
    if s.dielectrics:
        return None
    t = _unit(t)
    metals = [b for b in scene.bodies if b.name in s.metals]
    cyl = [(b, _cylinder(b)) for b in metals]
    rods = [(b, c) for b, c in cyl if c and abs(t[c["axis"]]) > 0.999]
    if len(metals) == 2 and len(rods) == 2 and not s.sheets:
        (_, c1), (_, c2) = rods
        ax = c1["axis"]
        tr = [i for i in range(3) if i != ax]
        if c2["axis"] != ax or abs(c1["a"] - c2["a"]) > 0.03 * c1["a"] or \
                max(abs(c1["c"][i] - c2["c"][i]) for i in tr) > c1["a"]:
            return None
        length = max(c1["hi"], c2["hi"]) - min(c1["lo"], c2["lo"])
        a = 0.5 * (c1["a"] + c2["a"])
        return {"kind": "dipole", "length": length, "a": a, "axis": ax} if a < 0.05 * length / 2 else None
    if len(rods) != 1:
        return None
    (rod, c) = rods[0]
    ax = c["axis"]
    plates = []
    for b in metals:
        if b is rod:
            continue
        side = [b.bbox[i + 3] - b.bbox[i] for i in range(3)]
        if int(np.argmin(side)) == ax and side[ax] < 0.1 * min(side[i] for i in range(3) if i != ax) and \
                abs(b.volume - side[0] * side[1] * side[2]) < 0.02 * b.volume:
            plates.append((b.bbox[ax], b.bbox[ax + 3], min(side[i] for i in range(3) if i != ax)))
    for polys in s.sheets.values():
        if len(polys) == 1 and abs(polygon_normal(polys[0])[ax]) > 0.999:
            p = np.asarray(polys[0], float)
            ext = p.max(0) - p.min(0)
            plates.append((float(p[0, ax]), float(p[0, ax]), min(ext[i] for i in range(3) if i != ax)))
    if len(plates) != 1 or len(metals) + len(s.sheets) != 2:
        return None
    lo, hi, size = plates[0]
    h = c["hi"] - hi if c["lo"] >= hi - 1e-9 else (lo - c["lo"] if c["hi"] <= lo + 1e-9 else -1.0)
    if h <= 0 or c["a"] > 0.05 * h:
        return None
    return {"kind": "monopole", "length": 2 * h, "a": c["a"], "axis": ax, "plate": size,
            "plate_wavelengths": size * f0 / C0}


def _crossings(freqs, Z) -> list[tuple[float, complex]]:
    out = []
    x = [z.imag for z in Z]
    for i in range(len(x) - 1):
        if x[i] == 0 or x[i] * x[i + 1] < 0:
            u = x[i] / (x[i] - x[i + 1]) if x[i] != x[i + 1] else 0.0
            out.append((freqs[i] + u * (freqs[i + 1] - freqs[i]), Z[i] + u * (Z[i + 1] - Z[i])))
    return out


def resonance(freqs, Z, z0: float) -> tuple[float, complex, str] | None:
    """Where the antenna resonates: the zero of the reactance closest to a match, or, when the reactance never
    crosses zero, an interior peak of the resistance (a parallel resonance seen through a feed's inductance)."""
    cr = _crossings(freqs, Z)
    if cr:
        f, z = min(cr, key=lambda c: abs((c[1] - z0) / (c[1] + z0)))
        return f, z, "reactance zero"
    r = [z.real for z in Z]
    i = int(np.argmax(r))
    if 0 < i < len(r) - 1:
        x = np.array(freqs[i - 1:i + 2], float)
        c = np.polyfit(x - x[1], r[i - 1:i + 2], 2)
        f = float(x[1] - c[1] / (2 * c[0])) if c[0] < 0 else float(x[1])
        f = min(max(f, x[0]), x[2])
        j = i if f >= freqs[i] else i - 1
        u = (f - freqs[j]) / (freqs[j + 1] - freqs[j])
        return f, Z[j] + u * (Z[j + 1] - Z[j]), "resistance peak"
    return None


BEND = 0.08           # the largest change in the reflection coefficient's magnitude between neighboring samples left alone


def next_frequency(freqs, Z, s: WaveSetup) -> tuple[float, str] | None:
    """The next frequency worth a solve: the resonance when no sample sits near it, else the middle of the
    interval where the reflection coefficient's magnitude changes most (a resonance sharper than the sweep's
    spacing shows up as a dip between two samples), until nothing changes by more than BEND or the intervals are narrower than an eighth of the
    tolerance. None when the sweep has seen enough."""
    res = resonance(freqs, Z, s.port.z0)
    if res and min(abs(res[0] - f) for f in freqs) > 0.003 * s.f0:
        return float(res[0]), "where the resonance is estimated to be"
    g = (np.asarray(Z) - s.port.z0) / (np.asarray(Z) + s.port.z0)
    step = np.abs(np.diff(np.abs(g)))
    width = np.diff(freqs)
    ok = width > s.tolerance * s.f0 / 8
    if not ok.any() or float(np.max(np.where(ok, step, 0.0))) < BEND:
        return None
    i = int(np.argmax(np.where(ok, step, 0.0)))
    return float(0.5 * (freqs[i] + freqs[i + 1])), "where the return loss changes most between samples"


def _zs(z: complex) -> str:
    return f"{fmt(z.real)} {'+' if z.imag >= 0 else '-'} j{fmt(abs(z.imag))} ohm"


def s11_db(Z, z0: float) -> np.ndarray:
    g = np.abs((np.asarray(Z) - z0) / (np.asarray(Z) + z0))
    return 20 * np.log10(np.maximum(g, 1e-6))


# --- the study -------------------------------------------------------------------------------------
def _validate(scene: Scene, s: WaveSetup, quality: str) -> None:
    if s.port is None:
        raise ValueError("an antenna study needs a port, the two points across the feed gap (Port(a, b))")
    if not s.f0 or s.f0 <= 0:
        raise ValueError("give the working frequency f0 in Hz, for example 2.4e9")
    if not s.metals and not s.sheets:
        raise ValueError("the antenna has no metal. Mark its conductors as metals or give copper sheets")
    names = {b.name for b in scene.bodies}
    for n in list(s.metals) + list(s.dielectrics):
        if n not in names:
            raise ValueError(f"there is no body called '{n}'. The scene has {scene.listing()}")
    if "port" in names or "port" in s.sheets:
        raise ValueError("'port' is the name of the feed strip. Rename the body or sheet called port")
    if quality not in QUALITY:
        raise ValueError(f"unknown quality '{quality}'. Use fast, normal or fine")
    if not 0 < s.span < 0.9:
        raise ValueError("the sweep span is a fraction of f0 between 0 and 0.9, for example 0.25")


def _sweep(s: WaveSetup, points: int) -> list[float]:
    """The first frequencies: evenly spaced over the span, the working frequency and the hints."""
    freqs = [float(f) for f in np.linspace(s.f0 * (1 - s.span), s.f0 * (1 + s.span), max(int(points), 3))]
    freqs = [float(s.f0) if abs(f - s.f0) < 1e-6 * s.f0 else f for f in freqs]
    if s.f0 not in freqs:
        freqs = sorted(freqs + [float(s.f0)])
    return sorted(set(freqs) | {float(h) for h in s.hints if 0.5 * s.f0 <= h <= 1.5 * s.f0})


def run(scene: Scene, setup: WaveSetup, out_dir, question: str = "", quality: str = "normal",
        log=lambda stage, detail="": None, title: str = "", keep: bool = False, budget: float = 0.0,
        compare=None, notes=(), plan_only: bool = False):
    """Sweep the antenna's impedance, find its resonance and pattern, check the answer and write the report.

    budget is the wall time in seconds the study may be predicted to take (0 for no limit). The prediction comes from
    the base mesh before any frequency is solved. A study over its budget gives up frequencies, then mesh fineness,
    then its verification solves, and one far over refuses with a CostRefusal. compare(found) may return more checks
    against a reference the caller knows (found holds the resonance, the sweep and the working frequency's solve),
    and notes are added to the report's assumptions. With plan_only the run stops after the plan and returns it, which
    is how 'wobbl estimate' prices a study."""
    from dataclasses import replace

    from ..outcome_generic import GenericOutcome
    from . import wavecost

    t0 = time.time()
    s = setup
    _validate(scene, s, quality)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    k = QUALITY[quality]
    freqs = _sweep(s, s.points)
    lam_min, lam_max = C0 / max(freqs), C0 / min(freqs)
    log("mesh", "placing the port")
    fd = feed(scene, s)
    sheets = {**{n: [np.asarray(p, float) for p in ps] for n, ps in s.sheets.items()}, "port": [fd["poly"]]}
    center, r_ext = _extent(scene, s, sheets)
    R = r_ext + lam_max / 2
    rs = r_ext + 0.5 * (R - r_ext)
    sizes = element_sizes(scene, s, lam_min, sheets, fd["gap"])
    def make(tag: str, kk: float, radius: float):
        log("mesh", f"{tag} mesh, air sphere {fmt(radius * 1e3)} mm")
        sz = dict(sizes)
        for attempt in range(3):
            try:
                return mesh(scene, s.metals, list(s.dielectrics), work / tag / "m.msh", k=kk,
                            domain=Domain("sphere", radius=radius, max_size=lam_min / 10, growth=GROWTH), sizes=sz, sheets=sheets)
            except (SceneError, IsolatedTimeout):
                raise
            except Exception as e:  # gmsh fails on a surface whose elements are too coarse for it
                if attempt == 2:
                    raise ValueError(f"gmsh could not mesh the antenna ({e}). Give smaller element sizes for its "
                                     "thin or curved parts in the setup's sizes") from e
                log("mesh", "gmsh could not mesh a curved surface, trying finer elements on the conductors")
                sz = {n: 0.7 * h for n, h in sz.items()}

    t_mesh = time.time()
    guess = wavecost.assess(int(wavecost.nodes_from(mesh_inputs(scene, s, quality))), quality, len(freqs), s.refine,
                            budget)
    if plan_only:
        shutil.rmtree(work, ignore_errors=True)
        return guess
    log("plan", f"about {guess.nodes:,} nodes and {guess.solves:.0f} frequency solves, predicted "
                f"{wavecost._t(guess.seconds)} before meshing")
    if guess.quality != quality or guess.points < len(freqs):
        s = replace(s, points=guess.points, refine=guess.refine)
        freqs = _sweep(s, guess.points)
        quality, k = guess.quality, QUALITY[guess.quality]
    msh = make("base", k, R)
    plan = wavecost.assess(msh.n_nodes, quality, len(freqs), s.refine, budget)
    plan.gave_up = guess.gave_up + [g for g in plan.gave_up if g not in guess.gave_up]
    log("plan", f"{msh.n_nodes:,} nodes, about {plan.solves:.0f} frequency solves, predicted {wavecost._t(plan.seconds)}"
                + (f" (gave up {'; '.join(plan.gave_up)})" if plan.gave_up else ""))
    if plan.quality != quality or plan.points < len(freqs):
        s = replace(s, points=plan.points, refine=plan.refine)
        freqs = _sweep(s, plan.points)
        if plan.quality != quality:
            quality, k = plan.quality, QUALITY[plan.quality]
            msh = make("base", k, R)
    t_mesh = time.time() - t_mesh
    # about three samples per element across the sampling sphere, so the element scale ripple of the field
    # averages out of the radiation integrals instead of aliasing into the pattern
    n_th = int(np.clip(math.ceil(3 * math.pi * rs / (0.1 * lam_min * k)), 24, 64))
    sph = {**sphere_grid(n_th), "center": center, "radius": rs}

    bbox = np.array([b.bbox for b in scene.bodies if b.name in s.metals or b.name in s.dielectrics] or
                    [scene.bbox])
    pts = np.concatenate([np.asarray(p) for ps in sheets.values() for p in ps])
    box = tuple(np.minimum(bbox[:, :3].min(0), pts.min(0))) + tuple(np.maximum(bbox[:, 3:].max(0), pts.max(0)))
    pl = pro3d.slice_plane(box, through=tuple(0.5 * (fd["a"] + fd["b"])), n=64, margin=0.5)
    rows: dict[float, dict] = {}

    def at(f: float, i: int, n: int) -> dict:
        r = _solve(msh, s, f, fd, sph, work / "base" / f"f{len(rows)}", pl if f == s.f0 else None, cuts=f == s.f0)
        log("solve", f"{fmt(f / 1e6)} MHz ({i} of {n}), {r['dofs']:,} unknowns, Z = {_zs(r['Z'])}")
        rows[f] = r
        return r

    for i, f in enumerate(freqs):
        at(f, i + 1, len(freqs))
    for _ in range(max(int(s.refine), 0)):
        fs = sorted(rows)
        nxt = next_frequency(fs, [rows[f]["Z"] for f in fs], s)
        if nxt is None:
            break
        log("solve", f"adding {fmt(nxt[0] / 1e6)} MHz, {nxt[1]}")
        at(nxt[0], len(rows) + 1, len(rows) + 1)
    fs = sorted(rows)
    Z = [rows[f]["Z"] for f in fs]
    base = rows[s.f0]
    checks = []
    worst, worst_f = 0.0, s.f0
    for f in fs:
        r = rows[f]
        want = r["p_in"] - r["p_diel"]
        e = abs(r["p_rad"] - want) / max(abs(r["p_in"]), 1e-30)
        if e >= worst:
            worst, worst_f = e, f
    rb = rows[worst_f]
    pf = abs(base["p_ff"] - base["p_rad"]) / max(base["p_rad"], 1e-30)
    checks.append(Check("power balance", "pass" if worst < 0.05 else ("warn" if worst < 0.12 else "fail"),
                        f"the power flowing out through a sphere around the antenna matches the power fed in less "
                        f"the dielectric loss to {100 * worst:.2g}% or better at all {len(fs)} frequencies (worst at "
                        f"{fmt(worst_f / 1e6)} MHz, {fmt(rb['p_rad'])} W out against "
                        f"{fmt(rb['p_in'] - rb['p_diel'])} W for 1 A at the port), and the far field carries "
                        f"{fmt(base['p_ff'])} W at {fmt(s.f0 / 1e6)} MHz ({100 * pf:.2g}% apart)",
                        {"worst": worst, "far_field_vs_flux": pf}))
    extra = []
    mesh_shift_value = None
    nodes = [msh.n_nodes]
    dofs = [base["dofs"]]
    if plan.verify:
        log("verify", "the working frequency on a finer mesh")
        m2 = make("fine", 0.75 * k, R)
        r2 = _solve(m2, s, s.f0, fd, sph, work / "fine")
        nodes.append(m2.n_nodes)
        dofs.append(r2["dofs"])
        dz = abs(abs(r2["Z"]) - abs(base["Z"])) / abs(r2["Z"])
        shift = mesh_shift(fs, Z, s.f0, base["Z"], r2["Z"])
        worst_c = max(dz / 0.03, abs(shift) / 0.01 if shift is not None else 0.0)
        moved = ("" if shift is None else
                 f". Read as a shift of the whole response, the finer mesh moves a resonance {100 * shift:+.2g}% in "
                 f"frequency, and since the error of first order edge elements falls with the square of the element "
                 f"size, the coarse mesh's resonance is about {100 * shift * FINE_STEP:+.2g}% from the converged one")
        checks.append(Check("mesh convergence", "pass" if worst_c < 1 else ("warn" if worst_c < 3 else "fail"),
                            f"the impedance at {fmt(s.f0 / 1e6)} MHz went from {_zs(base['Z'])} to {_zs(r2['Z'])} on a "
                            f"mesh a quarter finer ({100 * dz:.2g}% in magnitude, {fmt(base['dofs'])} and "
                            f"{fmt(r2['dofs'])} unknowns){moved}",
                            {"z_coarse": [base["Z"].real, base["Z"].imag], "z_fine": [r2["Z"].real, r2["Z"].imag],
                             "resonance_shift": shift}))
        mesh_shift_value = shift
        log("verify", "the working frequency with the boundary farther out")
        m3 = make("far", k, 1.4 * R)
        r3 = _solve(m3, s, s.f0, fd, sph, work / "far")
        db = abs(r3["Z"] - base["Z"]) / abs(base["Z"])
        checks.append(Check("boundary distance", "pass" if db < 0.02 else ("warn" if db < 0.05 else "fail"),
                            f"moving the absorbing sphere from {fmt(R * 1e3)} to {fmt(1.4 * R * 1e3)} mm changes the "
                            f"impedance from {_zs(base['Z'])} to {_zs(r3['Z'])} ({100 * db:.2g}%)"))
    else:
        extra.append(("medium", "a fast run skips the finer mesh and the farther boundary, so the discretization "
                                "error is not measured"))
    wire = wire_geometry(scene, s, fd["t"], s.f0)
    res = resonance(fs, Z, s.port.z0)
    zt = None
    if wire:
        from .antenna import induced_emf

        dip = wire["kind"] == "dipole"
        zt = induced_emf(wire["length"], wire["a"], s.f0) * (1.0 if dip else 0.5)
        r_ref = 73.1 * (1.0 if dip else 0.5)
        what = "thin half wave dipole" if dip else "thin quarter wave monopole on an infinite ground plane"
        small = not dip and wire["plate_wavelengths"] < 0.75
        if res and res[2] == "reactance zero":
            dr = abs(res[1].real - r_ref) / r_ref
            st = "info" if small else ("pass" if dr < 0.08 else ("warn" if dr < 0.2 else "fail"))
            text = (f"resonance at {fmt(res[0] / 1e6)} MHz with {fmt(res[1].real)} ohm, against {fmt(r_ref)} ohm for "
                    f"a {what} ({100 * dr:.2g}% apart)")
        else:
            st, text = "info", "no resonance inside the sweep"
        text += (f". Induced EMF theory puts this length at {_zs(zt)} at {fmt(s.f0 / 1e6)} MHz, the field solve "
                 f"{_zs(base['Z'])}")
        if not dip:
            text += f", on a plate {wire['plate_wavelengths']:.2g} wavelengths across"
        checks.append(Check("closed form", st, text, {"induced_emf_ohm": [zt.real, zt.imag]}))
    if compare is not None:
        checks += compare({"resonance": res, "freqs": fs, "z": Z, "base": base, "rows": rows, "setup": s,
                           "gain_dBi": 10 * math.log10(max(_gain(base), 1e-12))})
    took = time.time() - t0
    checks.append(Check("run time", "info",
                        f"predicted {wavecost._t(plan.seconds)} from the {msh.n_nodes:,} node mesh before any solve, "
                        f"took {wavecost._t(took)} for {len(fs)} frequencies" +
                        (f". It gave up {'; '.join(plan.gave_up)} to fit the {wavecost._t(budget)} budget"
                         if plan.gave_up else ""), {"predicted_s": plan.seconds, "took_s": took}))
    lvl, reasons = confidence(checks, extra)
    s11 = s11_db(Z, s.port.z0)
    s11_0 = float(s11_db([base["Z"]], s.port.z0)[0])
    band = [f for f, v in zip(fs, s11) if v <= -10]
    bw = (max(band) - min(band)) if len(band) > 1 else 0.0
    if res:
        off = (res[0] - s.f0) / s.f0
        word = f"resonates at {fmt(res[0] / 1e6)} MHz ({100 * off:+.1f}% from {fmt(s.f0 / 1e6)} MHz)"
    else:
        off = 1.0
        word = "does not resonate inside the sweep"
    if res and abs(off) <= s.tolerance and s11_0 <= -10:
        verdict = "ok"
    elif res and abs(off) <= 2 * s.tolerance and s11_0 <= -6:
        verdict = "marginal"
    else:
        verdict = "fails"
    eff = min(max((base["p_in"] - base["p_diel"]) / base["p_in"], 0.0), 1.0) if base["p_in"] > 0 else 0.0
    gain = _gain(base)
    gdb = 10 * math.log10(max(gain, 1e-12))
    name = {"dipole": "dipole", "monopole": "monopole"}.get(wire["kind"], "antenna") if wire else "antenna"
    head = (f"The {name} {word}, with {_zs(base['Z'])} and S11 of {s11_0:.1f} dB against {fmt(s.port.z0)} ohm at "
            f"{fmt(s.f0 / 1e6)} MHz. Peak gain {gdb:.2f} dBi toward theta {base['theta_deg']:.0f} and phi "
            f"{base['phi_deg']:.0f} degrees.")
    fixes = []
    if verdict != "ok" and res:
        if wire:
            ln = (wire["length"] if wire["kind"] == "dipole" else wire["length"] / 2) * res[0] / s.f0
            fixes.append({"kind": "length", "title": f"Make the {name} about {fmt(ln * 1e3)} mm long",
                          "summary": "Resonance scales inversely with length, so this moves it onto the working "
                                     "frequency.", "auto": False})
        else:
            fixes.append({"kind": "length", "title": f"Scale the resonant length by {res[0] / s.f0:.3f}",
                          "summary": f"Make the part that sets the resonance {100 * abs(1 - res[0] / s.f0):.1f}% "
                                     f"{'longer' if res[0] > s.f0 else 'shorter'}, since resonance scales inversely "
                                     "with length.", "auto": False})
    if res and abs(res[1].real - s.port.z0) / s.port.z0 > 0.5:
        fixes.append({"kind": "match", "title": "Add a matching network",
                      "summary": f"At resonance the antenna is {fmt(res[1].real)} ohm, far from {fmt(s.port.z0)} ohm. "
                                 "An L network, a balun or moving the feed brings it closer.", "auto": False})
    images = {}
    fz = np.array(fs) / 1e6
    images["s11"] = plots.curve(out / "images" / "s11.png", fz, {"S11, dB": s11}, "frequency, MHz", "S11, dB",
                                hline=-10, hlabel="-10 dB")
    images["impedance"] = plots.curve(out / "images" / "impedance.png", fz,
                                      {"resistance, ohm": [z.real for z in Z], "reactance, ohm": [z.imag for z in Z]},
                                      "frequency, MHz", "ohm", hline=0.0)
    g_el = 10 * np.log10(np.maximum(base["d_el"] * eff, 1e-6))
    g_az = 10 * np.log10(np.maximum(base["d_az"] * eff, 1e-6))
    images["pattern"] = plots.polar(out / "images" / "pattern.png", base["angles"],
                                    {"elevation": g_el, "azimuth": g_az},
                                    "gain in dBi, elevation cut (0 is along the feed) and azimuth cut")
    sl = base.get("slice")
    if sl is not None and np.isfinite(sl).any():
        emag = np.sqrt(np.sum(np.abs(sl[..., :3] + 1j * sl[..., 3:]) ** 2, axis=-1))
        edb = 20 * np.log10(np.maximum(emag, 1e-30))
        top = float(np.nanpercentile(edb, 99.5)) if np.isfinite(edb).any() else float("nan")
        if np.isfinite(top):
            images["field"] = plots.grid_map(out / "images" / "field.png", edb, pl, "|E|, dB V/m for 1 A",
                                         f"field strength at {fmt(s.f0 / 1e6)} MHz on the plane {pl['normal']} = "
                                         f"{fmt(pl['at'] * 1e3)} mm", clim=(top - 50, top))
    values = {"freqs_Hz": fs, "z_ohm": [[z.real, z.imag] for z in Z], "s11_dB": [float(v) for v in s11],
              "resonance_Hz": res[0] if res else None, "resonance_kind": res[2] if res else None,
              "z_resonance_ohm": [res[1].real, res[1].imag] if res else None,
              "z_at_f0_ohm": [base["Z"].real, base["Z"].imag], "s11_at_f0_dB": s11_0, "bandwidth_Hz": bw,
              "directivity_dBi": 10 * math.log10(max(base["d_max"], 1e-12)), "gain_dBi": gdb, "efficiency": eff,
              "peak_theta_deg": base["theta_deg"], "peak_phi_deg": base["phi_deg"],
              "gain_over_sweep_dBi": [10 * math.log10(max(_gain(rows[f]), 1e-12)) for f in fs],
              "radiated_power_W_per_A2": base["p_rad"], "input_power_W_per_A2": base["p_in"],
              "dielectric_loss_W_per_A2": base["p_diel"], "unknowns": dofs, "nodes": nodes,
              "air_radius_m": R, "sampling_radius_m": rs, "port_width_m": fd["width"], "port_gap_m": fd["gap"],
              "plan": plan.to_dict(), "mesh_seconds": t_mesh, "mesh_resonance_shift": mesh_shift_value,
              "peak_memory_bytes": _peak_memory(),
              "solve_cpu_s": [rows[f]["cpu"] for f in fs], "solve_wall_s": [rows[f]["wall"] for f in fs],
              "_figures": [["s11", "S11 over the sweep"], ["impedance", "Input impedance"],
                           ["pattern", "Radiation pattern cuts"], ["field", "Field strength near the antenna"],
                           ["range", "Received power over distance"]]}
    if wire:
        values.update({"wire": wire, "induced_emf_ohm": [zt.real, zt.imag]})
    metrics = [("resonance", f"{fmt(res[0] / 1e6)} MHz" if res else "none in sweep"), ("S11", f"{s11_0:.1f} dB"),
               ("impedance", _zs(base["Z"])), ("gain", f"{gdb:.2f} dBi"), ("mesh", f"{fmt(base['dofs'])} unknowns")]
    assumptions = [f"Fed by a lumped port, a {fmt(fd['width'] * 1e3)} mm wide strip across the "
                   f"{fmt(fd['gap'] * 1e3)} mm gap from ({_mm(fd['a'])}) to ({_mm(fd['b'])}) mm carrying 1 A. The "
                   "impedance is the voltage across the gap over that current, so it holds the gap's capacitance and "
                   "the strip's inductance.",
                   "Metals and copper sheets are perfect conductors, so conductor loss is left out. Dielectric loss "
                   "comes from each material's loss tangent.",
                   f"Free space around the antenna, closed by a sphere of {fmt(R * 1e3)} mm radius that absorbs "
                   f"outgoing waves (a second order boundary condition). The far field comes from the fields on a "
                   f"sphere of {fmt(rs * 1e3)} mm.",
                   f"First order edge elements no larger than a tenth of the shortest wavelength in air, one mesh for "
                   f"the whole sweep with {fmt(base['dofs'])} unknowns."]
    assumptions += [f"'{n}' is {v[2] if len(v) > 2 else 'a dielectric'}, relative permittivity {v[0]:g}, loss tangent "
                    f"{v[1]:g}." for n, v in s.dielectrics.items()]
    assumptions += fd["notes"] + list(scene.notes) + list(notes)
    lk = None
    if s.link.get("tx_power") or s.link.get("distance"):
        from .antenna import link_budget

        lk = link_budget({**s.link, "frequency": s.f0, "tx_gain": s.link.get("tx_gain", gdb),
                          "rx_gain": s.link.get("rx_gain", gdb)})
        values.update(lk["values"])
        checks += lk["checks"]
        head += " " + lk["headline"]
        metrics.append(("range, 90%", f"{fmt(lk['values']['range_reliable_m'])} m"))
        if lk["verdict"] == "fails" or (lk["verdict"] == "marginal" and verdict == "ok"):
            verdict = lk["verdict"]
        assumptions += lk["assumptions"]
        if "rx_gain" not in s.link:
            assumptions.append("The far end of the link uses the same antenna.")
        fixes += lk["fixes"]
        images["range"] = lk["image"](out / "images" / "range.png")
        lvl, reasons = confidence(checks, extra + [("medium", "the range comes from an empirical path loss model "
                                                              f"with {fmt(lk['values']['sigma_dB'])} dB of shadowing")])
    o = GenericOutcome("antenna", title or "antenna", question, verdict, head, metrics, lvl, reasons, checks,
                       assumptions, values, images, fixes, solver=f"GetDP {solve.getdp().version()} with gmsh",
                       t_total=time.time() - t0)
    _playback(out, s, fd, pl, base, fs, Z, s11, g_el, g_az, lk)
    try:
        from . import antscene
        antscene.write(out, title=f"The {title or name} at {fmt(s.f0 / 1e6)} MHz", question=question, scene=scene, setup=s,
                       feed=fd, msh=msh, plane=pl, base=base, gain_db=gdb, eff=eff, f0=s.f0,
                       verdict=({"ok": "WORKS", "marginal": "MARGINAL", "fails": "FAILS"}[verdict], head,
                                {"ok": "pass", "marginal": "warn", "fails": "fail"}[verdict]))
    except Exception as e:  # noqa: BLE001  a picture must not lose a solved study
        o.checks.append(Check("3D scene", "info", f"the 3D scene could not be written ({type(e).__name__}: {e})"))
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _playback(out: Path, s: WaveSetup, fd: dict, pl: dict, base: dict, fs, Z, s11, g_el, g_az, lk) -> str:
    """The field at the working frequency as a phasor on the slice, the pattern cuts, the sweep and the range."""
    from . import playback as pb
    from .antenna import RANGE_MANIFEST, _range_playback

    data = {"frequency_Hz": s.f0,
            "slice": {k: (round(v * 1e3, 4) if k in ("at", "u0", "u1", "v0", "v1") else v) for k, v in pl.items()
                      if k not in ("iu", "iv", "ik")},
            "port": {"a_mm": [round(1e3 * x, 4) for x in fd["a"]], "b_mm": [round(1e3 * x, 4) for x in fd["b"]],
                     "width_mm": round(fd["width"] * 1e3, 4)},
            "pattern": {"angle_deg": [round(float(np.degrees(a)), 1) for a in base["angles"][::2]],
                        "elevation_dBi": [round(float(g), 3) for g in g_el[::2]],
                        "azimuth_dBi": [round(float(g), 3) for g in g_az[::2]],
                        "axes": [[round(float(x), 5) for x in v] for v in base["axes"]]},
            "sweep": pb.series(np.array(fs) / 1e6, {"s11_dB": s11, "r_ohm": [z.real for z in Z],
                                                     "x_ohm": [z.imag for z in Z]})}
    manifest = {"eu_re, eu_im, ev_re, ev_im": "real and imaginary parts of the electric field phasor along the "
                                              "slice's u and v axes, V/m for 1 A at the port, packed on one range "
                                              "clipped at the 98th percentile (the field is singular at metal edges); "
                                              "play Re(E e^(j w t)) = re cos(w t) - im sin(w t)",
                "slice": "the plane: its normal axis and position, the in-plane axes u and v and their ranges in mm, "
                         "nu by nv points, row major from the low v edge, -1 inside metal",
                "port": "the feed strip's ends and width in mm",
                "pattern": "gain in dBi on the elevation cut (angle from the feed axis, axes[0], toward axes[1]) and "
                           "the azimuth cut (angle from axes[1] toward axes[2])",
                "sweep": "x is frequency in MHz, with S11 in dB and the input resistance and reactance in ohm"}
    sl = base.get("slice")
    if sl is not None and np.isfinite(sl).any():
        iu, iv = pl["iu"], pl["iv"]
        comps = {"eu_re": sl[..., iu], "eu_im": sl[..., 3 + iu], "ev_re": sl[..., iv], "ev_im": sl[..., 3 + iv]}
        m = float(np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in comps.values()])), 98)) or 1.0
        data.update({k: pb.pack(v, -m, m) for k, v in comps.items()})
    if lk:
        data["range"] = _range_playback(lk)
        manifest.update(RANGE_MANIFEST)
    return pb.write(out, "antenna-3d", manifest, data)


# --- test geometry ---------------------------------------------------------------------------------
def _write_dipole(length: float, diameter: float, gap: float, path: str) -> str:
    from .scene import _session

    gmsh = _session()
    try:
        occ = gmsh.model.occ
        arm = (length - gap) / 2
        lo = occ.addCylinder(0, 0, -length / 2, 0, 0, arm, diameter / 2)
        hi = occ.addCylinder(0, 0, gap / 2, 0, 0, arm, diameter / 2)
        occ.synchronize()
        gmsh.model.setEntityName(3, lo, "arm 1")
        gmsh.model.setEntityName(3, hi, "arm 2")
        gmsh.write(path)
        return path
    finally:
        gmsh.finalize()


def wire_dipole_scene(length_mm: float, diameter_mm: float, gap_mm: float, out_dir) -> tuple[Scene, Port]:
    """A straight wire dipole along z, centered on the origin: two rods written to dipole.step and loaded as a
    scene, with the port across the gap between them."""
    from ..mesh.isolate import run_isolated
    from .scene import load

    if diameter_mm <= 0 or gap_mm <= 0 or gap_mm >= length_mm / 2:
        raise ValueError("give a positive diameter and a gap shorter than half the dipole's length")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = out / "dipole.step"
    run_isolated(_write_dipole, float(length_mm), float(diameter_mm), float(gap_mm), str(step), timeout=120)
    sc = load([str(step)], out / "scene")
    g = 0.5e-3 * gap_mm
    return sc, Port((0.0, 0.0, -g), (0.0, 0.0, g))
