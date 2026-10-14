"""The 3D view of a general field run, written straight from the solved data through wobbl.viz.

Every body is drawn from the boundary of its own mesh, so the view shows the user's geometry as it was solved.
The field is the solved slice, standing in the scene where it was cut, colored by the run's values. Magnetic
runs add flux lines traced through the slice's in-plane field and the force on the target as an arrow, and
sound runs play each mode or the radiated wave over one period from the solved phasor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROLE_COLOR = {"steel": "#8d939c", "linear": "#8d939c", "magnet": "#c6463d", "coil": "#c47a3a", "conductor": "#c8ccd2",
              "dielectric": "#3d7bd9", "air": "#7fb3ff", "wall": "#6b7079", "source": "#ffd400"}


def surfaces(msh_path: str | Path, tags: dict, names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each named region's outer surface from a gmsh 2.2 mesh: a volume's faces that only it touches, or a surface
    region's own triangles. Coordinates in mm."""
    lines = Path(msh_path).read_text().splitlines()
    i0 = lines.index("$Nodes") + 2
    n = int(lines[i0 - 1])
    ids = np.empty(n, np.int64)
    X = np.empty((n, 3))
    for j in range(n):
        t = lines[i0 + j].split()
        ids[j] = int(t[0])
        X[j] = [float(t[1]), float(t[2]), float(t[3])]
    pos = {int(k): j for j, k in enumerate(ids)}
    e0 = lines.index("$Elements") + 2
    by = {t: name for name, t in tags.items() if name in names}
    tets: dict[str, list] = {k: [] for k in names}
    tris: dict[str, list] = {k: [] for k in names}
    for j in range(int(lines[e0 - 1])):
        t = lines[e0 + j].split()
        reg = by.get(int(t[3]))
        if reg is None:
            continue
        nodes = [pos[int(x)] for x in t[3 + int(t[2]):]]
        if t[1] in ("4", "11"):
            tets[reg].append(nodes[:4])
        elif t[1] in ("2", "9"):
            tris[reg].append(nodes[:3])
    out = {}
    for name in names:
        if tris[name]:
            T = np.array(tris[name])
        elif tets[name]:
            tt = np.array(tets[name])
            faces = np.concatenate([tt[:, [0, 2, 1]], tt[:, [0, 1, 3]], tt[:, [0, 3, 2]], tt[:, [1, 2, 3]]])
            key = np.sort(faces, axis=1)
            _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            T = faces[cnt[inv.ravel()] == 1]
        else:
            continue
        used = np.unique(T)
        remap = -np.ones(len(X), np.int64)
        remap[used] = np.arange(len(used))
        out[name] = (X[used] * 1e3, remap[T])
    return out


def slice_mesh(s, pl: dict, grid: np.ndarray, field: str, unit: str, cmap: str = "turbo", frames: bool = False,
               rng=None, name: str = "slice"):
    """The solved slice as a sheet of quads where it was cut, colored per vertex; cells outside the mesh are
    left out. grid is (nv, nu) or, with frames, (frames, nv, nu)."""
    g = np.asarray(grid, float)
    nv, nu = g.shape[-2:]
    U, V = np.meshgrid(np.linspace(pl["u0"], pl["u1"], nu), np.linspace(pl["v0"], pl["v1"], nv))
    X = np.zeros((U.size, 3))
    X[:, pl["iu"]], X[:, pl["iv"]], X[:, pl["ik"]] = U.ravel(), V.ravel(), pl["at"]
    ok = np.isfinite(g[0] if frames else g).ravel()
    idx = np.arange(nu * nv).reshape(nv, nu)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    keep = ok[a] & ok[b] & ok[c] & ok[d]
    T = np.concatenate([np.c_[a, b, c][keep], np.c_[a, c, d][keep]])
    if not len(T):
        return None
    m = s.mesh(X * 1e3, T, name=name, max_vertices=len(X) + 1, opacity=0.92)
    vals = np.nan_to_num(g.reshape(len(g), -1) if frames else g.ravel())
    m.color_by(field, vals, unit=unit, colormap=cmap, range=rng)
    return m


def draw_bodies(s, msh, roles: dict[str, str], opacity: dict | None = None) -> None:
    names = [n for n in roles if n in msh.tags]
    for name, (X, T) in surfaces(msh.path, msh.tags, names).items():
        role = roles.get(name, "air")
        s.mesh(X, T, name=name, color=ROLE_COLOR.get(role, "#b9bdc4"),
               opacity=(opacity or {}).get(name, 0.35 if role in ("air", "dielectric") else 1.0), max_vertices=6000)


def stream(bu: np.ndarray, bv: np.ndarray, pl: dict, n: int = 18, steps: int = 400) -> list[np.ndarray]:
    """Field lines through the slice's in-plane field, traced both ways from seeds where the field is strong."""
    nv, nu = bu.shape
    u = np.linspace(pl["u0"], pl["u1"], nu)
    v = np.linspace(pl["v0"], pl["v1"], nv)
    mag = np.hypot(np.nan_to_num(bu), np.nan_to_num(bv))
    if not mag.any():
        return []
    du, dv = u[1] - u[0], v[1] - v[0]
    order = np.argsort(mag.ravel())[::-1]
    seeds, taken = [], np.zeros_like(mag, bool)
    for k in order:
        i, j = divmod(int(k), nu)
        if taken[max(0, i - 3):i + 4, max(0, j - 3):j + 4].any():
            continue
        seeds.append((u[j], v[i]))
        taken[i, j] = True
        if len(seeds) >= n:
            break

    def at(p):
        fj, fi = (p[0] - u[0]) / du, (p[1] - v[0]) / dv
        if not (0 <= fi < nv - 1 and 0 <= fj < nu - 1):
            return None
        i, j = int(fi), int(fj)
        vec = np.array([bu[i, j], bv[i, j]])
        if not np.all(np.isfinite(vec)) or not vec.any():
            return None
        return vec / np.linalg.norm(vec)

    h = 0.6 * min(du, dv)
    out = []
    for sd in seeds:
        path = [np.array(sd)]
        for sign in (1, -1):
            p = np.array(sd)
            seg = []
            for _ in range(steps):
                d = at(p)
                if d is None:
                    break
                p = p + sign * h * d
                seg.append(p.copy())
            path = (seg[::-1] + path) if sign < 0 else (path + seg)
        if len(path) > 8:
            P = np.zeros((len(path), 3))
            P[:, pl["iu"]], P[:, pl["iv"]], P[:, pl["ik"]] = np.array(path)[:, 0], np.array(path)[:, 1], pl["at"]
            out.append(P * 1e3)
    return out


def tone(verdict: str) -> tuple[str, str]:
    v = (verdict or "").lower()
    return {"ok": ("OK", "pass"), "marginal": ("MARGINAL", "warn"), "fails": ("FAILS", "fail")}.get(
        v, (v.upper() or "DONE", "info"))
