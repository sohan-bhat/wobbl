"""Pictures for the manufacturing studies: a field cut open through the spot that matters, and 2D maps.

A casting's hot spot and a molding's weld line sit inside the part or across a
face, so the heatmap is a section: the part is cut by a plane through the
marked point and the cut face is colored, drawn with the renderer of the other
heatmaps.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK


def section(path: str | Path, X: np.ndarray, tets: np.ndarray, values: np.ndarray, label: str, unit: str,
            origin, normal, mark_xyz=None, mark_label: str = "", cmap: str = "inferno", clim=None) -> str:
    """The part cut by the plane (origin, normal), keeping the half the normal points away from, colored by the
    nodal field; values that are nan are drawn gray."""
    from ..mesh.clip import carry, clip_surface
    from ..report import render as R

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, float)
    # the color range skips the odd node on a boundary that the field puts far from the rest
    lo, hi = clim or (float(np.nanpercentile(v, 2)), float(np.nanmax(v)))
    try:
        (a, b, t), faces = clip_surface(X, tets, origin, normal)
        P = carry(X, a, b, t)
        surf = R.surface(P, faces, {"v": carry(v, a, b, t)})
        look = R.Look(cmap, (lo, hi), ambient=0.35, diffuse=0.75)
        size = (1600, 1000)
        cam = R.camera(iso_view(normal), P, size)
        g = np.asarray(mark_xyz if mark_xyz is not None else origin, float)
        R.compose(path, [R.Panel(R.draw(size, cam, surf, surf.fields["v"], look), cam,
                                 (cmap, (lo, hi), f"{label}\n{unit}"), [R.Mark(g, mark_label)] if mark_label else [])])
    except Exception:  # noqa: BLE001  matplotlib fallback: the nodes near the plane
        if os.environ.get("WOBBL_DEBUG"):
            raise
        _section_mpl(path, X, v, label, unit, origin, normal, mark_xyz, cmap, lo, hi)
    return str(path)


def _section_mpl(path, X, v, label, unit, origin, normal, mark_xyz, cmap, lo, hi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.asarray(normal, float) / np.linalg.norm(normal)
    d = (X - np.asarray(origin)) @ n
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    near = np.abs(d) < 0.03 * span
    u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    P = X[near] @ np.column_stack([u, w])
    fig, ax = plt.subplots(figsize=(10, 6.25), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    sc = ax.scatter(P[:, 0], P[:, 1], c=v[near], cmap=cmap, vmin=lo, vmax=hi, s=6)
    if mark_xyz is not None:
        m = np.asarray(mark_xyz) @ np.column_stack([u, w])
        ax.scatter([m[0]], [m[1]], s=120, c=MARK, edgecolors="black")
    ax.set_aspect("equal")
    ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label(f"{label} ({unit})", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def cut_surface(X: np.ndarray, tets: np.ndarray, fields: dict[str, np.ndarray], cut=None):
    """The surface of a tetrahedral mesh, cut by a plane (origin, normal) keeping the side the normal points away
    from, as (vertices, triangles, fields at the vertices); nan in a field becomes -1."""
    from ..mesh.clip import carry, clip_surface

    (a, b, t), faces = clip_surface(X, tets, *(cut if cut is not None else (None, None)))
    vals = {k: carry(np.nan_to_num(np.asarray(v, float), nan=-1.0), a, b, t) for k, v in fields.items()}
    return carry(np.asarray(X, float), a, b, t), faces, vals


def iso_view(normal) -> np.ndarray:
    """A three quarter view onto a face whose outward normal is given, from above when the face is upright."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    v = n + 0.55 * a - 0.55 * b * np.sign(b[2] if abs(b[2]) > 1e-9 else -1.0)
    return v / np.linalg.norm(v)


def plane_through(X: np.ndarray, point) -> tuple[np.ndarray, np.ndarray]:
    """A cutting plane through point, square to the part's thinnest direction, so a plate is cut through its
    middle and a tall part across its height."""
    ext = X.max(0) - X.min(0)
    n = np.zeros(3)
    n[int(np.argmin(ext))] = 1.0
    return np.asarray(point, float), n
