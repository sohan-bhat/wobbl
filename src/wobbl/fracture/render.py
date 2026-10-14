"""Pictures of a crack growth result: the path through the part, crack size against cycles, and how lives spread."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK, _camera, _grid, _plotter, _surface


def _ribbon(fronts: list[np.ndarray], k: int = 25) -> np.ndarray:
    """Every front resampled to k points by arc length, stacked (fronts, k, 3)."""
    out = []
    for f in fronts:
        seg = np.linalg.norm(np.diff(f, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], k)
        out.append(np.column_stack([np.interp(t, s, f[:, i]) for i in range(3)]))
    return np.asarray(out)


def path_image(path: str | Path, X, tets, fronts: list[list[np.ndarray]], cycles: list[float], origin,
               label: str, starts: list[np.ndarray] | None = None) -> str:
    """The part see through, the crack surface colored by the cycle count when the front passed, and each front
    drawn as a line. ``fronts`` holds, per crack side, the fronts in the order the crack grew through them, and
    ``starts`` the line each side starts from, so the starting flaw is drawn in gray."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _path_vtk(path, X, tets, fronts, cycles, origin, label, starts)
    except Exception as e:  # noqa: BLE001  no OpenGL: draw it with matplotlib
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return _path_mpl(path, X, fronts, cycles, origin, label, str(e))


def _path_vtk(path, X, tets, fronts, cycles, origin, label, starts=None):
    import pyvista as pv

    surf = _surface(_grid(X, tets))
    p = _plotter((1500, 950))
    p.add_mesh(surf, color="#8a9099", opacity=0.22, smooth_shading=True)
    p.add_mesh(surf.extract_feature_edges(35), color="#6b7078", line_width=1.2, opacity=0.5)
    pts_all = []
    lim = [0.0, max(float(np.nanmax(cycles)), 1.0)] if len(cycles) else [0.0, 1.0]
    for n, side in enumerate(fronts):
        if starts is not None and n < len(starts) and len(side):
            flaw = _ribbon([starts[n], side[0]])
            p.add_mesh(pv.StructuredGrid(*(flaw[:, :, i].T for i in range(3))), color="#9aa0a6")
        if len(side) < 2:
            continue
        rib = _ribbon(side)
        n_f, k = rib.shape[:2]
        grid = pv.StructuredGrid(*(rib[:, :, i].T for i in range(3)))
        grid.point_data["cycles"] = np.repeat(np.asarray(cycles[:n_f], float)[None, :], k, axis=0).ravel(order="F")
        p.add_mesh(grid, scalars="cycles", cmap="plasma", clim=lim, show_scalar_bar=True,
                   scalar_bar_args=dict(title="cycles when the front passed", color=FG, vertical=True, fmt="%.3g",
                                        title_font_size=18, label_font_size=15, position_x=0.86, position_y=0.12,
                                        height=0.72, width=0.045, font_family="courier", n_labels=5))
        for f in side:
            p.add_mesh(pv.lines_from_points(f), color="#f2f2ee", line_width=2)
        pts_all.append(rib.reshape(-1, 3))
    o = np.asarray(origin, float)
    pts = np.vstack(pts_all) if pts_all else o[None, :]
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) + 1e-9
    p.add_mesh(pv.Sphere(radius=min(0.008 * span, 0.04 * extent), center=o), color=MARK)
    p.add_point_labels(np.atleast_2d(o), [label], font_size=18, text_color="#111111", shape_color=MARK,
                       shape="rounded_rect", shape_opacity=0.95, point_size=1, show_points=False,
                       always_visible=True, font_family="courier", margin=6)
    nrm = np.cross(pts[len(pts) // 3] - pts[0], pts[-1] - pts[0]) if len(pts) > 2 else np.array([1.0, -1.0, 1.0])
    view = nrm / max(np.linalg.norm(nrm), 1e-12)
    view = 0.8 * view + 0.35 * np.array([0.0, 0.0, 1.0]) + 0.2 * np.array([1.0, -1.0, 0.0])
    view /= np.linalg.norm(view)
    zoom = float(np.clip(0.55 * span / extent, 1.2, 6.0))
    _camera(p, X, view, focus=pts.mean(0), zoom=zoom)
    p.screenshot(str(path))
    p.close()
    return str(path)


def _path_mpl(path, X, fronts, cycles, origin, label, reason):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 6), dpi=140, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    cmap = plt.get_cmap("plasma")
    top = max(float(np.nanmax(cycles)), 1.0) if len(cycles) else 1.0
    for side in fronts:
        for f, n in zip(side, cycles):
            ax.plot(f[:, 0], f[:, 1], f[:, 2], color=cmap(n / top), lw=1.6)
    o = np.asarray(origin)
    ax.scatter([o[0]], [o[1]], [o[2]], color=MARK, s=30)
    ax.set_title(f"{label} (matplotlib fallback: {reason[:40]})", color=FG, fontsize=9)
    ax.set_axis_off()
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)


def growth_image(path: str | Path, cycles, size, steps_n, steps_size, xlabel: str = "cycles") -> str:
    """Crack size against cycles for the nominal part, with the solved crack steps marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140, facecolor=BG)
    _axes(ax)
    ax.plot(cycles, size, color=MARK, lw=2, label="nominal part")
    ax.plot(steps_n, steps_size, "o", color="#4a90d9", ms=5, label="solved crack steps")
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel("crack size, mm", color=FG)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def life_image(path: str | Path, lives: np.ndarray, percentiles: dict, target: float | None) -> str:
    """The share of parts failed by each cycle count on log axes, with B50, B10 and B1 and the target marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.sort(lives[np.isfinite(lives) & (lives > 0)])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140, facecolor=BG)
    _axes(ax)
    if len(finite):
        frac = (np.arange(len(finite)) + 0.5) / len(lives)
        ax.semilogx(finite, 100 * frac, color=MARK, lw=2, label="parts failed")
        ax.set_yscale("log")
        ax.set_ylim(max(50.0 / len(lives), 0.01), 100)
    colors = {"B50": "#3ccf7a", "B10": "#4a90d9", "B1": "#ff6b6b", "B0.1": "#c77dff"}
    for name, v in percentiles.items():
        if np.isfinite(v) and v > 0:
            ax.axvline(v, color=colors.get(name, FG), ls=":", lw=1.2, label=f"{name} {v:.3g}")
    if target:
        ax.axvline(target, color="#ff4d4f", ls="--", lw=1.4, label=f"target {target:.3g}")
    ax.set_xlabel("cycles to failure", color=FG)
    ax.set_ylabel("parts failed, %", color=FG)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)
