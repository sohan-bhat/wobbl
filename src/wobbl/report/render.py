"""Heatmap images: stress, safety factor, deflection and a zoom on the governing spot.

Rendering uses VTK off screen through pyvista. On a machine without any
OpenGL context (a bare Linux server), Wobbl falls back to a matplotlib
renderer so there is always a picture.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", module="pyvista")
warnings.filterwarnings("ignore", message=".*extract_surface.*")

BG = "#0e0f11"
FG = "#f2f2ee"
MARK = "#ffd400"      # crash test yellow
ARTIFACT = "#9aa0a6"


def _grid(X: np.ndarray, tets: np.ndarray):
    import pyvista as pv

    if tets.shape[1] == 10:
        ctype = pv.CellType.QUADRATIC_TETRA
    else:
        ctype = pv.CellType.TETRA
    k = tets.shape[1]
    cells = np.hstack([np.full((len(tets), 1), k), tets]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(tets), ctype, dtype=np.uint8), X)


def _surface(grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return grid.extract_surface(nonlinear_subdivision=2, algorithm="dataset_surface")
        except TypeError:
            return grid.extract_surface(nonlinear_subdivision=2)


ISO_DIRS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0.8, -0.8)], float)


def _view_dir(X: np.ndarray, focus: np.ndarray | None, normal: np.ndarray | None) -> np.ndarray:
    """An isometric style direction that faces the point of interest."""
    dirs = ISO_DIRS / np.linalg.norm(ISO_DIRS, axis=1)[:, None]
    if normal is None:
        if focus is None:
            return dirs[[d[2] > 0 and d[0] > 0 and d[1] < 0 for d in dirs]][0]
        c = X.mean(0)
        normal = focus - c
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([1.0, -1.0, 1.0])
    normal = normal / np.linalg.norm(normal)
    score = dirs @ normal + 0.15 * (dirs[:, 2] > 0)
    return dirs[int(np.argmax(score))]


def _surface_normal_at(X: np.ndarray, tris: np.ndarray, p: np.ndarray) -> np.ndarray | None:
    if tris is None or len(tris) == 0:
        return None
    t = tris[:, :3]
    c = X[t].mean(1)
    i = np.argsort(np.linalg.norm(c - p, axis=1))[:6]
    n = np.cross(X[t[i, 1]] - X[t[i, 0]], X[t[i, 2]] - X[t[i, 0]])
    n = n.sum(0)
    return n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else None


def _plotter(size):
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background(BG)
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:  # noqa: BLE001
        pass
    return p


def _camera(p, X, view: np.ndarray, focus=None, zoom=1.3):
    c = X.mean(0) if focus is None else np.asarray(focus)
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    up = np.array([0, 0, 1.0]) if abs(view[2]) < 0.95 else np.array([0, 1.0, 0])
    p.camera_position = [tuple(c + view * span * 2.2), tuple(c), tuple(up)]
    p.reset_camera_clipping_range()
    if focus is None:
        p.reset_camera()
    p.camera.zoom(zoom)


def _bar(title):
    return dict(title=title, color=FG, vertical=True, fmt="%.3g", title_font_size=20, label_font_size=16,
                position_x=0.88, position_y=0.12, height=0.72, width=0.045, font_family="courier", n_labels=6)


def render_structural(out_dir: str | Path, X, tets, tris, measure, U, governing_xyz, governing_value,
                      measure_label: str, allowable: float, target_sf: float, spots: list[dict],
                      unit: str = "MPa", scale_note: str = "") -> dict[str, str]:
    """Write stress.png, zoom.png, safety.png and deformed.png; returns name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _render_vtk(out, X, tets, tris, measure, U, governing_xyz, governing_value, measure_label,
                           allowable, target_sf, spots, unit)
    except Exception as e:  # noqa: BLE001  no OpenGL: fall back to matplotlib
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return _render_mpl(out, X, tris, measure, U, governing_xyz, governing_value, measure_label, allowable,
                           target_sf, spots, unit, reason=str(e))


def _render_vtk(out, X, tets, tris, measure, U, gxyz, gval, mlabel, allowable, target_sf, spots, unit):
    import pyvista as pv

    grid = _grid(X, tets)
    grid.point_data["stress"] = measure
    grid.point_data["U"] = U
    grid.point_data["umag"] = np.linalg.norm(U, axis=1)
    with np.errstate(divide="ignore"):
        grid.point_data["sf"] = np.where(measure > 0, allowable / np.maximum(measure, 1e-30), 99.0)
    surf = _surface(grid)
    g = np.asarray(gxyz, float)
    n_g = _surface_normal_at(X, tris, g)
    view = _view_dir(X, g, n_g)
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    files = {}
    clim = [0.0, float(gval) if gval > 0 else float(measure.max())]
    artifacts = [s for s in spots if s.get("excluded") and s.get("values") and s["values"][-1] > gval]

    def stress_plot(path, focus=None, zoom=1.3, size=(1600, 1000)):
        p = _plotter(size)
        mspan = span / (zoom / 1.3)
        p.add_mesh(surf, scalars="stress", cmap="turbo", clim=clim, smooth_shading=True, show_scalar_bar=True,
                   ambient=0.3, diffuse=0.8, specular=0.15,
                   scalar_bar_args=_bar(f"{mlabel}\n{unit}"), above_color="#ffffff" if artifacts else None)
        _marker(p, g, mspan, f"{gval:.3g} {unit}", MARK)
        for s in artifacts[:3]:
            _marker(p, np.asarray(s["xyz"]), mspan, "artifact", ARTIFACT, small=True)
        _camera(p, X, view, focus, zoom)
        p.screenshot(str(path))
        p.close()

    stress_plot(out / "stress.png")
    files["stress"] = str(out / "stress.png")
    stress_plot(out / "zoom.png", focus=g, zoom=3.2, size=(1200, 900))
    files["zoom"] = str(out / "zoom.png")

    p = _plotter((1600, 1000))
    top = max(2.0 * target_sf, 3.0)
    p.add_mesh(surf, scalars="sf", cmap="RdYlGn", clim=[0, top], smooth_shading=True, ambient=0.3, diffuse=0.8,
               scalar_bar_args=_bar("safety factor"))
    _marker(p, g, span, f"SF {allowable / gval:.2f}" if gval > 0 else "", MARK)
    _camera(p, X, view)
    p.screenshot(str(out / "safety.png"))
    p.close()
    files["safety"] = str(out / "safety.png")

    umax = float(np.linalg.norm(U, axis=1).max())
    factor = 0.08 * span / umax if umax > 0 else 1.0
    warped = surf.warp_by_vector("U", factor=factor)
    p = _plotter((1600, 1000))
    p.add_mesh(surf.extract_feature_edges(35), color="#6b7078", line_width=1.5, opacity=0.6)
    p.add_mesh(warped, scalars="umag", cmap="viridis", smooth_shading=True, ambient=0.3, diffuse=0.8,
               scalar_bar_args=_bar("deflection\nmm"))
    p.add_text(f"shape exaggerated {factor:.3g}x", position="upper_left", font_size=12, color=FG,
               font="courier")
    _camera(p, X, view)
    p.screenshot(str(out / "deformed.png"))
    p.close()
    files["deformed"] = str(out / "deformed.png")
    return files


def _marker(p, xyz, span, label, color, small=False):
    import pyvista as pv

    r = span * (0.005 if small else 0.008)
    p.add_mesh(pv.Sphere(radius=r * 1.5, center=xyz), color="#111111")
    p.add_mesh(pv.Sphere(radius=r, center=xyz), color=color)
    if label:
        p.add_point_labels(np.atleast_2d(xyz), [label], font_size=20 if not small else 14, text_color="#111111",
                           shape_color=color, shape="rounded_rect", shape_opacity=0.95, point_size=1,
                           show_points=False, always_visible=True, font_family="courier", margin=6)


def render_scalar(path: str | Path, X, tets, tris, values, label: str, unit: str, cmap: str = "inferno",
                  mark_xyz=None, mark_label: str = "", clim=None) -> str:
    """One heatmap of any nodal field (temperature, current density) with an optional marked point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        grid = _grid(X, tets)
        grid.point_data["v"] = values
        surf = _surface(grid)
        g = np.asarray(mark_xyz if mark_xyz is not None else X[int(np.argmax(values))], float)
        view = _view_dir(X, g, _surface_normal_at(X, tris, g))
        span = float(np.linalg.norm(X.max(0) - X.min(0)))
        p = _plotter((1600, 1000))
        p.add_mesh(surf, scalars="v", cmap=cmap, clim=clim or [float(values.min()), float(values.max())],
                   smooth_shading=True, ambient=0.3, diffuse=0.8, scalar_bar_args=_bar(f"{label}\n{unit}"))
        if mark_label:
            _marker(p, g, span, mark_label, MARK)
        _camera(p, X, view)
        p.screenshot(str(path))
        p.close()
    except Exception:  # noqa: BLE001  matplotlib fallback
        if os.environ.get("WOBBL_DEBUG"):
            raise
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        t = tris[:, :3]
        fig = plt.figure(figsize=(10, 6.25), dpi=150, facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        v = values[t].mean(1)
        lo_, hi_ = clim or (float(values.min()), float(values.max()))
        norm = plt.Normalize(lo_, hi_)
        ax.add_collection3d(Poly3DCollection(X[t], facecolors=plt.get_cmap(cmap)(norm(v)), linewidths=0))
        lo, hi = X.min(0), X.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off()
        fig.savefig(path, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
    return str(path)


def line_plot(path: str | Path, x, ys: dict, xlabel: str, ylabel: str, hline: float | None = None,
              hlabel: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    colors = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b"]
    for (name, y), col in zip(ys.items(), colors):
        ax.plot(x, y, color=col, lw=2, label=name)
    if hline is not None:
        ax.axhline(hline, color="#ff4d4f", ls="--", lw=1.2, label=hlabel)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def _render_mpl(out, X, tris, measure, U, gxyz, gval, mlabel, allowable, target_sf, spots, unit, reason=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    t = tris[:, :3]
    files = {}
    g = np.asarray(gxyz, float)
    view = _view_dir(X, g, _surface_normal_at(X, tris, g))
    elev = float(np.degrees(np.arcsin(np.clip(view[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(view[1], view[0])))

    def draw(values, cmap, clim, title, path, Xd=X):
        fig = plt.figure(figsize=(10, 6.25), dpi=150, facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        v = values[t].mean(1)
        norm = plt.Normalize(*clim)
        col = Poly3DCollection(Xd[t], facecolors=plt.get_cmap(cmap)(norm(v)), linewidths=0)
        ax.add_collection3d(col)
        lo, hi = Xd.min(0), Xd.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.scatter(*g, s=80, c=MARK, edgecolors="black", depthshade=False)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.7)
        cb.set_label(title, color=FG)
        cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
        fig.savefig(path, facecolor=BG, bbox_inches="tight")
        plt.close(fig)

    draw(measure, "turbo", (0, gval), f"{mlabel} ({unit})", out / "stress.png")
    files["stress"] = str(out / "stress.png")
    sf = np.where(measure > 0, allowable / np.maximum(measure, 1e-30), 99.0)
    draw(sf, "RdYlGn", (0, max(2 * target_sf, 3)), "safety factor", out / "safety.png")
    files["safety"] = str(out / "safety.png")
    um = np.linalg.norm(U, axis=1)
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    fac = 0.08 * span / um.max() if um.max() > 0 else 1.0
    draw(um, "viridis", (0, um.max()), "deflection (mm)", out / "deformed.png", X + fac * U)
    files["deformed"] = str(out / "deformed.png")
    files["renderer"] = f"matplotlib fallback ({reason[:80]})"
    return files


def convergence_plot(path: str | Path, levels: list[dict], spots: list[dict], unit: str = "MPa",
                     label: str = "governing stress") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    n = [L["n_nodes"] for L in levels]
    g = [L["governing"] for L in levels]
    for s in spots:
        vals = s.get("values", [])
        if len(vals) == len(n):
            style = dict(color=ARTIFACT, lw=1, ls="--") if s.get("excluded") else dict(color="#4a90d9", lw=1)
            ax.plot(n, vals, marker="o", ms=3, alpha=0.7, **style)
    ax.plot(n, g, marker="o", color=MARK, lw=2.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("mesh nodes", color=FG)
    ax.set_ylabel(unit, color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG)
    ax.set_title("mesh convergence: yellow is the reported value, dashed are excluded artifacts", color=FG,
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)
