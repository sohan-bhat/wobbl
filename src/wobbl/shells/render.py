"""Heatmaps of shell results, and small charts, saved as light palette PNGs.

A 6 node triangle is drawn as four flat triangles through its midside nodes, so curved walls stay smooth.
The images use the look of the solid heatmaps; a machine with no OpenGL gets the matplotlib fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..report import render as R

SIZE = (1200, 750)


def split_tris(tris: np.ndarray) -> np.ndarray:
    """(m,6) quadratic triangles to (4m,3) flat ones through the midside nodes."""
    if tris.shape[1] == 3:
        return tris
    a, b, c, ab, bc, ca = tris.T
    return np.vstack([np.column_stack(x) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])


def compress_png(path: str | Path) -> str:
    """Rewrite a PNG with a 256 color palette: a heatmap's smooth colors survive and the file shrinks about
    three times, which keeps the self contained report light."""
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    except Exception:  # noqa: BLE001  an uncompressed image is still a picture
        pass
    return str(path)


def render_shell(path: str | Path, X: np.ndarray, tris: np.ndarray, values: np.ndarray, label: str, unit: str,
                 cmap: str = "turbo", mark_xyz=None, mark_label: str = "", clim=None, U: np.ndarray | None = None,
                 note: str = "") -> str:
    """One heatmap of a nodal field on a shell, optionally on the deformed shape (U exaggerated to 8% of the
    part's size), with a marked point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t3 = split_tris(tris)
    span = float(np.linalg.norm(np.ptp(X, axis=0)))
    Xd = X
    if U is not None:
        um = float(np.linalg.norm(U, axis=1).max())
        Xd = X + (0.08 * span / um if um > 0 else 0.0) * U
    g = np.asarray(mark_xyz if mark_xyz is not None else X[int(np.argmax(values))], float)
    lo, hi = clim or (float(np.nanmin(values)), float(np.nanmax(values)))
    try:
        import pyvista as pv

        faces = np.hstack([np.full((len(t3), 1), 3), t3]).ravel()
        surf = pv.PolyData(Xd, faces)
        surf.point_data["v"] = values
        view = R._view_dir(X, g, R._surface_normal_at(X, tris, g))
        p = R._plotter(SIZE)
        p.add_mesh(surf, scalars="v", cmap=cmap, clim=[lo, hi], smooth_shading=True, ambient=0.3, diffuse=0.8,
                   scalar_bar_args=R._bar(f"{label}\n{unit}" if unit else label))
        if mark_label:
            R._marker(p, Xd[int(np.argmin(np.linalg.norm(X - g, axis=1)))], span, mark_label, R.MARK)
        if note:
            p.add_text(note, position="upper_left", font_size=11, color=R.FG, font="courier")
        R._camera(p, Xd, view)
        p.screenshot(str(path))
        p.close()
    except Exception:  # noqa: BLE001  matplotlib fallback
        if os.environ.get("WOBBL_DEBUG"):
            raise
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(8, 5), dpi=150, facecolor=R.BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=R.BG)
        norm = plt.Normalize(lo, hi)
        ax.add_collection3d(Poly3DCollection(Xd[t3], facecolors=plt.get_cmap(cmap)(norm(values[t3].mean(1))),
                                             linewidths=0))
        c, r = (Xd.min(0) + Xd.max(0)) / 2, np.ptp(Xd, axis=0).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off()
        fig.savefig(path, facecolor=R.BG, bbox_inches="tight")
        plt.close(fig)
    return compress_png(path)


def bar_chart(path: str | Path, labels: list[str], values: list[float], ylabel: str, line: float | None = None,
              line_label: str = "", highlight: int | None = None) -> str:
    """Bars in the report's colors, one highlighted, with an optional limit line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130, facecolor=R.BG)
    ax.set_facecolor(R.BG)
    colors = ["#4a90d9"] * len(values)
    if highlight is not None:
        colors[highlight] = R.MARK
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, color=R.FG, fontsize=9)
    if line is not None:
        ax.axhline(line, color="#ff4d4f", ls="--", lw=1.2, label=line_label)
        ax.legend(facecolor=R.BG, edgecolor="#3a3d42", labelcolor=R.FG)
    ax.set_ylabel(ylabel, color=R.FG)
    ax.tick_params(colors=R.FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(path, facecolor=R.BG)
    plt.close(fig)
    return compress_png(path)


def history_plot(path: str | Path, t, ys: dict, xlabel: str, ylabel: str) -> str:
    """Time histories in the report's colors, compressed."""
    return compress_png(R.line_plot(path, t, ys, xlabel, ylabel))


__all__ = ["render_shell", "bar_chart", "history_plot", "compress_png", "split_tris"]
