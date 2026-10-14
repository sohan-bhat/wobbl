"""Pictures of flow answers: charts with the operating point marked, the Moody chart, and 2D velocity and pressure
fields. They share the report's dark style, and every PNG is saved with a 256 colour palette, which keeps a chart
under about 100 kB with no visible change."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK

LINES = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b", "#c58af9"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel: str, ylabel: str, title: str = ""):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG, which="both")
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15, which="both")
    if title:
        ax.set_title(title, color=FG, fontsize=10)


def save(fig, path: str | Path) -> str:
    """Write a figure as a palette PNG."""
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    except Exception:  # noqa: BLE001  the plain PNG is still a valid picture
        pass
    return str(path)


def chart(path: str | Path, series: dict, xlabel: str, ylabel: str, points: dict | None = None,
          title: str = "", logx: bool = False, logy: bool = False, hline: tuple[float, str] | None = None) -> str:
    """Lines (name to (x, y)), measured points (name to (x, y)) drawn as dots, and an optional limit line."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    for (name, (x, y)), col in zip(series.items(), LINES):
        ax.plot(x, y, color=col, lw=2, label=name)
    for (name, (x, y)), col in zip((points or {}).items(), LINES[1:] + LINES[:1]):
        ax.plot(x, y, "o", color=col, ms=6, mec=BG, label=name)
    if hline:
        ax.axhline(hline[0], color="#ff4d4f", ls="--", lw=1.2, label=hline[1])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style(ax, xlabel, ylabel, title)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=9)
    fig.tight_layout()
    return save(fig, path)


def field2d(path: str | Path, x, y, values, label: str, cmap: str = "viridis", vectors=None, mask=None,
            stretch: float | None = None, spacing=None) -> str:
    """A 2D field at cell centres, filled by triangulating the centres, with optional velocity arrows. mask(x, y)
    hides triangles whose centre lies inside a body. stretch draws y that many times taller, for thin gaps."""
    import matplotlib.tri as mtri

    plt = _plt()
    x, y, values = np.asarray(x, float), np.asarray(y, float), np.asarray(values, float)
    tri = mtri.Triangulation(x, y)
    hide = np.zeros(len(tri.triangles), bool)
    if mask is not None:
        cx, cy = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
        hide |= np.asarray(mask(cx, cy), bool)
    if spacing is not None:      # a triangle longer than its corners' cells bridges a wall or a gap, so hide it
        t = tri.triangles
        e = np.max([np.hypot(x[t[:, i]] - x[t[:, j]], y[t[:, i]] - y[t[:, j]]) for i, j in ((0, 1), (1, 2), (2, 0))],
                   axis=0)
        hide |= e > 2.5 * np.asarray(spacing, float)[t].max(1)
    if hide.any():
        tri.set_mask(hide)
    span_x, span_y = np.ptp(x), np.ptp(y) * (stretch or 1)
    fig, ax = plt.subplots(figsize=(9, max(2.6, min(6.0, 9 * span_y / max(span_x, 1e-12)) + 1.2)), facecolor=BG)
    tc = ax.tricontourf(tri, values, 64, cmap=cmap)
    if vectors is not None:
        v = np.asarray(vectors, float)
        k = np.linspace(0, len(x) - 1, min(len(x), 400)).astype(int)
        ax.quiver(x[k], y[k], v[k, 0], v[k, 1] / (stretch or 1), color="white", alpha=0.5, width=0.0015)
    if stretch:
        ax.set_aspect(stretch)
    else:
        ax.set_aspect("equal")
    cb = fig.colorbar(tc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=FG)
    _style(ax, "mm", "mm", label)
    ax.grid(False)
    fig.tight_layout()
    return save(fig, path)


def moody(path: str | Path, re: float, f: float, rel_rough: float, f_cfd: float | None = None) -> str:
    """The Moody chart with this pipe's roughness curve, its operating point and the CFD friction factor."""
    from .correlations import churchill, colebrook

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    lam = np.logspace(2.7, np.log10(2300), 40)
    ax.plot(lam, 64 / lam, color="#8a8f96", lw=1.2)
    turb = np.logspace(np.log10(4000), 8, 120)
    for e in (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 0.02):
        ax.plot(turb, [colebrook(r, e) for r in turb], color="#3a3d42" if e != rel_rough else MARK, lw=1)
        ax.text(1.05e8, colebrook(1e8, e), f"{e:g}", color="#8a8f96", fontsize=7, va="center")
    trans = np.logspace(np.log10(2300), np.log10(4000), 20)
    ax.plot(trans, [churchill(r, rel_rough) for r in trans], color="#8a8f96", lw=1, ls=":")
    ax.plot(turb, [colebrook(r, rel_rough) for r in turb], color=MARK, lw=2, label=f"this pipe, e/D {rel_rough:.2g}")
    ax.plot([re], [f], "o", color=MARK, ms=9, mec=BG, label=f"operating point, Re {re:.3g}, f {f:.3g}")
    if f_cfd is not None:
        ax.plot([re], [f_cfd], "s", color="#4a90d9", ms=8, mec=BG, label=f"CFD, f {f_cfd:.3g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 2e8)
    ax.set_ylim(0.007, 0.13)
    _style(ax, "Reynolds number", "Darcy friction factor",
           "Moody chart (64/Re, Churchill through transition, Colebrook)")
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=9, loc="upper right")
    fig.tight_layout()
    return save(fig, path)


def surface_scalar(path: str | Path, X, tris, values, label: str, cmap: str = "coolwarm", arrow=None) -> str:
    """A part's surface coloured by a value at each vertex, seen from above and upstream, with the stream's
    direction drawn as an arrow when given."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt = _plt()
    X, tris, values = np.asarray(X, float), np.asarray(tris, int), np.asarray(values, float)
    fig = plt.figure(figsize=(9, 6), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    v = values[tris].mean(1)
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    norm = plt.Normalize(lo, hi if hi > lo else lo + 1)
    cm = plt.get_cmap(cmap)
    ax.add_collection3d(Poly3DCollection(X[tris], facecolors=cm(norm(v)), linewidths=0))
    c, r = (X.min(0) + X.max(0)) / 2, (X.max(0) - X.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    if arrow is not None:
        a = np.asarray(arrow, float) / np.linalg.norm(arrow)
        start = c - a * 1.6 * r
        ax.quiver(*start, *(a * 0.8 * r), color=FG, linewidth=2, arrow_length_ratio=0.25)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=FG)
    cb.set_label(label, color=FG)
    return save(fig, path)
