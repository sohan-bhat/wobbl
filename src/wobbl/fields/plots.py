"""Pictures of field results: colored maps on the 2D mesh with part outlines, and curves.

Images use the report's dark theme and are written as palette PNGs, which keeps
each one well under 200 KB so result folders stay small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig, path: str | Path) -> str:
    """Write a figure as a palette PNG (lossless to the eye at these sizes, about a quarter of the bytes)."""
    plt = _mpl()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG, dpi=fig.dpi)
    plt.close(fig)
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        im.quantize(colors=160, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    except Exception:  # noqa: BLE001  keep the plain PNG
        pass
    return str(path)


def _axes(plt, size=(8, 5.2)):
    fig, ax = plt.subplots(figsize=size, dpi=120, facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    return fig, ax


def outline_edges(tris: np.ndarray, region: np.ndarray, skip: set[int] = frozenset()) -> np.ndarray:
    """Edges between triangles of different regions, and edges on the mesh boundary next to a part."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    r = np.concatenate([region, region, region])
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * (int(tris.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, e, r = key[order], e[order], r[order]
    out = []
    i = 0
    n = len(key)
    while i < n:
        j = i + 1
        while j < n and key[j] == key[i]:
            j += 1
        regs = set(r[i:j].tolist())
        if (j - i == 1 and not regs <= skip) or (len(regs) > 1):
            out.append(e[i])
        i = j
    return np.array(out) if out else np.zeros((0, 2), int)


def field_map(path: str | Path, X: np.ndarray, tris: np.ndarray, region: np.ndarray, tri_values: np.ndarray,
              label: str, title: str = "", mirror: bool = False, lines: np.ndarray | None = None,
              skip_outline: set[int] = frozenset(), view: tuple | None = None, cmap: str = "inferno",
              clim: tuple | None = None, mask: np.ndarray | None = None, marks: list | None = None,
              scale: float = 1e3, unit: str = "mm") -> str:
    """A colored map of one value per triangle. lines: node values whose contours are drawn (flux lines).
    mirror reflects an axisymmetric half model about its axis so the whole cross section shows."""
    plt = _mpl()
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    keep = np.ones(len(tris), bool) if mask is None else mask
    keep = keep & np.isfinite(tri_values)
    if view:
        w, h = view[1] - view[0], view[3] - view[2]
        fig, ax = _axes(plt, (float(np.clip(5.0 * w / h + 1.6, 4.8, 10.5)), 5.2))
    else:
        fig, ax = _axes(plt)
    Xs = X * scale
    sides = [1.0, -1.0] if mirror else [1.0]
    vmin, vmax = clim if clim else (float(np.nanmin(tri_values[keep])), float(np.nanpercentile(tri_values[keep], 99.5)))
    pc = None
    edges = outline_edges(tris, region, skip_outline)
    if mirror and len(edges):
        tol = 1e-9 * np.ptp(X[:, 1])
        edges = edges[~((np.abs(X[edges[:, 0], 0]) < tol) & (np.abs(X[edges[:, 1], 0]) < tol))]
    for s in sides:
        tr = Triangulation(s * Xs[:, 0], Xs[:, 1], tris[keep])
        pc = ax.tripcolor(tr, facecolors=tri_values[keep], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                          rasterized=True)
        if lines is not None:
            lv = np.linspace(np.nanmin(lines), np.nanmax(lines), 24)[1:-1]
            ax.tricontour(tr, lines, levels=lv, colors="#e8e8e8", linewidths=0.5, alpha=0.55)
        if len(edges):
            seg = np.stack([Xs[edges[:, 0]] * [s, 1], Xs[edges[:, 1]] * [s, 1]], axis=1)
            ax.add_collection(LineCollection(seg, colors=MARK, linewidths=0.8, alpha=0.9))
    for x, y, txt in marks or []:
        ax.plot([x * scale], [y * scale], "o", color="#4a90d9", ms=4)
        ax.annotate(txt, (x * scale, y * scale), xytext=(6, 6), textcoords="offset points", color=FG, fontsize=8)
    if view:
        ax.set_xlim(view[0] * scale, view[1] * scale)
        ax.set_ylim(view[2] * scale, view[3] * scale)
    ax.set_aspect("equal")
    ax.set_xlabel(unit, color=FG, fontsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=10)
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label, color=FG, fontsize=9)
    cb.ax.tick_params(colors=FG, labelsize=8)
    fig.tight_layout()
    return save(fig, path)


def curve(path: str | Path, x, ys: dict, xlabel: str, ylabel: str, hline: float | None = None, hlabel: str = "",
          marks: list | None = None, logx: bool = False, logy: bool = False) -> str:
    """Curves with dots at the solved points, an optional limit line and labeled marks (x, y, text)."""
    plt = _mpl()
    fig, ax = _axes(plt, (8, 4.5))
    colors = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b", "#c58af9"]
    for (name, y), col in zip(ys.items(), colors):
        ax.plot(x, y, color=col, lw=2, label=name, marker="o" if len(x) < 30 else None, ms=4)
    if hline is not None:
        ax.axhline(hline, color="#ff4d4f", ls="--", lw=1.2, label=hlabel or None)
    for mx, my, txt in marks or []:
        ax.plot([mx], [my], "o", color="#ff4d4f", ms=6)
        ax.annotate(txt, (mx, my), xytext=(8, -12), textcoords="offset points", color=FG, fontsize=8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.grid(alpha=0.15)
    if len(ys) > 1 or hline is not None:
        ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    return save(fig, path)


def polar(path: str | Path, theta, gains: dict, label: str = "gain, dBi") -> str:
    """A radiation pattern cut: theta in radians from the antenna axis, gain in dBi."""
    plt = _mpl()
    fig = plt.figure(figsize=(5.6, 5.2), dpi=120, facecolor=BG)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    colors = [MARK, "#4a90d9", "#3ccf7a"]
    lo = min(float(np.max(g)) for g in gains.values()) - 30
    for (name, g), col in zip(gains.items(), colors):
        ax.plot(theta, np.maximum(g, lo), color=col, lw=2, label=name)
    ax.set_rlim(lo, max(float(np.max(g)) for g in gains.values()) + 2)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(label, color=FG, fontsize=9)
    if len(gains) > 1:
        ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8, loc="lower right")
    fig.tight_layout()
    return save(fig, path)
