"""Pictures for life results: mode shapes, a frequency map against the excitation, response curves and S-N plots.

Three dimensional renders are saved as JPEG at a modest size, since a report embeds every picture and a heatmap of
smooth color compresses several times smaller that way. Charts stay PNG, where flat color is smaller.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK

SIZE = (1200, 750)
JPEG_QUALITY = 82


def to_jpeg(png: Path) -> str:
    """Re-encode a PNG screenshot as JPEG and remove the PNG."""
    from PIL import Image

    jpg = Path(png).with_suffix(".jpg")
    with Image.open(png) as im:
        im.convert("RGB").save(jpg, "JPEG", quality=JPEG_QUALITY, optimize=True)
    Path(png).unlink(missing_ok=True)
    return str(jpg)


def _exaggerated(X: np.ndarray, U: np.ndarray, share: float = 0.08) -> tuple[np.ndarray, float]:
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    umax = float(np.linalg.norm(U, axis=1).max())
    k = share * span / umax if umax > 0 else 1.0
    return X + k * U, k


def _across(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """A view that looks across a mode's main motion and across the part's longest side, tilted a little so it
    still reads as three dimensional: the bend of a mode shows in full."""
    from ..report.render import ISO_DIRS

    w = np.linalg.norm(U, axis=1)
    # the main direction of motion, with sign, from the second moment of the displacements
    vals, vecs = np.linalg.eigh((U * w[:, None]).T @ U)
    d = vecs[:, -1]
    a = np.eye(3)[int(np.argmax(X.max(0) - X.min(0)))]
    side = np.cross(d, a)
    if np.linalg.norm(side) < 0.3:     # the mode moves along the longest side: an isometric view across it
        dirs = ISO_DIRS / np.linalg.norm(ISO_DIRS, axis=1)[:, None]
        return dirs[int(np.argmax(-np.abs(dirs @ d) + 0.1 * (dirs[:, 2] > 0)))]
    side /= np.linalg.norm(side)
    if side[2] < 0 or (abs(side[2]) < 1e-9 and side[1] > 0):
        side = -side
    v = side + 0.35 * a + 0.25 * np.cross(side, a)
    return v / np.linalg.norm(v)


def render_modes(path: str | Path, X, tets, shapes: list[np.ndarray], labels: list[str],
                 size: tuple[int, int] | None = None) -> str:
    """One or more mode shapes side by side, each drawn deflected and colored by how far it moves (normalized),
    over the outline of the part at rest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k = len(shapes)
    size = size or ((SIZE[0], SIZE[1]) if k == 1 else (520 * k, 520))
    try:
        from ..mesh.mesher import _boundary
        from ..report.render import OVERVIEW, Lines, Look, Panel, camera, compose, draw, feature_edges, fitted

        tris = _boundary(tets, X, 2 if tets.shape[1] == 10 else 1)[0]
        edges = Lines(feature_edges(X, tris), alpha=0.5, width=1.2)
        look = Look("viridis", (0.0, 1.0))
        each = (size[0] // k, size[1])
        panels = []
        for U, label in zip(shapes, labels):
            mag = np.linalg.norm(U, axis=1)
            Xd = _exaggerated(X, U)[0]
            box = OVERVIEW if k == 1 else (0.05, 0.12, 0.95, 0.95)
            cam = camera(_across(X, U), np.vstack([Xd[np.unique(tris)], X]), each, box=box)
            surf = fitted(cam, each, Xd, tris, {"m": mag / max(float(mag.max()), 1e-30)})
            panels.append(Panel(draw(each, cam, surf, surf.fields["m"], look, edges), cam,
                                (look.cmap, look.clim, "relative\nmotion") if k == 1 else None, text=label,
                                text_size=11 if k > 1 else 13))
        compose(path, panels)
    except Exception:  # noqa: BLE001  a flat matplotlib drawing of the deflected surface
        if os.environ.get("WOBBL_DEBUG"):
            raise
        _modes_mpl(path, X, tets, shapes, labels)
    return to_jpeg(path)


def _modes_mpl(path, X, tets, shapes, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..mesh.mesher import _boundary

    tris, _, _, _ = _boundary(tets[:, :4] if tets.shape[1] == 10 else tets, X, 1)
    fig = plt.figure(figsize=(4.5 * len(shapes), 4.0), dpi=110, facecolor=BG)
    for j, (U, label) in enumerate(zip(shapes, labels)):
        ax = fig.add_subplot(1, len(shapes), j + 1, projection="3d", facecolor=BG)
        Xd, _ = _exaggerated(X, U)
        m = np.linalg.norm(U, axis=1)
        v = (m / max(m.max(), 1e-30))[tris].mean(1)
        ax.add_collection3d(Poly3DCollection(Xd[tris], facecolors=plt.get_cmap("viridis")(v), linewidths=0))
        lo, hi = Xd.min(0), Xd.max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off()
        ax.set_title(label, color=FG, fontsize=9)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def render_field(path: str | Path, X, tets, tris, values, label: str, unit: str, mark_xyz, mark_label: str,
                 cmap: str = "turbo") -> str:
    from ..report.render import render_scalar

    p = render_scalar(Path(path), X, tets, tris, np.asarray(values, float), label, unit, cmap, mark_xyz, mark_label,
                      clim=[0.0, float(np.max(values)) if np.max(values) > 0 else 1.0])
    return to_jpeg(Path(p))


def _axes(figsize=(8, 4.2)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)
    return plt, fig, ax


def _finish(plt, fig, ax, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def frequency_map(path, freqs: np.ndarray, weights: np.ndarray, tones: list[tuple[float, str]],
                  bands: list[tuple[float, float, str]], margin: float,
                  inputs: list[tuple[float, float, str]] = ()) -> str:
    """Natural frequencies as stems, as tall as the share of mass each moves along the excitation. Tones and bands
    to stay clear of are red, with their keep out margin, and the band of a test input is grey behind them."""
    plt, fig, ax = _axes()
    top = max(float(np.max(weights)) if len(weights) else 1.0, 0.05) * 1.15
    for lo, hi, name in inputs:
        ax.axvspan(lo, hi, color="#9aa0a6", alpha=0.10, label=name)
    reds = ["#ff4d4f", "#ff9f43", "#c56cf0"]
    for (lo, hi, name), c in zip(bands, reds * 4):
        ax.axvspan(lo / (1 + margin), hi * (1 + margin), color=c, alpha=0.08)
        ax.axvspan(lo, hi, color=c, alpha=0.25, label=name)
    for f, name in tones:
        ax.axvspan(f * (1 - margin), f * (1 + margin), color="#ff4d4f", alpha=0.18)
        ax.axvline(f, color="#ff4d4f", lw=1.4, ls="--", label=name)
    ax.vlines(freqs, 0, np.maximum(weights, 0.01 * top), color=MARK, lw=2.4)
    ax.scatter(freqs, np.maximum(weights, 0.01 * top), color=MARK, s=18, zorder=3, label="natural frequencies")
    for k, f in enumerate(freqs[:12]):
        ax.annotate(f"{k + 1}", (f, max(weights[k], 0.01 * top)), textcoords="offset points", xytext=(0, 5),
                    ha="center", color=FG, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, top)
    ax.set_xlabel("Hz", color=FG)
    ax.set_ylabel("share of mass moved", color=FG)
    return _finish(plt, fig, ax, path)


def response_curve(path, f: np.ndarray, curves: dict[str, np.ndarray], ylabel: str, marks=None) -> str:
    plt, fig, ax = _axes()
    colors = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b"]
    for (name, y), c in zip(curves.items(), colors):
        ax.plot(f, y, color=c, lw=2, label=name)
    for x, y, name in marks or []:
        ax.scatter([x], [y], color="#ff4d4f", s=30, zorder=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hz", color=FG)
    ax.set_ylabel(ylabel, color=FG)
    return _finish(plt, fig, ax, path)


def sn_plot(path, curve, points: list[tuple[float, float, str]], extend_to: float = 1e9) -> str:
    """The S-N line used, with each operating point at its equivalent fully reversed amplitude and life."""
    plt, fig, ax = _axes()
    n = np.geomspace(1e3, max(extend_to, curve.ne * 10), 200)
    s = np.array([curve.strength(x) for x in n])
    ax.plot(n, s, color=MARK, lw=2.2, label="S-N line used")
    ax.axvline(curve.ne, color="#6b7078", lw=1, ls=":")
    for life, amp, name in points:
        x = life if np.isfinite(life) else n[-1]
        ax.scatter([x], [amp], color="#ff4d4f", s=34, zorder=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cycles", color=FG)
    ax.set_ylabel("fully reversed amplitude, MPa", color=FG)
    return _finish(plt, fig, ax, path)


def haigh_plot(path, curve, sn: float, points: list[tuple[float, float, str]], model: str) -> str:
    """The mean stress diagram: Goodman, Gerber and Soderberg lines for the required life, and the operating
    points."""
    plt, fig, ax = _axes()
    m = np.linspace(0, curve.su, 200)
    ax.plot(m, sn * (1 - m / curve.su), color=MARK if model == "goodman" else "#6b7078", lw=2, label="Goodman")
    ax.plot(m, sn * (1 - (m / curve.su) ** 2), color=MARK if model == "gerber" else "#6b7078", lw=1.5, ls="--",
            label="Gerber")
    ms = np.linspace(0, curve.sy, 100)
    ax.plot(ms, sn * (1 - ms / curve.sy), color=MARK if model == "soderberg" else "#6b7078", lw=1.5, ls=":",
            label="Soderberg")
    for mean, amp, name in points:
        ax.scatter([max(mean, 0.0)], [amp], color="#ff4d4f", s=34, zorder=3, label=name)
    ax.set_xlim(0, curve.su * 1.02)
    ax.set_ylim(0, max(sn, max((a for _, a, _ in points), default=0)) * 1.15)
    ax.set_xlabel("mean stress, MPa", color=FG)
    ax.set_ylabel("stress amplitude, MPa", color=FG)
    return _finish(plt, fig, ax, path)


def convergence(path, nodes: list[int], series: dict[str, list[float]], ylabel: str) -> str:
    plt, fig, ax = _axes(figsize=(7, 3.6))
    colors = [MARK, "#4a90d9", "#3ccf7a", "#ff6b6b"]
    for (name, y), c in zip(series.items(), colors):
        ax.plot(nodes, y, marker="o", color=c, lw=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("mesh nodes", color=FG)
    ax.set_ylabel(ylabel, color=FG)
    return _finish(plt, fig, ax, path)
