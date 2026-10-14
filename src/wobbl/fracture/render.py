"""Pictures of a crack growth result: the path through the part, crack size against cycles, and how lives spread."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK


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
        return _path_numpy(path, np.asarray(X, float), np.asarray(tets), fronts, cycles, origin, label, starts)
    except Exception as e:  # noqa: BLE001  a picture is never a reason to lose a result: draw it plainly
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return _path_mpl(path, X, fronts, cycles, origin, label, str(e))


def _strip(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points and triangles of a strip through rows of equally many points, (rows, k, 3)."""
    n, k = rows.shape[:2]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(k - 1), indexing="ij")
    a = (i * k + j).ravel()
    return rows.reshape(-1, 3), np.vstack([np.column_stack([a, a + k, a + k + 1]),
                                           np.column_stack([a, a + k + 1, a + 1])])


def _path_numpy(path, X, tets, fronts, cycles, origin, label, starts=None):
    """Three layers: the crack surface and its fronts, the part over them at 22% opacity, and the part's sharp
    edges drawn through it all, as the eye sees a crack inside a part made of tinted glass."""
    from matplotlib.colors import ListedColormap

    from ..mesh.mesher import _boundary
    from ..report import render as R

    size = (1500, 950)
    lim = (0.0, max(float(np.nanmax(cycles)), 1.0) if len(cycles) else 1.0)
    pts, tris, vals, segs, crack_pts = [], [], [], [], []
    for n, side in enumerate(fronts):
        if starts is not None and n < len(starts) and len(side):
            P, T = _strip(_ribbon([np.asarray(starts[n], float), np.asarray(side[0], float)]))
            pts.append(P), tris.append(T + sum(len(q) for q in pts[:-1])), vals.append(np.full(len(P), np.inf))
        if len(side) < 2:
            continue
        rib = _ribbon(side)
        P, T = _strip(rib)
        pts.append(P), tris.append(T + sum(len(q) for q in pts[:-1]))
        vals.append(np.repeat(np.asarray(cycles[:len(rib)], float), rib.shape[1]))
        segs += [np.stack([f[:-1], f[1:]], axis=1) for f in map(np.asarray, side)]
        crack_pts.append(P)
    o = np.asarray(origin, float)
    crack = np.vstack(crack_pts) if crack_pts else o[None, :]
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    extent = float(np.linalg.norm(crack.max(0) - crack.min(0))) + 1e-9
    nrm = np.cross(crack[len(crack) // 3] - crack[0], crack[-1] - crack[0]) if len(crack) > 2 else np.array([1.0, -1, 1])
    view = nrm / max(np.linalg.norm(nrm), 1e-12)
    view = 0.8 * view + 0.35 * np.array([0.0, 0.0, 1.0]) + 0.2 * np.array([1.0, -1.0, 0.0])
    cam = R.camera(view, X, size, focus=crack.mean(0), zoom=float(np.clip(0.55 * span / extent, 1.2, 6.0)), span=span)
    bg = R._rgb(BG)
    if pts:
        surf = R.surface(np.vstack(pts), np.vstack(tris), {"c": np.concatenate(vals)})
        # the starting flaw has no cycle count, so it reads as above the scale and is drawn gray
        img = R.draw(size, cam, surf, surf.fields["c"], R.Look("plasma", lim, above="#9aa0a6"),
                     R.Lines(np.vstack(segs), color=FG, alpha=1.0, width=2.0) if segs else None, two_sided=True)
    else:
        img = np.broadcast_to(bg, (size[1], size[0], 3)).copy()
    part_tris = _boundary(tets, X, 2 if tets.shape[1] == 10 else 1)[0]
    part = R.surface(X, part_tris)
    shade, cover = R.draw(size, cam, part, np.zeros(len(part.P)), R.Look(ListedColormap(["#8a9099"])), coverage=True)
    img = img * (1 - 0.22 * cover[..., None]) + 0.22 * (shade - (1 - cover)[..., None] * bg)
    empty = R.Surface(np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)))
    _, lines = R.draw(size, cam, empty, np.zeros(0), R.Look(),
                      R.Lines(R.feature_edges(X, part_tris), alpha=1.0, width=1.2, depth_test=False), coverage=True)
    img = img * (1 - 0.5 * lines[..., None]) + 0.5 * lines[..., None] * R._rgb(R.EDGE)
    return R.compose(path, [R.Panel(img, cam, ("plasma", lim, "cycles when the front passed"), [R.Mark(o, label)])])


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
