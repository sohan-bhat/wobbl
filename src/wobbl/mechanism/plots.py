"""Pictures of a mechanism run: the needed torque against each motor's line, and the mechanism drawn at several
instants from MuJoCo's geom positions with matplotlib, so nothing opens a window or needs OpenGL."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..report.render import BG, FG, MARK

FAIL = "#ff4d4f"
SOFT = "#4a90d9"
DPI = 110


def _fig(w=8.0, h=4.5, ncols=1, nrows=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=DPI, facecolor=BG, squeeze=False)
    for ax in axes.ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3d42")
        ax.grid(alpha=0.15)
    return fig, axes


def _save(fig, path) -> str:
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG, pil_kwargs={"optimize": True})
    plt.close(fig)
    return str(path)


def torque_speed(path, panels: list[dict]) -> str:
    """One panel per drive: the torque the move needs at each speed against the motor's torque speed line and
    its continuous rating. Each panel dict has name, motor, w (rad/s) and tau (N m)."""
    n = len(panels)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = _fig(8.0, 2.9 * rows + 0.4, cols, rows)
    for ax, p in zip(axes.ravel(), panels):
        w = np.abs(np.asarray(p["w"])) * 60 / (2 * math.pi)
        tau = np.abs(np.asarray(p["tau"]))
        mot = p["motor"]
        top = max(float(tau.max()) if len(tau) else 0.0, mot.stall_torque if mot else 0.0) * 1.15 or 1.0
        if mot is not None:
            w0 = mot.no_load_speed * 60 / (2 * math.pi) if math.isfinite(mot.no_load_speed) else max(w.max(), 1) * 2
            if mot.curve == "box" or not math.isfinite(mot.no_load_speed):
                ax.plot([0, w0, w0], [mot.stall_torque, mot.stall_torque, 0], color=FG, lw=1.6, label="motor limit")
            else:
                ax.plot([0, w0], [mot.stall_torque, 0], color=FG, lw=1.6, label="motor line")
            ax.axhline(mot.rated, color=SOFT, ls="--", lw=1.1, label="continuous")
            ax.set_xlim(0, max(w0, float(w.max()) if len(w) else 0) * 1.05)
            over = mot.usage(p["tau"], p["w"]) > 1
        else:
            over = np.zeros(len(w), bool)
        ax.plot(w, tau, color=MARK, lw=1.8, label="needed")
        if over.any():
            ax.scatter(w[over], tau[over], color=FAIL, s=6, zorder=3, label="beyond the motor")
        ax.set_ylim(0, top)
        ax.set_title(p["name"], color=FG, fontsize=10)
        ax.set_xlabel("rpm", color=FG, fontsize=8)
        ax.set_ylabel("N m", color=FG, fontsize=8)
        ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return _save(fig, path)


def series(path, t, ys: dict, ylabel: str, hlines: dict | None = None, xlabel: str = "s") -> str:
    """Lines over time, with optional labeled limits."""
    fig, axes = _fig()
    ax = axes[0, 0]
    colors = [MARK, SOFT, "#3ccf7a", "#ff6b6b", "#c99cf2", "#f29a3f"]
    for (name, y), col in zip(ys.items(), colors * 3):
        ax.plot(t, y, color=col, lw=1.6, label=name)
    for (name, v), col in zip((hlines or {}).items(), [FAIL, SOFT, FG]):
        ax.axhline(v, color=col, ls="--", lw=1.1, label=name)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    return _save(fig, path)


def _outline(kind: str, size, pos, mat) -> np.ndarray | None:
    """Points on a geom's surface in world coordinates, enough for a convex outline."""
    R = np.asarray(mat).reshape(3, 3)
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    if kind == "box":
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * size[:3]
    elif kind == "sphere":
        u = np.concatenate([ring, ring[:, [0, 2, 1]], ring[:, [2, 0, 1]]])
        c = u * size[0]
    elif kind in ("capsule", "cylinder"):
        r, h = size[0], size[1]
        if kind == "capsule":
            u = np.concatenate([ring, ring[:, [0, 2, 1]], ring[:, [2, 0, 1]]]) * r
            c = np.concatenate([u + [0, 0, h], u - [0, 0, h]])
        else:
            c = np.concatenate([ring * r + [0, 0, h], ring * r - [0, 0, h]])
    else:
        return None
    return np.asarray(pos) + c @ R.T


def strobe(path, run, views=(("x", "z"),), caption_times: bool = True, highlight: set | None = None) -> str:
    """The mechanism at each recorded instant, faint early and solid at the end, in one or more side views."""
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    axis = {"x": 0, "y": 1, "z": 2}
    geoms = run.model_info["geoms"]
    n = len(views)
    fig, axes = _fig(8.0 if n > 1 else 6.4, 4.2, n, 1)
    frames = run.frames
    for ax, (hx, vy) in zip(axes.ravel(), views):
        i, j = axis[hx], axis[vy]
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for k, (t, xpos, xmat) in enumerate(frames):
            last = k == len(frames) - 1
            alpha = 0.95 if last else 0.12 + 0.35 * k / max(len(frames) - 1, 1)
            for g, info in enumerate(geoms):
                pts = _outline(info["type"], np.asarray(info["size"]), xpos[g], xmat[g])
                if pts is None:
                    continue
                p2 = pts[:, [i, j]]
                try:
                    hull = p2[ConvexHull(p2).vertices]
                except Exception:  # noqa: BLE001  flat or degenerate outline
                    hull = p2
                rgba = list(info["rgba"])
                col = FAIL if highlight and info["name"] in highlight and last else tuple(rgba[:3])
                ax.add_patch(Polygon(hull, closed=True, facecolor=col, edgecolor=col, alpha=alpha, lw=0.6))
                lo, hi = np.minimum(lo, p2.min(0)), np.maximum(hi, p2.max(0))
        if any(info["type"] == "plane" for info in geoms) and vy == "z":
            ax.axhline(0, color="#6b7078", lw=1.2)
            lo[1] = min(lo[1], 0)
        span = max(float(np.max(hi - lo)), 1e-6)
        mid = (lo + hi) / 2
        ax.set_xlim(mid[0] - 0.6 * span, mid[0] + 0.6 * span)
        ax.set_ylim(mid[1] - 0.6 * span, mid[1] + 0.6 * span)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{hx} (m)", color=FG, fontsize=8)
        ax.set_ylabel(f"{vy} (m)", color=FG, fontsize=8)
    if caption_times and frames:
        fig.suptitle(f"{len(frames)} instants from {frames[0][0]:.2g} s to {frames[-1][0]:.3g} s, latest solid",
                     color=FG, fontsize=9)
    return _save(fig, path)
