"""Charts for combustion and motor results, in the report's colors, saved as 256 color PNGs to keep them light."""
from __future__ import annotations

import io
from pathlib import Path

from ..report.render import BG, FG, MARK

BLUE, RED, QUIET = "#4a90d9", "#ff4d4f", "#9aa0a6"


def _figure(rows: int = 1, size=(8, 4.5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, 1, figsize=size, dpi=120, facecolor=BG, sharex=rows > 1)
    for ax in (axes if rows > 1 else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color("#3a3d42")
        ax.grid(alpha=0.15)
    return fig, axes


def _label(ax, xlabel: str, ylabel: str):
    if xlabel:
        ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)


def _save(fig, path: str | Path) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=BG, format="png")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256).save(path, optimize=True)
    return str(path)


def motor_chart(path, t, thrust, pressure, allowable: float | None = None, allow_label: str = "") -> str:
    """Thrust over the burn above, chamber pressure below with the casing's allowable pressure."""
    fig, (a1, a2) = _figure(2, (8, 6))
    a1.plot(t, thrust, color=MARK, lw=2)
    _label(a1, "", "thrust, N")
    a2.plot(t, [p / 1e6 for p in pressure], color=BLUE, lw=2, label="chamber pressure")
    if allowable:
        a2.axhline(allowable / 1e6, color=RED, ls="--", lw=1.2, label=allow_label)
        a2.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=9, loc="lower center")
    _label(a2, "time, s", "pressure, MPa")
    return _save(fig, path)


def sweep_chart(path, xs, ys, xlabel: str, ylabel: str, mark: tuple[float, float] | None = None,
                hline: float | None = None, hlabel: str = "", label: str | None = None,
                second: tuple[list, str] | None = None) -> str:
    """One quantity against another, with the operating point marked, an optional second curve dashed and an
    optional limit line."""
    fig, ax = _figure()
    ax.plot(xs, ys, color=MARK, lw=2, label=label)
    if second:
        ax.plot(xs, second[0], color=BLUE, lw=2, ls="--", label=second[1])
    if mark:
        ax.scatter([mark[0]], [mark[1]], s=40, color=FG, zorder=5)
    if hline is not None:
        ax.axhline(hline, color=RED, ls="--", lw=1.2, label=hlabel)
    if second or hline is not None:
        ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=9)
    _label(ax, xlabel, ylabel)
    return _save(fig, path)
