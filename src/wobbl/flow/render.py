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
    _style(ax, "Reynolds number", "Darcy friction factor", "Moody chart (64/Re, Churchill through transition, Colebrook)")
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=9, loc="upper right")
    fig.tight_layout()
    return save(fig, path)
