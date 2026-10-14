"""Charts of a flight: altitude and speed against time, and the path through the air, in the report's colors."""
from __future__ import annotations

from pathlib import Path

from ..report.render import BG, FG, MARK

PHASE_COLOR = {"rail": FG, "boost": MARK, "coast": "#4a90d9", "drogue": "#3ccf7a", "main": "#ff6b6b",
               "fall": "#4a90d9", "landed": "#9aa0a6"}
QUIET = "#9aa0a6"


def _axes(ax, xlabel: str, ylabel: str):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#3a3d42")
    ax.grid(alpha=0.15)


def _by_phase(ax, xs, ys, phases, labels: set):
    """One line per run of rows in the same phase, each joined to the next so the curve has no gaps."""
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            ph = phases[start]
            end = min(i + 1, len(phases))
            ax.plot(xs[start:end], ys[start:end], color=PHASE_COLOR.get(ph, MARK), lw=2,
                    label=ph if ph not in labels and ph != "landed" else None)
            labels.add(ph)
            start = i


def _save(fig, path: str | Path) -> str:
    """Write the chart as a PNG with a 256 color palette, about a third the size of full color, so reports and
    committed examples stay light. Flat chart colors lose nothing to the palette."""
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, facecolor=BG, format="png")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256).save(path, optimize=True)
    return str(path)


def _mark(ax, events, xkey: str, ykey: str):
    for e in events:
        ax.scatter([e[xkey]], [e[ykey]], s=22, color=FG, zorder=5)
        ax.annotate(e["name"], (e[xkey], e[ykey]), textcoords="offset points", xytext=(6, 6), color=FG, fontsize=8)


def flight_chart(path: str | Path, rows: list[dict], events: list[dict], vacuum: dict | None = None) -> str:
    """Altitude and speed against time, colored by phase, with the events marked. ``vacuum`` adds the same body's
    fall without air as dashed lines, {"t": [...], "alt": [...], "speed": [...]}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), dpi=150, facecolor=BG, sharex=True)
    t = [r["t_s"] for r in rows]
    phases = [r["phase"] for r in rows]
    labels: set = set()
    _by_phase(top, t, [r["alt_m"] for r in rows], phases, labels)
    _by_phase(bottom, t, [r["speed_mps"] for r in rows], phases, set(labels))
    if vacuum:
        top.plot(vacuum["t"], vacuum["alt"], color=QUIET, lw=1.2, ls="--", label="without air")
        bottom.plot(vacuum["t"], vacuum["speed"], color=QUIET, lw=1.2, ls="--", label="without air")
    shown = [e for e in events if e["name"] not in ("release", "liftoff", "drogue deploy")]
    _mark(top, [{**e, "t": e["t_s"], "z": e["alt_m"]} for e in shown], "t", "z")
    _mark(bottom, [{**e, "t": e["t_s"], "v": e["speed_mps"]} for e in shown], "t", "v")
    _axes(top, "", "altitude above the ground, m")
    _axes(bottom, "time, s", "speed, m/s")
    top.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def path_chart(path: str | Path, rows: list[dict], events: list[dict], wind: float) -> str:
    """Altitude against distance along the launch direction, the way the wind carries the flight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=BG)
    _by_phase(ax, [r["x_m"] for r in rows], [r["alt_m"] for r in rows], [r["phase"] for r in rows], set())
    shown = [e for e in events if e["name"] in ("burnout", "apogee", "main deploy", "landing", "impact")]
    _mark(ax, shown, "x_m", "alt_m")
    ax.scatter([0], [0], s=40, marker="^", color=MARK, zorder=6)
    xlabel = "distance from the start along the launch direction, m"
    if wind:
        xlabel += f" (wind {abs(wind):.3g} m/s toward {'-' if wind > 0 else '+'}x)"
    _axes(ax, xlabel, "altitude above the ground, m")
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)
