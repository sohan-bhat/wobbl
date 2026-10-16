"""The frames of a clip: a slow orbit over the part while the run plays, its input ramping to the value asked about,
with the verdict and the key number in a caption.

A sweep over an input the user controls (a load, a throat, a supply) ramps from the start of the sweep to its
design value and holds there. A run over time plays from start to end and holds the last moment. Either way the
camera swings slowly to each side of the run's own view and back, so the clip loops without a jump.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .render import ACCENT, DIM, FG, caption, frame, orbit, start_view
from .sample import Scene

SWING = 26.0      # degrees to each side of the run's view


@dataclass
class Shot:
    u: float          # playback position, 0 .. frames - 1
    angle: float      # degrees about the up axis


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def design_position(sc: Scene) -> float:
    """Where the value the question asked about sits on an input sweep, as a playback position."""
    t = sc.times[:sc.n]
    d = sc.tl.get("design")
    if d is None or len(t) < 2:
        return float(sc.n - 1)
    idx = np.arange(len(t), dtype=float)
    return float(np.interp(d, t, idx) if t[-1] >= t[0] else np.interp(d, t[::-1], idx[::-1]))


def seconds(sc: Scene) -> float:
    if sc.n <= 1:
        return 5.0
    if sc.tl.get("kind") == "input":
        return 6.0
    return min(max(float(sc.tl.get("seconds") or 6.0), 4.0), 9.0) / 0.85


def shots(sc: Scene, fps: float) -> list[Shot]:
    n = max(2, int(round(seconds(sc) * fps)))
    out = []
    end = design_position(sc) if sc.tl.get("kind") == "input" else sc.n - 1.0
    for i in range(n):
        s = i / n
        if sc.n <= 1:
            u = 0.0
        elif sc.tl.get("kind") == "input":
            u = end * ease(s / 0.6)
        else:
            u = end * min(s / 0.85, 1.0)
        out.append(Shot(u, SWING * math.sin(2 * math.pi * s)))
    return out


def still(sc: Scene) -> Shot:
    """The one moment a picture shows: the design value of a sweep, or the end of a run over time."""
    return Shot(design_position(sc) if sc.tl.get("kind") == "input" else sc.n - 1.0, 0.0)


# ---------------------------------------------------------------- caption text
def _num(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "n/a"
    s = f"{v:.{max(digits, 1)}g}"
    return s if "e" not in s else f"{v:.3g}"


def _short(text: str, limit: int = 74) -> str:
    """The first sentence, cut at a comma or a word so it fits on one caption line."""
    first = text.split(". ")[0].strip().rstrip(".")
    if len(first) <= limit:
        return first
    cut = first[:limit]
    at = cut.rfind(",")
    return (cut[:at] if at > limit * 0.5 else cut[:cut.rfind(" ")]).rstrip(" ,")


def summary(scene_path: Path) -> tuple[str, str]:
    """The verdict word and the key number of the run a scene belongs to."""
    folder = scene_path.parent
    for p in (folder / "outcome.json", folder.parent / "simulation.json", folder / "simulation.json"):
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            verdict = str(d.get("verdict") or "")
            head = str(d.get("headline") or "")
            if not head and d.get("steps"):
                head = str(d["steps"][0].get("headline") or "")
            return verdict.upper(), _short(head)
    return "", ""


def lines(sc: Scene, u: float, verdict: str, key: str) -> list[tuple[str, str]]:
    title = str(sc.h.get("title") or "")
    if len(title) > 52:
        title = title[:52].rsplit(" ", 1)[0]
    top = "  ".join(x for x in (verdict, title) if x)
    live = []
    if sc.tl.get("kind") == "input" and sc.n > 1:
        live.append(f"{sc.tl.get('label', 'input')} {_num(sc.time(u))} {sc.tl.get('unit', '')}".strip())
    elif sc.n > 1 and sc.tl.get("unit") in ("s", "ms", "min", "h"):
        live.append(f"t {_num(sc.time(u))} {sc.tl.get('unit')}")
    shown = [r for r in sc.readouts(u) if r[0] != sc.tl.get("label")]
    for name, v, unit, digits in shown[:3 if not live else 2]:
        live.append(f"{name} {_num(v, digits)} {unit}".strip())
    text = ""
    for item in live:
        if len(text) + len(item) + 3 > 82:
            break
        text = f"{text}   {item}" if text else item
    return [(top, ACCENT), (key, FG), (text, DIM)]


# ---------------------------------------------------------------- rendering
_W: dict = {}


def _start(path: str, size: tuple[int, int], verdict: str, key: str, pad: int = 0) -> None:
    sc = Scene(path)
    _W.update(sc=sc, size=size, view=start_view(sc, size[0] / size[1]), verdict=verdict, key=key, pad=pad)


def _draw(shot: Shot) -> np.ndarray:
    sc, size = _W["sc"], _W["size"]
    pic = frame(sc, shot.u, orbit(_W["view"], sc.up, shot.angle), size)
    said = lines(sc, shot.u, _W["verdict"], _W["key"])
    said[-1] = (said[-1][0].ljust(_W["pad"]), said[-1][1])     # a steady caption box while the numbers change
    return caption(pic, said, size[1] / 720)


def workers(n: int) -> int:
    return max(1, min(n, (os.cpu_count() or 2) - 1, 6))


def render(scene_path: Path, plan: list[Shot], size: tuple[int, int], log=None) -> list[np.ndarray]:
    """Every shot drawn, several at a time on their own processes when there are many."""
    verdict, key = summary(scene_path)
    sc = Scene(scene_path)
    pad = max(len(lines(sc, s.u, verdict, key)[-1][0]) for s in plan)
    args = (str(scene_path), size, verdict, key, pad)
    k = workers(len(plan) // 8)
    if k <= 1:
        _start(*args)
        return [_draw(s) for s in plan]
    out = []
    with ProcessPoolExecutor(k, initializer=_start, initargs=args) as pool:
        for i, pic in enumerate(pool.map(_draw, plan, chunksize=4)):
            out.append(pic)
            if log and (i + 1) % 24 == 0:
                log("export", f"drew {i + 1} of {len(plan)} frames")
    return out
