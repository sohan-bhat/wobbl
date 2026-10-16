"""wobbl.export: a finished run as a file to keep or send, a GIF, an MP4, a PNG or the report page.

    from wobbl.export import export
    done = export("wobbl-out/bracket", "gif", out="downloads", name="bracket-stress")
    done.path, done.note

The picture comes from the run's own scene.wobbl, drawn with Wobbl's numpy rasterizer (no OpenGL, no VTK): a slow
orbit over the part while the run plays, the input of a sweep ramping to the value asked about, and the verdict
and key number in a small caption. MP4 needs an ffmpeg already on the machine. Without one the clip is written as
a GIF and the note says so.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

FORMATS = ("gif", "mp4", "png", "report")
SIZE = {"gif": (1280, 720), "mp4": (1280, 720), "png": (1920, 1080)}
FPS = {"gif": 20.0, "mp4": 24.0}


@dataclass
class Exported:
    path: Path
    format: str
    frames: int = 1
    note: str = ""
    scene: Path | None = None

    def to_dict(self) -> dict:
        return {"path": str(self.path), "format": self.format, "frames": self.frames, "note": self.note,
                "scene": str(self.scene or "")}


class ExportError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


def find_scene(result_dir: str | Path) -> Path | None:
    """The run's scene: its own, the first step's of a run with several, or one built from its files."""
    out = Path(result_dir)
    order = []
    try:
        steps = json.loads((out / "simulation.json").read_text()).get("steps", [])
        order = [Path(s["result_dir"]) for s in steps if s.get("result_dir")]
    except (OSError, ValueError):
        pass
    places = [out] + order + sorted(p.parent for p in out.glob("*/scene.wobbl"))
    for p in places:
        if (p / "scene.wobbl").exists():
            return p / "scene.wobbl"
    try:
        from ..viz.adapters import build_scene
    except ImportError:
        return None
    for p in places:
        try:
            built = build_scene(p)
        except Exception:  # noqa: BLE001  a run that cannot be drawn is reported below, not raised from here
            built = None
        if built:
            return Path(built)
    return None


def export(result_dir: str | Path, fmt: str, out: str | None = None, name: str = "", log=None) -> Exported:
    """Write the run in result_dir as fmt (gif, mp4, png or report) to out: a file, a folder, a folder named in
    words such as 'downloads', or the results folder when left out. name is the file's stem."""
    from . import save

    fmt = fmt.lower().strip(".")
    if fmt not in FORMATS:
        raise ExportError(f"Wobbl exports gif, mp4, png or report, not {fmt}")
    result_dir = Path(result_dir)
    name = name or result_dir.name
    if fmt == "report":
        src = next((p for p in (result_dir / "report.html", *sorted(result_dir.glob("*/report.html")))
                    if p.exists()), None)
        if src is None:
            raise ExportError("this run wrote no report page", "Run it again without --quick to get report.html.")
        dest = save.destination(out, fmt, name, result_dir)
        shutil.copyfile(src, dest)
        return Exported(dest, fmt)
    scene = find_scene(result_dir)
    if scene is None:
        raise ExportError("this run has no 3D scene to draw", "Run it again without --quick, then export it.")
    from .clip import Scene, render, shots, still

    note = ""
    if fmt == "mp4" and not save.ffmpeg():
        fmt, note = "gif", "ffmpeg is not on this machine, so the clip is a GIF instead of an MP4"
    dest = save.destination(out, fmt, name, result_dir)
    sc = Scene(scene)
    if fmt == "png":
        pic = render(scene, [still(sc)], SIZE["png"], log)[0]
        return Exported(save.png(pic, dest), fmt, 1, note, scene)
    plan = shots(sc, FPS[fmt])
    if log:
        log("export", f"drawing {len(plan)} frames at {SIZE[fmt][0]}x{SIZE[fmt][1]}")
    frames = render(scene, plan, SIZE[fmt], log)
    if fmt == "mp4":
        save.mp4(frames, FPS[fmt], dest)
    else:
        save.gif(frames, FPS[fmt], dest)
    return Exported(dest, fmt, len(frames), note, scene)


__all__ = ["FORMATS", "ExportError", "Exported", "export", "find_scene"]
