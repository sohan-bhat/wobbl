"""Where an export goes and how each kind of file is written: a GIF with Pillow, an MP4 through ffmpeg when the
machine already has it, a PNG, or the report page copied out.

Folders people name in words (downloads, desktop, documents, pictures, videos) resolve the way each system lays
them out: the home folder on macOS, the XDG user folders on Linux, and the profile folder on Windows, where the
desktop may live under OneDrive.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SUFFIX = {"gif": ".gif", "mp4": ".mp4", "png": ".png", "report": ".html"}
FOLDERS = {"downloads": ("XDG_DOWNLOAD_DIR", "Downloads"), "desktop": ("XDG_DESKTOP_DIR", "Desktop"),
           "documents": ("XDG_DOCUMENTS_DIR", "Documents"), "pictures": ("XDG_PICTURES_DIR", "Pictures"),
           "videos": ("XDG_VIDEOS_DIR", "Movies" if sys.platform == "darwin" else "Videos")}
GIF_LIMIT = 15_000_000


def known_folder(name: str) -> Path:
    """A folder named in words, as this system lays it out."""
    key, default = FOLDERS[name]
    home = Path.home()
    if sys.platform.startswith("linux"):
        cfg = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "user-dirs.dirs"
        try:
            m = re.search(rf'^{key}="?([^"\n]+)"?', cfg.read_text(), re.M)
        except OSError:
            m = None
        if m:
            return Path(m.group(1).replace("$HOME", str(home))).expanduser()
    if sys.platform == "win32" and name == "desktop":
        one = Path(os.environ.get("OneDrive") or home / "OneDrive") / "Desktop"
        if one.is_dir() and not (home / "Desktop").is_dir():
            return one
    return home / default


def destination(out: str | None, fmt: str, name: str, result_dir: Path) -> Path:
    """The file to write: out when it names a file, a clearly named file in out when it is a folder or a folder
    named in words, and in the results folder otherwise. An existing file is never overwritten."""
    suffix = SUFFIX[fmt]
    if out and out.lower() in FOLDERS:
        folder = known_folder(out.lower())
    elif out and Path(out).suffix.lower() in SUFFIX.values():
        p = Path(out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.with_suffix(suffix)
    else:
        folder = Path(out).expanduser() if out else Path(result_dir)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{name}{suffix}"
    k = 2
    while p.exists():
        p = folder / f"{name}-{k}{suffix}"
        k += 1
    return p


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def png(pic: np.ndarray, path: Path) -> Path:
    from PIL import Image

    Image.fromarray(pic).save(path, optimize=True)
    return path


def gif(frames: list[np.ndarray], fps: float, path: Path, limit: int = GIF_LIMIT) -> Path:
    """A looping GIF on one palette taken from frames across the clip, without dithering so the flat dark
    background stays flat and small. Over the size limit it drops to fewer colors, then to every other frame."""
    from PIL import Image

    def write(fr, colors, delay):
        picks = fr[:: max(1, len(fr) // 8)][:8]
        sheet = Image.fromarray(np.concatenate([p[::2, ::2] for p in picks], axis=0))
        pal = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        ims = [Image.fromarray(p).quantize(palette=pal, dither=Image.Dither.NONE) for p in fr]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=delay, loop=0, optimize=False,
                    disposal=1)
        return path.stat().st_size

    delay = int(round(1000 / fps))
    for fr, colors, d in ((frames, 255, delay), (frames, 128, delay), (frames[::2], 128, delay * 2),
                          (frames[::3], 96, delay * 3)):
        if write(fr, colors, d) <= limit:
            break
    return path


def mp4(frames: list[np.ndarray], fps: float, path: Path) -> Path:
    """An H.264 MP4 that plays everywhere, through the ffmpeg already on this machine."""
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
           f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
    finally:
        p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg could not write the video")
    return path
