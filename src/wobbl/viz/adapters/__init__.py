"""Scenes from what runs already write: each adapter reads a result folder's own files and builds its scene."""
from __future__ import annotations

from pathlib import Path


def build_scene(out_dir: str | Path):
    """Build out_dir/scene.wobbl from the run's files. Returns the path, or None when nothing fits."""
    return None
