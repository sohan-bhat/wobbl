"""Scenes from what runs already write: each adapter reads a result folder's own files and builds its scene.

A domain that writes its scene directly through wobbl.viz.Scene needs none of this. These adapters cover the
playback.json shapes and the trajectory tables the domains wrote before the scene format existed, so every run has
a view today. Strength runs write theirs during the run, since the mesh and fields are not kept on disk.
"""
from __future__ import annotations

from pathlib import Path

from .common import read


def build_scene(out_dir: str | Path):
    """Build out_dir/scene.wobbl from the run's files. Returns the path, or None when nothing fits."""
    out = Path(out_dir)
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    kind = pb.get("kind") or pb.get("manifest", {}).get("kind") or ""
    domain = o.get("domain", "")
    if domain == "motion":
        from . import motion
        return motion.build(out, o=o)
    if kind == "mechanism":
        from . import mechanism
        return mechanism.build(out, pb)
    return None
