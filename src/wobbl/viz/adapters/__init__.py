"""Scenes from what runs already write: each adapter reads a result folder's own files and builds its scene.

A domain that writes its scene directly through wobbl.viz.Scene needs none of this. These adapters cover the
playback.json shapes, trajectory tables, heat fields and circuit board plans the domains wrote before the scene
format existed, so every run has a view today. Strength runs write theirs during the run, since their mesh and
fields are not kept on disk.
"""
from __future__ import annotations

from pathlib import Path

from .common import read

CHEM = ("solid-motor", "flame", "cell-runaway", "nozzle")
LIFE = ("vibration", "buckling", "drop", "fatigue")


def build_scene(out_dir: str | Path):
    """Build out_dir/scene.wobbl from the run's files. Returns the path, or None when nothing fits."""
    out = Path(out_dir)
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    man = pb.get("manifest") if isinstance(pb.get("manifest"), dict) else {}
    kind = pb.get("kind") or man.get("kind") or ""
    domain = o.get("domain", "")
    if domain == "motion":
        from . import motion
        return motion.build(out, o=o)
    if domain == "thermal" and (out / "fields.npz").exists():
        from . import thermal
        return thermal.build(out)
    if kind == "cooling" and (out / "heat" / "fields.npz").exists():
        from . import thermal
        return thermal.cooled(out)
    if kind == "channel" and (out / "fields.npz").exists():
        from . import thermal
        return thermal.fan_sink(out)
    if (out / "board.json").exists():
        from . import circuits
        return circuits.build(out)
    if kind == "mechanism":
        from . import mechanism
        return mechanism.build(out, pb)
    if kind == "solid-motor":
        from . import motor
        return motor.build(out, pb, o)
    if kind in CHEM:
        from . import chem
        return chem.build(out, pb)
    if man.get("format") == "wobbl-crack-playback":
        from . import fracture
        return fracture.build(out, pb)
    if kind in ("molding", "casting", "printing", "print"):
        from . import process
        return process.build(out, pb)
    if "grid" in pb and "h_re" in pb:
        from . import grids
        return grids.build_wave(out, pb)
    if "grid" in pb:
        from . import grids
        return grids.build_fields(out, pb)
    if isinstance(pb.get("fields"), list) and pb["fields"] and "components" in pb["fields"][0]:
        from . import grids
        return grids.build_flow(out, pb)
    if kind in LIFE or (pb.get("mesh") and isinstance(pb.get("frames"), list)):
        from . import life
        return life.build(out, pb)
    return None
