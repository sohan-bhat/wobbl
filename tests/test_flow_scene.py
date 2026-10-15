"""Flow playback as a 3D scene, and the run time budget of a scale resolving run."""
from __future__ import annotations

import json

import numpy as np

from wobbl.flow import playback, scene
from wobbl.viz.scene import load


def _field(nx=40, ny=20):
    x, y = np.meshgrid(np.linspace(0, 100, 400), np.linspace(0, 50, 200))
    u = np.full(x.size, 2.0)
    v = 0.2 * np.sin(x.ravel() / 20)
    return playback.grid_field("flow", x.ravel(), y.ravel(), {"u": u, "v": v}, {"u": "m/s", "v": "m/s"},
                               nx=nx, ny=ny, unit="mm")


def test_steady_playback_becomes_streamlines(tmp_path):
    playback.write(tmp_path, "case", "test", "about", [_field()])
    assert (tmp_path / "scene.wobbl").exists()
    head, _ = load(tmp_path / "scene.wobbl")
    kinds = [o["kind"] for o in head["objects"]]
    assert "lines" in kinds


def test_unsteady_playback_becomes_particles(tmp_path):
    frames = [{"t": 0.1 * i, "fields": [_field()]} for i in range(4)]
    playback.write(tmp_path, "case", "test", "about", [_field()], frames=frames)
    head, _ = load(tmp_path / "scene.wobbl")
    assert "points" in [o["kind"] for o in head["objects"]]


def test_a_run_without_plane_flow_writes_no_scene(tmp_path):
    playback.write(tmp_path, "case", "test", "about")
    assert scene.write(tmp_path) is None and not (tmp_path / "scene.wobbl").exists()
    assert json.loads((tmp_path / "playback.json").read_text())["fields"] == []


def test_scale_resolving_cost_grows_with_cells_and_time():
    from wobbl.flow.external import cost
    a, b = cost(100_000, 1.0, 1e-3), cost(200_000, 1.0, 1e-3)
    assert b["seconds"] == 2 * a["seconds"] and a["steps"] == 1000
