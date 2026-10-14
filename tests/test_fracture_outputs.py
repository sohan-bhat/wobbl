"""The crack growth pictures and the playback file, from made up path data with no solver."""
import json

import numpy as np
import pytest

from wobbl.fracture import playback
from wobbl.fracture.render import growth_image, life_image, path_image


@pytest.fixture(scope="module")
def box_mesh(tmp_path_factory):
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.structural.run import describe_part
    from wobbl.validation import geometry as G

    d = tmp_path_factory.mktemp("box")
    prep, _ = describe_part(G.box(d / "box.step", 40, 20, 5), d / "geo")
    return build_mesh(prep, Sizing(h=4.0))


def _fronts(n=4):
    z = np.linspace(0, 5, 7)
    return [np.column_stack([np.full(7, 2.0 + k), np.full(7, 10.0), z]) for k in range(n)]


def test_pictures(box_mesh, tmp_path):
    m = box_mesh
    start = np.column_stack([np.zeros(7), np.full(7, 10.0), np.linspace(0, 5, 7)])
    p = path_image(tmp_path / "path.png", m.X, m.tets, [_fronts()], [0, 10, 20, 30], [0, 10, 2.5], "origin",
                   [start])
    lives = np.exp(np.random.default_rng(1).normal(10, 0.5, 2000))
    g = growth_image(tmp_path / "growth.png", [0, 10, 20, 30], [2, 3, 4, 5], [0, 10, 20, 30], [2, 3, 4, 5])
    life = life_image(tmp_path / "life.png", lives, {"B50": 2.2e4, "B10": 1.2e4, "B1": 7e3}, 1e4)
    for f in (p, g, life):
        assert f.endswith(".png") and (tmp_path / f.split("/")[-1]).stat().st_size > 1000


def test_playback_is_small_valid_json(box_mesh, tmp_path):
    m = box_mesh
    steps = [{"step": k, "size_mm": 2.0 + k, "cycles": 1000.0 * k, "k_lead": 8.0 + k,
              "fronts": [{"xyz": f.tolist(), "k": np.full((7, 3), 8.0 + k).tolist()}]} for k, f in enumerate(_fronts())]
    steps[-1]["cycles"] = float("inf")
    lives = np.exp(np.random.default_rng(1).normal(10, 0.5, 2000))
    lives[:5] = np.inf
    out = playback.write(tmp_path / "playback.json", part=playback.part_surface(m.X, m.tris), origin=[0, 10, 2.5],
                         normal=[0, 1, 0], steps=steps, starts=[np.zeros((7, 3))],
                         growth=(np.array([2, 3, 4, 5.0]), np.array([0, 1000, 2000, 3000.0])), lives=lives,
                         percentiles={"B50": 2.2e4, "B10": 1.2e4, "B1": float("inf")}, target=1e4, pof=0.03,
                         inspection={"first": 500.0, "repeat": 300.0}, verdict="fails", headline="h",
                         material="Aluminum 7075-T6", spectrum="R = 0.1", end="the step limit")
    d = json.loads(open(out).read())
    assert d["manifest"]["format"] == "wobbl-crack-playback" and d["manifest"]["units"]["length"] == "mm"
    assert len(d["crack"]["steps"]) == 4 and d["crack"]["steps"][-1]["cycles"] is None
    assert d["life"]["percentiles"]["B1"] is None and len(d["life"]["cdf"]["cycles"]) == 61
    assert d["part"]["triangles"] and len(d["part"]["vertices"][0]) == 3
    assert (tmp_path / "playback.json").stat().st_size < 300_000


def test_part_shape(box_mesh):
    from wobbl.fracture.place import PartShape

    part = PartShape(box_mesh)
    got = part.inside(np.array([[20, 10, 2.5], [1, 1, 1], [41, 10, 2.5], [20, 10, 6], [-0.5, 5, 2]]))
    assert got.tolist() == [True, True, False, False, False]
    assert part.ahead([20, 10, 2.5], [1, 0, 0]) == pytest.approx(20.0)
    # a ray lying in the top face runs to the far edge rather than stopping on the face it lies in
    assert part.ahead([20, 10, 5.0], [0, 1, 0], 1e-6) == pytest.approx(10.0)
