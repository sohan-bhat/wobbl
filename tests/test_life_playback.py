"""playback.json: typed arrays that decode back to the values, a decimated surface of real nodes, damage frames."""
import base64
import json

import numpy as np
import pytest

from wobbl.life import playback as P
from wobbl.life.fatigue import Block


def decode(d: dict) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(d["data"]), dtype="<" + np.dtype(d["dtype"]).str[1:]).reshape(d["shape"])
    a = a.astype(float)
    return d.get("offset", 0) + np.asarray(d.get("scale", 1.0)) * a


def test_quantized_arrays_decode_back():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(50, 3))
    back = decode(P.quantize(v, "int16"))
    assert np.abs(back - v).max() <= np.abs(v).max() / 32767 * 0.51
    pts = rng.uniform(-40, 90, size=(30, 3))
    faces = rng.integers(0, 30, size=(20, 3))
    m = P.mesh_block(pts, faces)
    assert np.abs(decode(m["vertices"]) - pts).max() < 130 / 65535
    assert (decode(m["triangles"]) == faces).all() and m["triangles"]["dtype"] == "uint16"


def test_surface_keeps_original_nodes():
    from types import SimpleNamespace

    import pyvista as pv

    sphere = pv.Sphere(radius=10, theta_resolution=80, phi_resolution=80).triangulate()
    tris = sphere.faces.reshape(-1, 4)[:, 1:]
    mesh = SimpleNamespace(X=np.asarray(sphere.points), tris=tris)
    pts, faces, nodes = P.surface(mesh, max_vertices=1000)
    assert len(pts) <= 1100 and faces.max() < len(pts)
    assert np.allclose(mesh.X[nodes], pts)          # every vertex is a node of the mesh, so fields map exactly


def test_damage_builds_up_block_by_block(tmp_path):
    blocks = [Block(100.0, 0.0, 1e4, "start"), Block(50.0, 0.0, 1e6, "run")]
    lives = [1e5, 1e7]
    doc = P.fatigue_doc(blocks, lives, 3.0, False, 3.0, {"sf": 1.4, "xyz": [1, 2, 3]})
    d, c = doc["frames"]["damage"], doc["frames"]["cycles"]
    assert d[0] == 0 and all(b >= a for a, b in zip(d, d[1:]))
    assert d[-1] == pytest.approx(3 * (1e4 / 1e5 + 1e6 / 1e7)) and c[-1] == pytest.approx(3 * (1e4 + 1e6))
    p = P.write(tmp_path / "playback.json", doc)
    assert json.loads(open(p).read())["kind"] == "fatigue"


def test_buckling_growth_follows_the_amplification_factor():
    from types import SimpleNamespace

    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    mesh = SimpleNamespace(X=X, tris=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    shape = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 0.5]])
    doc = P.buckling_doc(mesh, shape, 4.0, 3.0, 9.0, 400.0, 0.4, "L/1000")
    lam, bow = np.array(doc["frames"]["load_factor"]), np.array(doc["frames"]["bow_mm"])
    assert bow == pytest.approx(0.4 / (1 - lam / 4.0), rel=1e-4) and lam[-1] == pytest.approx(3.8)   # 5 decimals
    assert np.abs(decode(doc["shape"])).max() == pytest.approx(1.0, abs=0.01)
