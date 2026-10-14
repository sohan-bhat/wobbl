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

    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(4)
    d = rng.normal(size=(6000, 3))
    X = 10 * d / np.linalg.norm(d, axis=1, keepdims=True)
    hull = ConvexHull(X)
    tris = hull.simplices.copy()
    flip = np.einsum("ij,ij->i", np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]]),
                     X[tris].mean(1)) < 0
    tris[flip] = tris[flip][:, ::-1]
    mesh = SimpleNamespace(X=X, tris=tris)
    pts, faces, nodes = P.surface(mesh, max_vertices=1000)
    assert 600 <= len(pts) <= 1000 and faces.max() < len(pts) and len(np.unique(faces)) == len(pts)
    assert np.allclose(mesh.X[nodes], pts)          # every vertex is a node of the mesh, so fields map exactly
    # still a closed sphere facing out: the triangles add up to most of its area and none points inward
    n = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    assert 0.5 * np.linalg.norm(n, axis=1).sum() > 0.9 * 4 * np.pi * 100
    assert (np.einsum("ij,ij->i", n, pts[faces].mean(1)) > 0).mean() > 0.99


def test_surface_keeps_both_sides_of_a_thin_wall():
    """A 0.5 mm plate decimated far coarser than its thickness keeps its two faces apart."""
    from types import SimpleNamespace

    k = 60
    g = np.linspace(0, 100, k)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    top = np.column_stack([xx.ravel(), yy.ravel(), np.full(k * k, 0.5)])
    X = np.vstack([top, top * [1, 1, 0]])
    i, j = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
    a = (i * k + j).ravel()
    quads = np.column_stack([a, a + k, a + k + 1, a + 1])
    up = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    tris = np.vstack([up, up[:, ::-1] + k * k])
    pts, faces, nodes = P.surface(SimpleNamespace(X=X, tris=tris), max_vertices=500)
    assert set(np.round(pts[:, 2], 3)) == {0.0, 0.5}
    z = pts[faces][:, :, 2]
    assert (np.ptp(z, axis=1) == 0).all()         # no triangle spans the wall


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
