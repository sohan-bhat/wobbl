import numpy as np
import pytest

from wobbl.geometry.cad import GeometryError, prepare
from wobbl.geometry.faces import analyze
from wobbl.geometry.select import SelectionError, Selector
from wobbl.mesh.mesher import Sizing, build_mesh, build_surface_mesh, flat_elements


@pytest.fixture(scope="module")
def plate(parts, tmp_path_factory):
    prep = prepare(parts["plate_hole"], tmp_path_factory.mktemp("plate"))
    return prep, analyze(build_surface_mesh(prep))


def test_prepare_reads_step(plate):
    prep, _ = plate
    assert prep.n_bodies == 1
    assert prep.volume == pytest.approx(100 * 40 * 5 - np.pi * 16 * 5, rel=1e-3)
    assert np.allclose(prep.size, [100, 40, 5], atol=1e-3)


def test_features(plate):
    _, f = plate
    holes = f.holes()
    assert len(holes) == 1
    assert 2 * holes[0].radius == pytest.approx(8.0, rel=0.01)
    assert holes[0].length == pytest.approx(5.0, rel=0.02)
    assert f.thickness_typical == pytest.approx(5.0, rel=0.02)
    assert f.corners == []


def test_sharp_corner_found(parts, tmp_path):
    prep = prepare(parts["lbracket"], tmp_path)
    f = analyze(build_surface_mesh(prep))
    assert len(f.corners) == 1
    c = f.corners[0]
    assert c.angle_deg == pytest.approx(90, abs=2)
    assert c.length == pytest.approx(20, rel=0.02)


@pytest.mark.parametrize("sel,count", [("x=min", 1), ("x=max", 1), ("top", 1), ("holes", 1), ("hole d=8", 1),
                                       ("normal -z", 1), ("largest", 1), ("x=min or x=max", 2),
                                       ("top and within 5 of 50,20,5", 1)])
def test_selectors(plate, sel, count):
    _, f = plate
    assert len(Selector(f.faces, f.bbox).select(sel).faces) == count


def test_selector_errors_are_helpful(plate):
    _, f = plate
    with pytest.raises(SelectionError) as e:
        Selector(f.faces, f.bbox).select("hole d=20")
    assert "8" in str(e.value)
    with pytest.raises(SelectionError):
        Selector(f.faces, f.bbox).select("gibberish")


def test_second_order_mesh(plate):
    prep, f = plate
    m = build_mesh(prep, Sizing(h=4.0))
    assert m.tets.shape[1] == 10 and m.tris.shape[1] == 6
    assert m.volume() == pytest.approx(prep.volume, rel=2e-3)
    assert (m.tri_surf >= 0).all()
    # outward normals: the bottom face (z=0) points down
    n, a, c = m.tri_geometry()
    bottom = c[:, 2] < 1e-6
    assert (n[bottom][:, 2] < -0.99).all()
    assert flat_elements(m.X, m.tets) == 0


def test_flat_elements_are_found():
    # a sliver the 3D mesher once left in the NAFEMS LE1 membrane: four corners on one through thickness face
    X = np.array([[2692.8882, 644.293, 100.0], [2781.615, 634.887, 100.0], [2781.615, 634.887, 0.0],
                  [2692.8882, 644.293, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                  [0.0, 0.0, 10.0]])
    assert flat_elements(X, np.array([[0, 1, 2, 3]])) == 1
    assert flat_elements(X, np.array([[4, 5, 6, 7]])) == 0
    thin = X[[4, 5, 6, 7]].copy()
    thin[3, 2] = 0.01                                   # a thin but real element is kept
    assert flat_elements(thin, np.array([[0, 1, 2, 3]])) == 0


def test_missing_file(tmp_path):
    with pytest.raises(GeometryError):
        prepare(tmp_path / "nope.step", tmp_path)


def test_thickness_comes_from_the_cad_and_repeats(parts, tmp_path):
    prep = prepare(parts["lbracket"], tmp_path)
    a, b = build_surface_mesh(prep), build_surface_mesh(prep)
    assert a.samples is not None
    # every sample stands for a share of the CAD surface: two boxes, less the faces hidden where they overlap
    area = 2 * (80 * 6 + 80 * 20 + 6 * 20) + 2 * (6 * 70 + 70 * 20 + 6 * 20) - 2 * 6 * 6 - 4 * 6 * 20
    assert a.samples["weights"].sum() == pytest.approx(area, rel=1e-6)
    fa, fb = analyze(a), analyze(b)
    assert (fa.thickness_typical, fa.thickness_min) == (fb.thickness_typical, fb.thickness_min)
    assert fa.thickness_typical == pytest.approx(6.0, rel=0.02)


def test_thin_wall_estimate_moves_smoothly():
    from wobbl.geometry.faces import thickness_stats

    th = np.array([3.0, 6.0])
    # a 3 mm wall on 4% of the surface: the thinnest 5% is 4 parts at 3 mm and 1 part at 6 mm
    typical, thin = thickness_stats(th, np.array([4.0, 96.0]))
    assert typical == 6.0 and thin == pytest.approx(3.6)
    _, thin = thickness_stats(th, np.array([6.0, 94.0]))
    assert thin == pytest.approx(3.0)
