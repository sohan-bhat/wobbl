"""Cutting a crack into a part, opening it, and reading K from the opening."""
import numpy as np
import pytest

from wobbl.fracture import shapes
from wobbl.fracture.crackmesh import CRACK_FACE, Sizing, build_cracked_mesh
from wobbl.structural.run import describe_part
from wobbl.validation import geometry as G


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    d = tmp_path_factory.mktemp("crackplate")
    part = G.box(d / "plate.step", 100.0, 150.0, 4.0)
    return describe_part(part, d / "geo")


def test_center_crack_is_cut_in_and_opened(plate):
    prep, feat = plate
    crack = shapes.center((50, 75, 2), (0, 1, 0), (1, 0, 0), 15.0, (2.0, 2.0), 1.0)
    cm = build_cracked_mesh(prep, crack, Sizing(h=8.0, h_front=1.0, h_crack=3.75))
    mesh = cm.mesh
    assert cm.area == pytest.approx(2 * 15.0 * 4.0, rel=1e-6)
    assert len(cm.fronts) == 2 and all(len(f) >= 4 for f in cm.fronts)
    assert cm.quarter_points > 0
    # both faces of the crack are now free surface, so the crack faces add twice their area to the boundary
    _, area, _ = mesh.tri_geometry()
    assert area[mesh.tri_surf == CRACK_FACE].sum() == pytest.approx(2 * cm.area, rel=1e-6)
    assert (mesh.tri_surf >= 0).sum() + (mesh.tri_surf == CRACK_FACE).sum() == len(mesh.tri_surf)
    # every crack face node away from the front has a copy at the same place, and the copies are used
    lo, up = cm.pairs.T
    assert np.allclose(mesh.X[lo], mesh.X[up])
    assert np.isin(up, mesh.tets).all() and np.isin(lo, mesh.tets).all()
    front = np.concatenate(cm.fronts)
    assert not np.isin(front, cm.pairs).any()


def test_edge_crack_at_a_hole_keeps_the_part_whole(tmp_path):
    part = G.plate_with_hole(tmp_path / "ph.step", 120.0, 60.0, 5.0, 10.0)
    prep, feat = describe_part(part, tmp_path / "geo")
    crack = shapes.through((5.0, 0, 2.5), (0, 1, 0), (1, 0, 0), 3.0, (2.5, 2.5), 1.0)
    cm = build_cracked_mesh(prep, crack, Sizing(h=6.0, h_front=0.3, h_crack=1.0))
    assert cm.area == pytest.approx(3.0 * 5.0, rel=1e-3)
    assert len(cm.fronts) == 1
    hole = [f.id for f in feat.faces.values() if f.role == "hole"]
    assert np.isin(cm.mesh.tri_surf, hole).any()      # the hole's faces are still found after the cut

