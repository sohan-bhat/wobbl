"""Shell models of thin walled parts: the midsurface, carrying supports and loads over, CalculiX's shell deck, and
a clamped plate against plate theory."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import _write, needs_ccx


@pytest.fixture(scope="module")
def open_box(tmp_path_factory):
    """An open bottom box, 80 x 60 x 30 mm with 2 mm walls: a lid on four sides, standing on its rim at z = 0."""
    d = tmp_path_factory.mktemp("shells")

    def build(occ):
        outer = occ.addBox(0, 0, 0, 80, 60, 30)
        inner = occ.addBox(2, 2, -1, 76, 56, 29)
        occ.cut([(3, outer)], [(3, inner)])

    path = _write(d / "box.step", build)
    from wobbl.structural.run import describe_part

    return describe_part(str(path), d / "work")


def test_point_triangle_distance():
    from wobbl.shells.mapping import point_triangle_distance

    A, B, C = (np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]]))
    for p, d, u, v in (((0.25, 0.25, 2.0), 2.0, 0.25, 0.25), ((2.0, 0.0, 0.0), 1.0, 1.0, 0.0),
                       ((-1.0, -1.0, 0.0), np.sqrt(2), 0.0, 0.0)):
        got = point_triangle_distance(np.array([p], float), A, B, C)
        assert got[0][0] == pytest.approx(d) and got[1][0] == pytest.approx(u) and got[2][0] == pytest.approx(v)


def test_midsurface_of_a_box(open_box):
    from wobbl.shells.mesh import midsurface, thin_walled

    prep, feats = open_box
    assert thin_walled(feats)[0]
    sh = midsurface(prep, feats, 6.0)
    assert np.allclose(sh.thickness, 2.0, atol=0.05)
    # the lid's midsurface sits a wall's half below its top, the sides a half inside, and the rim stays on z = 0
    lid = sh.X[:, 2] > 28.5
    assert np.allclose(sh.X[lid, 2], 29.0, atol=1e-6)
    assert sh.X[:, 0].min() == pytest.approx(1.0, abs=1e-6) and sh.X[:, 0].max() == pytest.approx(79.0, abs=1e-6)
    assert sh.X[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert 0.95 < sh.volume() / prep.volume < 1.05
    # every element faces out, toward the skin it came from
    en, a, _ = sh.element_geometry()
    assert (a > 0).all() and (np.einsum("ij,ij->i", en, sh.normals[sh.tris[:, :3]].mean(1)) > 0.5).all()


def test_supports_and_loads_carry_over(open_box):
    from wobbl.geometry.select import Selector
    from wobbl.shells.ccx import element_node_normals, folded_expansion
    from wobbl.shells.mapping import SolidSurface, nodes_on, transfer_forces
    from wobbl.shells.mesh import midsurface
    from wobbl.structural.build import traction_forces

    prep, feats = open_box
    sh = midsurface(prep, feats, 6.0)
    solid = SolidSurface(sh.solid)
    sel = Selector(feats.faces, feats.bbox)
    rim = nodes_on(sh, solid, sel.select("z=min").tri_mask(solid))
    assert len(rim) and np.allclose(sh.X[rim, 2], 0.0, atol=1e-6)
    top = sel.select("top").tri_mask(solid)
    _, a, _ = solid.tri_geometry(top)
    F = transfer_forces(sh, solid.X, traction_forces(solid, top, np.array([0, 0, -50.0]) / a.sum()))
    assert F.sum(0) == pytest.approx([0, 0, -50.0], abs=1e-9)
    moved = np.flatnonzero(np.abs(F).sum(1) > 0)
    assert np.allclose(sh.X[moved, 2], 29.0, atol=1e-6)
    # the lid and the sides fold at 90 degrees: the fold's nodes get two normals, no expansion folds over
    en, _, _ = sh.element_geometry()
    nn = element_node_normals(sh)
    assert not folded_expansion(sh, nn, en).any()
    edge = np.flatnonzero((np.abs(sh.X[:, 2] - 29.0) < 1e-6) & (np.abs(sh.X[:, 0] - 1.0) < 1e-6))
    found = {tuple(np.round(nn[e, k], 3)) for e in range(sh.n_elems) for k in range(6) if sh.tris[e, k] in edge}
    assert (0.0, 0.0, 1.0) in found and (-1.0, 0.0, 0.0) in found


@pytest.mark.solver
@needs_ccx
def test_clamped_disk_matches_plate_theory(tmp_path):
    """A clamped circular plate under pressure: center deflection q a^4 / (64 D), exact for a thin plate."""
    from wobbl.structural.shell_path import analyze_shell
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    a, t, q, E, nu = 50.0, 1.0, 0.01, 200000.0, 0.3
    part = _write(tmp_path / "disk.step", lambda occ: occ.addCylinder(0, 0, 0, 0, 0, t, a))
    st = StructuralStudy(str(part), f"custom:E={E},nu={nu},yield=250", [parse_load(f"{q} MPa @ top")],
                         [parse_fixture("cylinders")], quality="fast")
    o = analyze_shell(st, tmp_path / "out", render=False)
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    assert o.max_disp_mm == pytest.approx(q * a ** 4 / (64 * D), rel=0.03)
    names = {c.name: c.status for c in o.checks}
    cross = next(c for c in o.checks if c.name == "solid cross check")   # a 1 mm wall is thin enough to skip it by size
    assert names["force balance"] == "pass" and (cross.status in ("pass", "warn") or cross.detail.get("skipped") == "size")
    assert (tmp_path / "out" / "report.html").exists() and not (tmp_path / "out" / "work").exists()
