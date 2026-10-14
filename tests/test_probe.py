"""Reading fields inside a mesh and across a cut, without VTK."""
from types import SimpleNamespace

import numpy as np
import pytest

from wobbl.mesh.probe import sample
from wobbl.verify.beam import cut_tets


@pytest.fixture(scope="module")
def bar(parts, tmp_path_factory):
    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh

    return build_mesh(prepare(parts["cantilever"], tmp_path_factory.mktemp("g")), Sizing(h=5.0))


def test_sample_reads_a_quadratic_field_exactly(bar):
    """Quadratic tetrahedra carry a quadratic field exactly, so sampling one anywhere inside gives it back, and a
    point outside the part reads as outside."""
    X = bar.X
    f = 3.0 + 0.2 * X[:, 0] - 0.5 * X[:, 1] * X[:, 2] + 0.01 * X[:, 0] ** 2
    rng = np.random.default_rng(2)
    pts = rng.uniform([1, 1, 1], [199, 19, 9], size=(200, 3))
    got, inside = sample(X, bar.tets, f, np.vstack([pts, [[250.0, 10, 5]]]))
    exact = 3.0 + 0.2 * pts[:, 0] - 0.5 * pts[:, 1] * pts[:, 2] + 0.01 * pts[:, 0] ** 2
    assert inside[:-1].all() and not inside[-1]
    assert np.abs(got[:-1] - exact).max() < 1e-6 * np.abs(exact).max()
    vec, _ = sample(X, bar.tets, np.column_stack([f, 2 * f]), pts[:5])
    assert vec.shape == (5, 2) and np.allclose(vec[:, 1], 2 * vec[:, 0])


def test_crack_plane_section_carries_the_axial_force(bar):
    """A bar pulled at 7 MPa: the cut square to it is the 20 x 10 section and carries 1400 N, tilted cuts too."""
    from wobbl.fracture.place import section

    S = np.zeros((len(bar.X), 6))
    S[:, 0] = 7.0
    unc = SimpleNamespace(mesh=bar, S=S)
    A, F = section(unc, SimpleNamespace(normal=np.array([1.0, 0, 0]), origin=np.array([80.3, 10, 5])))
    assert A == pytest.approx(200.0, rel=1e-9) and F == pytest.approx(1400.0, rel=1e-9)
    n = np.array([1.0, 0.25, 0.0]) / np.linalg.norm([1.0, 0.25, 0.0])
    A, F = section(unc, SimpleNamespace(normal=n, origin=np.array([100.0, 10, 5])))
    assert A == pytest.approx(200.0 / n[0], rel=1e-9)
    assert F == pytest.approx(7.0 * n[0] ** 2 * A, rel=1e-9)   # normal stress sigma n_x^2 over the tilted area


def test_quadratic_cut_counts_midside_values(bar):
    """Cutting the 10 node elements as eight linear ones lets a field that only bulges at midside nodes show."""
    f = np.zeros(len(bar.X))
    f[np.unique(bar.tets[:, 4:])] = 1.0
    f[np.unique(bar.tets[:, :4])] = 0.0
    n, o = np.array([1.0, 0, 0]), np.array([100.3, 10, 5])
    for quadratic, expect in ((False, 0.0), (True, None)):
        pts, faces, a, b, t = cut_tets(bar.X, bar.tets, n, o, quadratic=quadratic)
        v = f[a] + t * (f[b] - f[a])
        tri = pts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        assert area.sum() == pytest.approx(200.0, rel=1e-9)
        if expect is not None:
            assert np.abs(v).max() == expect
        else:
            assert (area * v[faces].mean(1)).sum() > 0.1 * area.sum()


def test_clipped_half_is_closed_and_holds_half_the_volume(bar):
    """Cutting the 200 x 20 x 10 bar at a slant leaves a closed surface facing out, whose enclosed volume (by the
    divergence theorem) is exactly the half on the kept side, with a linear field carried onto it exactly."""
    from wobbl.mesh.clip import carry, clip_surface

    n = np.array([1.0, 0.0, 0.3]) / np.linalg.norm([1.0, 0.0, 0.3])
    (a, b, t), faces = clip_surface(bar.X, bar.tets, [100.0, 10.0, 5.0], n)
    P = carry(bar.X, a, b, t)
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, count = np.unique(e, axis=0, return_counts=True)
    assert (count == 2).all()                                  # closed, no cracks along the cut
    tri = P[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    assert volume == pytest.approx(200 * 20 * 10 / 2, rel=1e-9)   # the plane passes through the bar's center
    assert ((P - [100.0, 10.0, 5.0]) @ n).max() < 1e-9
    f = 2.0 * bar.X[:, 0] - bar.X[:, 2]
    assert np.allclose(carry(f, a, b, t), 2.0 * P[:, 0] - P[:, 2])
    (a2, _, _), whole = clip_surface(bar.X, bar.tets)
    tri = bar.X[a2][whole]
    assert np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6 == pytest.approx(40000.0)
