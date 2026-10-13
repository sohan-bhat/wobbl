"""The pieces of the transient energy balance that need no solver: exact integration weights, the consistent
matrices that account for held faces, and the time integral of a flow that starts at t = 0."""
import numpy as np
import pytest

from wobbl.thermal import study as heat


def test_element_matrices_integrate_the_shape_functions():
    M = heat._tet10_mass()
    assert np.allclose(M, M.T) and M.sum() == pytest.approx(1.0)
    assert np.allclose(M.sum(1)[:4], -1 / 20) and np.allclose(M.sum(1)[4:], 1 / 5)
    T = heat._tri6_mass()
    assert np.allclose(T, T.T) and T.sum() == pytest.approx(1.0)
    assert np.allclose(T.sum(1)[:3], 0.0) and np.allclose(T.sum(1)[3:], 1 / 3)


@pytest.fixture(scope="module")
def bar(parts, tmp_path_factory):
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path_factory.mktemp("bar"))
    return build_mesh(prep, Sizing(h=5.0))


def test_stored_heat_weights_are_exact_for_a_quadratic_field(bar):
    """The 200 x 20 x 10 bar: the weights give its volume and the exact integral of x squared."""
    w = heat._quad_vol_weights(bar)
    assert w.sum() == pytest.approx(200 * 20 * 10, rel=1e-9)
    exact = 200 ** 3 / 3 * 20 * 10
    assert w @ bar.X[:, 0] ** 2 == pytest.approx(exact, rel=1e-9)
    assert heat._vol_weights(bar) @ bar.X[:, 0] ** 2 != pytest.approx(exact, rel=1e-6)   # the even split is not


def test_held_terms_reduce_to_plain_integrals(bar):
    """When every node is held, the held capacity is the stored heat and the held surface term is the whole
    surface integral, which ties the consistent matrices to the plain weights."""
    held = np.ones(bar.n_nodes, bool)
    dT = 1.0 + 0.01 * bar.X[:, 0] + 1e-4 * bar.X[:, 1] ** 2
    assert heat._held_capacity(bar, held, dT) == pytest.approx(heat._quad_vol_weights(bar) @ dT, rel=1e-9)
    mask = np.ones(len(bar.tris), bool)
    w, _ = heat._tri_weights(bar, mask)
    field = np.vstack([dT, 2 * dT])
    assert np.allclose(heat._held_surface(bar, mask, held, field), field @ w, rtol=1e-9)
    nothing = np.zeros(bar.n_nodes, bool)
    assert heat._held_capacity(bar, nothing, dT) == 0 and not heat._held_surface(bar, mask, nothing, field).any()


@pytest.mark.parametrize("shape,exact", [
    (lambda t: 5.0 + 0 * t, lambda T: 5.0 * T),                    # settles before the first saved point
    (lambda t: 2.0 * np.sqrt(t), lambda T: 4 / 3 * T ** 1.5),      # grows like sqrt(t)
    (lambda t: 3.0 / np.sqrt(t), lambda T: 6.0 * np.sqrt(T)),       # a face held hotter than the part
])
def test_flow_from_the_start_is_integrated_over_the_first_interval(shape, exact):
    times = np.linspace(0, 600, 41)
    got = heat._integrate_from_start(shape(times[1:]), times)
    assert got == pytest.approx(exact(600.0), rel=0.01)


def test_cubic_follows_a_fast_fall_between_spaced_out_saved_times():
    """Heat through a thin plate's face that has just ramped to a new temperature dies away about as fast as the
    ramp, and it is saved at times a ratio of 1.3 apart from the ramp's end. The cubic integrates it within 0.3%,
    where the trapezoid rule overshoots by more than 2%."""
    ramp = 0.01
    t = ramp * heat.EARLY_RATIO ** np.arange(25)
    y = np.exp(-(t - ramp) / ramp)
    exact = ramp * (1 - np.exp(-(t[-1] - ramp) / ramp))
    assert heat._integrate(y, t) == pytest.approx(exact, rel=3e-3)
    assert heat._trapezoid(y, t) > 1.02 * exact


def test_cubic_starts_afresh_where_a_ramp_ends():
    """A flow that rises through the ramp and then holds has a kink at the ramp's end, and integrating each side
    on its own gets it exactly."""
    t = np.array([0.0, 0.5, 1.0, 1.3, 1.69, 2.2, 2.86, 3.7])
    y = np.minimum(t, 1.0)
    exact = 0.5 + (t[-1] - 1.0)
    assert heat._integrate(y, t, 1.0) == pytest.approx(exact, rel=1e-12)
    assert heat._integrate(y, t) != pytest.approx(exact, rel=1e-3)
    assert heat._integrate(y[:2], t[:2]) == pytest.approx(0.125)     # two points: a straight line
