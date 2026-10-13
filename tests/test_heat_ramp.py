"""A held temperature that starts away from the part's temperature: the ramp time, the saved times around it and
the CalculiX deck, all without a solver."""
import numpy as np
import pytest

from wobbl.fe import HeatModel
from wobbl.solvers import calculix as ccx
from wobbl.thermal import study as heat

AL = {"conductivity": 167.0, "density": 2.7e-9, "specific_heat": 8.96e8}   # 6061-T6 in mW, mm, t, s
ALPHA = 167.0 / (2.7e-9 * 8.96e8)                                          # mm2/s


@pytest.fixture(scope="module")
def bar(parts, tmp_path_factory):
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path_factory.mktemp("bar"))
    return build_mesh(prep, Sizing(h=5.0))


def _model(bar, duration, held, initial=20.0):
    """The 200 x 20 x 10 bar with x=min and x=max held at the given temperatures (None leaves a face free)."""
    m = HeatModel(bar, **AL, initial_temp=initial, transient=duration > 0, duration=duration)
    for name, x, temp in (("x=min", 0.0, held[0]), ("x=max", 200.0, held[1])):
        if temp is not None:
            m.fixed_temps.append((name, np.nonzero(np.abs(bar.X[:, 0] - x) < 1e-6)[0], temp))
    return m


def _smallest_element_time(bar, x):
    near = (np.abs(bar.X[bar.tets[:, :4], 0] - x) < 1e-6).any(axis=1)
    c = bar.X[bar.tets[near, :4]]
    v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    return (6 * np.sqrt(2) * v.min()) ** (2 / 3) / ALPHA


def test_no_ramp_when_every_held_face_starts_at_the_part_temperature(bar):
    assert heat.held_ramp(bar, _model(bar, 600.0, (20.0, None))) == (0.0, "")
    assert heat.held_ramp(bar, _model(bar, 600.0, (None, None))) == (0.0, "")
    assert heat.held_ramp(bar, _model(bar, 0.0, (120.0, None))) == (0.0, "")    # steady state


def test_ramp_is_a_tenth_of_the_time_heat_takes_to_cross_the_smallest_element(bar):
    tau = _smallest_element_time(bar, 0.0)
    run = 0.1 * tau / 6e-4      # long enough that a thousandth of it is longer, short enough for CalculiX
    ramp, why = heat.held_ramp(bar, _model(bar, run, (40.0, None)))
    assert ramp == pytest.approx(0.1 * tau, rel=1e-9) and "smallest element" in why
    # only the faces that jump count: a held face at the part's temperature changes nothing
    assert heat.held_ramp(bar, _model(bar, run, (40.0, 20.0)))[0] == pytest.approx(ramp, rel=1e-9)


def test_ramp_is_at_most_a_thousandth_of_a_short_run(bar):
    ramp, why = heat.held_ramp(bar, _model(bar, 0.2, (40.0, None)))
    assert ramp == pytest.approx(0.2e-3) and why == "a thousandth of the run"


def test_a_big_jump_over_a_long_run_ramps_as_quick_as_calculix_can_follow(bar):
    """Over ten hours CalculiX takes no increment under 36 ms, and each moves a temperature by at most 2 C, so a
    1000 C jump needs at least 500 of them, four times over."""
    ramp, why = heat.held_ramp(bar, _model(bar, 36000.0, (1020.0, None)))
    assert ramp == pytest.approx(4 * 0.036 * 1000 / 2.0) and "1000 C jump" in why


def test_deltmx_gives_a_small_jump_as_many_increments_as_a_big_one():
    assert ccx.transient_deltmx(0.0) == 2.0           # nothing jumps: the usual 2 C
    assert ccx.transient_deltmx(100.0) == 2.0
    assert ccx.transient_deltmx(30.0) == pytest.approx(0.6)
    assert ccx.transient_deltmx(2.0) == 0.25          # but never so small that the run crawls


def test_saved_times_follow_the_ramp_and_the_fast_response_after_it():
    t = heat.early_times(0.01, 7.5)
    assert t[:2] == [0.005, 0.01]
    assert np.allclose(np.diff(np.log(t[1:])), np.log(heat.EARLY_RATIO))
    assert t[-1] < 7.5 and t[-1] * heat.EARLY_RATIO ** 1.5 >= 7.5
    assert heat.early_times(1.0, 1.2) == [0.5, 1.0]


def test_deck_ramps_jumping_faces_and_holds_the_rest(bar, tmp_path):
    m = _model(bar, 60.0, (0.0, 20.0))      # x=min drops to 0 C, x=max stays at the starting 20 C
    m.ramp_time, _ = heat.held_ramp(bar, m)
    m.output_times = [30.0, 60.0]
    ccx.write_heat_inp(tmp_path / "heat.inp", m)
    deck = (tmp_path / "heat.inp").read_text()
    tr = f"{m.ramp_time:.10g}"
    # the amplitude holds temperatures, so a 0 C target works, and the boundary value of 1 takes them as they are
    assert f"*AMPLITUDE, NAME=RAMP1\n0.,20,{tr},0,60,0\n" in deck
    assert "*BOUNDARY, AMPLITUDE=RAMP1\nTFIX1,11,11,1.\n" in deck
    assert "*BOUNDARY\nTFIX2,11,11,20\n" in deck and "RAMP2" not in deck
    assert "*HEAT TRANSFER, DELTMX=0.4\n" in deck     # a 20 C jump: a fiftieth of it per increment
    first = float(deck.split("*HEAT TRANSFER, DELTMX=0.4\n")[1].split(",")[0])
    assert first == pytest.approx(m.ramp_time * 0.4 / 20)    # the first increment moves the face by DELTMX


def test_deck_without_a_jump_is_unchanged(bar, tmp_path):
    m = _model(bar, 60.0, (20.0, None))
    m.ramp_time, _ = heat.held_ramp(bar, m)
    m.time_step, m.output_times = 0.3, [30.0, 60.0]
    ccx.write_heat_inp(tmp_path / "heat.inp", m)
    deck = (tmp_path / "heat.inp").read_text()
    assert "*AMPLITUDE" not in deck and "*BOUNDARY\nTFIX1,11,11,20\n" in deck
    assert "*HEAT TRANSFER, DELTMX=2\n0.3,60,3e-05,3\n" in deck
