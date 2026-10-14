import numpy as np
import pytest

from wobbl.verify.beam import section_properties, slice_tets
from wobbl.verify.checks import Check, balance_check, confidence, convergence_check


def test_rectangle_section():
    # 20 x 6 rectangle in the yz plane, split into two triangles
    p = np.array([[0, 0, 0], [0, 20, 0], [0, 20, 6], [0, 0, 6]], float)
    tris = np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])
    A, G, I = section_properties(tris, np.array([1.0, 0, 0]))
    assert A == pytest.approx(120)
    assert np.allclose(G, [0, 10, 3])
    assert I[2, 2] == pytest.approx(20 * 6 ** 3 / 12)
    assert I[1, 1] == pytest.approx(6 * 20 ** 3 / 12)


def test_slice_of_a_bar_is_its_cross_section(parts, tmp_path):
    """A plane square to a 200 x 20 x 10 bar cuts its tetrahedra into exactly the 20 x 10 rectangle, tilted or not."""
    from wobbl.geometry.cad import prepare
    from wobbl.mesh.mesher import Sizing, build_mesh

    m = build_mesh(prepare(parts["cantilever"], tmp_path / "g"), Sizing(h=5.0))
    x = np.array([1.0, 0, 0])
    pts, faces = slice_tets(m.X, m.tets, x, [100.3, 7.0, 2.0])
    A, G, I = section_properties(pts[faces], x)
    assert A == pytest.approx(200.0, rel=1e-9)
    assert np.allclose(G, [100.3, 10.0, 5.0])
    assert I[2, 2] == pytest.approx(20 * 10 ** 3 / 12, rel=1e-9)
    assert I[1, 1] == pytest.approx(10 * 20 ** 3 / 12, rel=1e-9)
    n = np.array([1.0, 0.3, 0.0]) / np.linalg.norm([1.0, 0.3, 0.0])
    pts, faces = slice_tets(m.X, m.tets, n, [100.0, 10.0, 5.0])
    assert section_properties(pts[faces], n)[0] == pytest.approx(200.0 / n[0], rel=1e-9)
    assert len(slice_tets(m.X, m.tets, x, [300.0, 0, 0])[1]) == 0


def test_convergence_levels():
    assert convergence_check([100, 103, 103.5], 0.03, "s", "MPa").status == "pass"
    assert convergence_check([100, 101], 0.03, "s", "MPa").status == "warn"  # only two meshes
    assert convergence_check([100, 120, 140], 0.03, "s", "MPa").status == "fail"
    assert convergence_check([100], 0.03, "s", "MPa").status == "fail"


def test_balance():
    good = balance_check(np.array([0, 0, -100.0]), np.array([0, 0, 100.0]), np.zeros(3), np.zeros(3), 100)
    assert good.status == "pass"
    bad = balance_check(np.array([0, 0, -100.0]), np.array([0, 0, 90.0]), np.zeros(3), np.zeros(3), 100)
    assert bad.status == "fail"


def test_confidence_never_high_after_failure():
    lvl, why = confidence([Check("a", "pass", ""), Check("b", "fail", "broken")])
    assert lvl == "low" and any("broken" in w for w in why)
    lvl, _ = confidence([Check("a", "pass", ""), Check("b", "warn", "")])
    assert lvl == "medium"
    lvl, _ = confidence([Check("a", "pass", "")], [("medium", "a sharp corner")])
    assert lvl == "medium"
    assert confidence([Check("a", "pass", "")])[0] == "high"


def _level(gov, xyz, n=50000, t=20.0):
    from wobbl.structural.run import Level

    return Level(0, n, 5 * n, 4.0, 1.0, t, t, gov, list(xyz), gov, list(xyz), 0.1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
                 [0, 0, 0], [0, 0, 0])


def test_variant_refines_when_verdict_flips_or_peak_moves():
    from wobbl.structural.hotspots import Hotspot
    from wobbl.structural.run import variant_needs_refinement

    main = _level(120.0, (40, 9, 80))
    spot = Hotspot(1, [40, 9, 80], "feature", radius=4.0)
    # yield 165, target 2: 120 MPa misses the target, 50 MPa meets it
    assert variant_needs_refinement(main, _level(50.0, (-34, 0, 29)), spot, 165, 2) == "the verdict flips"
    assert "away" in variant_needs_refinement(main, _level(110.0, (-34, 0, 29)), spot, 165, 2)
    assert variant_needs_refinement(main, _level(115.0, (41, 9.5, 81)), spot, 165, 2) == ""
    assert "away" in variant_needs_refinement(main, _level(115.0, (41, 9.5, 81)), None, 165, 2)


def _budget(spent, seconds=600.0, clock=0.0):
    import time

    from wobbl.structural.budget import Budget

    return Budget(seconds, time.time() - clock, spent)


def test_stop_reason_matches_the_main_loop_rules():
    from wobbl.structural.run import QUALITY, _stop_reason

    q, p = QUALITY["normal"], (0, 0, 0)
    one = [_level(50.0, p)]
    assert _stop_reason(one, [], 0.03, 3, 4, q, _budget(100)) == ""
    assert _stop_reason(one, [], 0.03, 3, 4, q, _budget(590)) == "time budget"
    two = [_level(50.0, p), _level(50.5, p, n=60000)]
    assert _stop_reason(two, [], 0.03, 3, 4, q, _budget(100)) == ""        # two meshes are not enough evidence
    three = two + [_level(50.6, p, n=70000)]
    assert _stop_reason(three, [], 0.03, 3, 4, q, _budget(100)) == "converged"
    moving = [_level(40.0, p), _level(45.0, p, n=60000), _level(50.0, p, n=70000), _level(55.0, p, n=80000)]
    assert _stop_reason(moving, [], 0.03, 3, 4, q, _budget(100)) == "level limit"
    assert _stop_reason(moving[:2], [], 0.03, 3, 4, q, _budget(100)) == ""
    assert _stop_reason([_level(50.0, p, n=240000)], [], 0.03, 3, 4, q, _budget(100)) == "node limit"


def test_the_clock_never_decides_the_level_reached():
    """A busy or slow machine runs longer, but the study stops at the same level: the budget is spent by the
    modeled cost of each mesh. Only a run past three times its budget is stopped by the clock."""
    from wobbl.structural.budget import Budget, mesh_cost, solve_cost
    from wobbl.structural.run import QUALITY, _stop_reason

    q, one = QUALITY["normal"], [_level(50.0, (0, 0, 0), t=400.0)]
    assert _stop_reason(one, [], 0.03, 3, 4, q, _budget(100, clock=1500)) == ""
    assert _stop_reason(one, [], 0.03, 3, 4, q, _budget(100, clock=1900)) == "clock limit"
    b = Budget(600)
    b.charge_level(50000, solves=3)
    assert b.spent == pytest.approx(mesh_cost(50000) + 3 * solve_cost(50000))
    assert solve_cost(100000) > 2 * solve_cost(50000)   # direct solves grow faster than the mesh


def _variant(values, converged, refined=True, stopped_by="converged"):
    return {"primary": "distributed", "alternative": "rigid", "primary_peak": 120.0, "alternative_peak": values[-1],
            "alternative_levels": [{"level": i, "n_nodes": 50000 + 10000 * i, "h_min": 1.0, "governing": v,
                                    "governing_xyz": [0, 0, 0]} for i, v in enumerate(values)],
            "refined": refined, "converged": converged, "stopped_by": stopped_by}


def test_load_introduction_reports_the_variant_convergence():
    from wobbl.structural.outcome import load_introduction_check

    c, why = load_introduction_check(_variant([48.0, 54.8, 55.5], True), 165, 2, "on face 2 at (1, 2, 3)")
    assert c.status == "warn"
    assert "converges to 55.5 MPa on face 2" in c.summary
    assert "over 3 meshes refined around that peak: 48 -> 54.8 -> 55.5 MPa" in c.summary
    assert "depends on how the load enters" in why

    same, why = load_introduction_check(_variant([110.0], False, refined=False, stopped_by="not needed"), 165, 1.2)
    assert same.status == "pass" and "same spot" in same.summary and "same either way" in same.summary
    assert why == ""


def test_load_introduction_never_presents_an_unconverged_variant_as_a_result():
    from wobbl.structural.outcome import load_introduction_check

    c, why = load_introduction_check(_variant([48.0, 54.8], False, stopped_by="time budget"), 165, 2)
    assert "did not converge (stopped by time budget)" in c.summary
    assert "best mesh reached reads 54.8 MPa" in c.summary and "not a result" in c.summary
    assert "converges" not in c.summary
    assert c.status == "warn" and "may depend" in why

    c, why = load_introduction_check({**_variant([100.0, 112.0], False, stopped_by="level limit"),
                                      "primary_peak": 115.0}, 165, 1.2)
    assert c.status == "info" and "not a result" in c.summary and why == ""
