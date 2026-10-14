"""Pipe and drag studies from start to finish, without a CFD solver: the numbers against hand calculations, the
verdicts, the checks that should fire and the fixes."""
from __future__ import annotations

import math

import pytest

from wobbl.flow import correlations as C
from wobbl.flow.tools import build


def _run(kind, params, tmp_path):
    st, run = build(kind, params)
    return run(st, tmp_path / kind)


def test_laminar_pipe_matches_hagen_poiseuille(tmp_path):
    """0.2 L/min of water through 3 m of 4 mm tube: dp = 128 mu L Q / (pi d^4)."""
    o = _run("pipe", {"length": "3 m", "diameter": "4 mm", "flow": "0.2 L/min"}, tmp_path)
    from wobbl.flow.fluids import get_fluid
    mu = get_fluid("water", 20).mu
    want = 128 * mu * 3 * (0.2e-3 / 60) / (math.pi * 0.004 ** 4)
    assert o.values["dp_Pa"] == pytest.approx(want, rel=1e-9)
    assert o.confidence == "high" and o.verdict == "ok"
    assert (tmp_path / "pipe" / "images" / "moody.png").stat().st_size < 150_000


def test_turbulent_pipe_adds_fittings_and_suggests_a_bore(tmp_path):
    p = {"length": "40 m", "diameter": "16 mm", "flow": "30 L/min", "fittings": "6 elbow90, globe-valve",
         "max_dp": "50 kPa", "material": "steel"}
    o = _run("pipe", p, tmp_path)
    v = 30e-3 / 60 / (math.pi * 0.016 ** 2 / 4)
    from wobbl.flow.fluids import get_fluid
    w = get_fluid("water", 20)
    re = w.rho * v * 0.016 / w.mu
    f = C.colebrook(re, 4.5e-5 / 0.016)
    assert o.values["dp_friction_Pa"] == pytest.approx(f * 40 / 0.016 * 0.5 * w.rho * v * v, rel=1e-9)
    assert o.values["dp_fittings_Pa"] == pytest.approx((6 * 0.3 + 8.0) * 0.5 * w.rho * v * v, rel=1e-9)
    assert o.verdict == "fails" and o.confidence == "medium"
    bore = next(f for f in o.fixes if f["kind"] == "diameter")
    d = float(bore["title"].split()[4]) / 1000
    st, _ = build("pipe", {**p, "diameter": f"{d * 1000} mm"})
    from wobbl.flow.pipe import solve_pipe
    assert solve_pipe(st).total == pytest.approx(0.9 * 50e3, rel=0.02)


def test_ducts_and_transition_are_read(tmp_path):
    o = _run("pipe", {"length": "10 m", "width": "200 mm", "height": "100 mm", "flow": "300 CFM", "fluid": "air",
                      "material": "galvanized"}, tmp_path)
    assert o.values["re"] > 4000 and any(c.name == "entrance length" for c in o.checks)
    o = _run("pipe", {"length": "5 m", "diameter": "10 mm", "flow": "1.4 L/min"}, tmp_path)
    assert any(c.name == "flow regime" and c.status == "warn" for c in o.checks)
    with pytest.raises(ValueError):
        build("pipe", {"length": "5 m", "flow": "1 L/min"})
    with pytest.raises(ValueError):
        build("pipe", {"length": "5 m", "diameter": "10 mm", "flow": "1 L/min", "fittings": "3 wormholes"})


def test_drag_on_a_sphere_and_the_drag_crisis(tmp_path):
    o = _run("drag", {"shape": "sphere", "size": "4 mm", "speed": "0.05 m/s", "fluid": "water"}, tmp_path)
    re = o.values["re"]
    assert o.values["cd"] == pytest.approx(C.sphere_cd(re))
    area = math.pi * 0.004 ** 2 / 4
    assert o.values["drag_N"] == pytest.approx(o.values["cd"] * o.values["dynamic_pressure_Pa"] * area)
    assert next(c for c in o.checks if c.name == "second curve").status == "pass"
    ball = _run("drag", {"shape": "sphere", "size": "22 cm", "speed": "20 m/s"}, tmp_path)
    assert ball.confidence == "low" and any(c.name == "drag crisis" and c.status == "warn" for c in ball.checks)


def test_finite_cylinder_and_a_drag_limit(tmp_path):
    o = _run("drag", {"shape": "cylinder", "size": "30 mm", "speed": "10 m/s", "length": "0.6 m",
                      "max_drag": "1 N"}, tmp_path)
    assert o.values["cd"] == pytest.approx(C.cylinder_cd(o.values["re"]) * 0.91, rel=0.01)   # L/D 20
    assert o.verdict == "fails" and {f["kind"] for f in o.fixes} == {"shape", "speed"}
    with pytest.raises(ValueError):
        _run("drag", {"shape": "banana", "size": "1 m", "speed": "1 m/s"}, tmp_path)
