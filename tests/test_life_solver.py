"""Buckling, vibration and fatigue end to end with CalculiX, on parts small enough to run in seconds."""
import math

import numpy as np
import pytest

from conftest import needs_ccx

pytestmark = pytest.mark.solver

STEEL = "custom:E=200000,nu=0.3,yield=250,density=7850"


@needs_ccx
def test_euler_column_buckles_where_euler_says(tmp_path):
    from wobbl.life.buckling import BucklingStudy, run_buckling
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "col.step", 400, 10, 10)
    base = StructuralStudy(str(part), STEEL, [parse_load("1000 N -x @ x=max")], [parse_fixture("x=min")],
                           quality="fast")
    o = run_buckling(BucklingStudy(base), tmp_path / "out")
    euler = math.pi ** 2 * 200000 * (10 ** 4 / 12) / (2 * 400) ** 2 / 1000
    assert o.values["load_factor"] == pytest.approx(euler, rel=0.01)
    assert o.values["shape"] == "column" and o.verdict == "holds"
    checks = {c.name: c.status for c in o.checks}
    assert checks["hand calculation"] == "pass" and checks["force balance"] == "pass"
    # the column curve cuts a slender steel column by about a tenth
    assert 0.85 < o.values["knockdown"] < 0.95
    assert (tmp_path / "out" / "report.html").exists() and not (tmp_path / "out" / "work").exists()
