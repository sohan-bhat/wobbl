"""Buckling, vibration and fatigue run on every example part and on parts they were never tuned for (thin, chunky,
curved, shells, an STL mesh and two touching bodies), each with its own supports and loads. Slow: about a minute
per part and run, so it is marked slow and runs with the solver tests."""
import math

import pytest

from conftest import needs_ccx
from life_generality import parts, run_part

pytestmark = [pytest.mark.solver, pytest.mark.slow]


@needs_ccx
@pytest.mark.parametrize("part", parts(), ids=lambda p: p.name.replace(" ", "-"))
def test_every_run_answers_on_any_part(part, tmp_path):
    rows = run_part(part, tmp_path)
    failed = [f"{r['analysis']}: {r.get('error')} ({r.get('where', '')})" for r in rows if not r["ok"]]
    assert not failed, failed
    for r in rows:
        assert r["verdict"] in ("ok", "holds", "marginal", "fails")
        assert r["playback_kB"] is None or r["playback_kB"] < 300
        if "frequencies" in r:
            assert all(f > 0 and math.isfinite(f) for f in r["frequencies"])
