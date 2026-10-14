"""Buckling, vibration and fatigue registered with wobbl simulate: questions reach them, and what a question says
about speeds, sweeps and cycles is read."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRUT = str(ROOT / "examples" / "strut-buckling" / "strut.py")


def test_registry_loads_and_routes_life_questions():
    from wobbl.simulate import errors, route

    assert not errors
    for q, want in [("will it buckle under 2 kN?", "buckling"),
                    ("does it resonate between 4000 and 9000 rpm?", "vibration"),
                    ("will it survive a 2 g sine sweep from 20 to 2000 Hz?", "vibration"),
                    ("will it last a million cycles at 150 N?", "fatigue"),
                    ("how long does it last under this load history?", "fatigue")]:
        p = route(STRUT, q)
        assert p.domain == "part" and want in p.names, (q, p.names, p.reasons)


def test_questions_give_speeds_sweeps_and_cycles():
    from wobbl.life.domain import _rpm, cycles_in, sweep_in

    assert _rpm("from 4000-9000 rpm") == ["4000-9000 rpm"]
    assert cycles_in("a million cycles") == "1e+06" and cycles_in("500k cycles") == "500000"
    assert sweep_in("a 5 N unbalance swept 10-500 Hz", loads=True) == "10-500 hz"
    assert sweep_in("a 2 g sine sweep from 20 to 2000 Hz", loads=False) == "2 g z 20-2000 hz"
    assert sweep_in("a 45 g motor at the tip, shaken 10-200 Hz", loads=False) is None   # 45 g is a mass
    assert sweep_in("does it resonate between 100 and 200 Hz", loads=True) is None


@pytest.mark.parametrize("name", ["vibration", "buckling", "fatigue"])
def test_each_analysis_takes_its_options(name):
    from wobbl.simulate import get

    a = get("part", name)
    assert a.legacy == f"wobbl {name}"
    want = {"vibration": ("sine", "psd", "rpm", "mass", "load"), "buckling": ("shape", "load", "fix"),
            "fatigue": ("history", "block", "ratio", "cycles")}[name]
    assert all(a.takes(o) for o in want), [o for o in want if not a.takes(o)]
