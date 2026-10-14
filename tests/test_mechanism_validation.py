"""The mechanism validation cases are registered and the quick ones land on their exact answers. Skipped when
MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install
from wobbl.validation.cases import CASES, run_cases

MECH = [c.id for c in CASES if c.domain == "mechanism"]


def test_the_mechanism_cases_are_registered_with_exact_references():
    assert MECH == ["pendulum-period", "fourbar-sweep", "arm-statics", "gear-output", "collision-elastic",
                    "collision-restitution"]
    assert all(c.exact and c.tol <= 0.005 for c in CASES if c.domain == "mechanism")


@pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed")
def test_the_collisions_and_the_gear_train_land_on_their_answers(tmp_path):
    rows = run_cases(tmp_path, ["gear-output", "collision-elastic", "collision-restitution"], log=lambda m: None)
    for r in rows:
        assert r["pass"] and r["error"] < 1e-3, r
