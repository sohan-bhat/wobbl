"""The manufacturing analyses through wobbl simulate: routing, the plan preview, and a small molding run."""
import json

import pytest


@pytest.mark.parametrize("question,names", [
    ("will this cast in A356 without shrinkage porosity", ["casting"]),
    ("where should the riser go if I pour at 700 C", ["casting"]),
    ("will this lid short shot in polystyrene", ["molding"]),
    ("does it fill before it freezes with a gate at 50,30,10", ["molding"]),
])
def test_questions_route_to_the_process(parts, question, names):
    from wobbl.simulate import errors, route

    assert not {k: v for k, v in errors.items() if "process" in k}
    p = route(parts["plate_hole"], question)
    assert p.domain == "part" and p.names == names, p.reasons


def test_the_plan_shows_the_guesses_without_solving(parts, tmp_path):
    from click.testing import CliRunner

    from wobbl.cli.main import cli

    r = CliRunner().invoke(cli, ["simulate", parts["plate_hole"], "will this short shot in polycarbonate",
                                 "--plan", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "resin pc" in r.output and "no gate given" in r.output and "--max-pressure" in r.output
    assert not (tmp_path / "o" / "outcome.json").exists()


def test_a_small_plate_fills_end_to_end(tmp_path):
    from wobbl.process.molding import MoldingStudy, run_molding
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "plate.step", 40, 20, 1.5)
    o = run_molding(MoldingStudy(str(part), quality="fast", mesh_size=2.0), tmp_path / "out")
    assert o.domain == "molding" and o.verdict == "ok"
    v = o.values
    assert v["gate_placed"] and not v["short_shot"]
    assert 0 < v["pressure_hot_MPa"] <= v["pressure_cold_MPa"] < v["p_max_MPa"]
    status = {c.name: c.status for c in o.checks}
    assert status["volume balance"] == "pass" and status["thin walls"] == "pass"
    pb = json.loads((tmp_path / "out" / "playback.json").read_text())
    assert pb["manifest"]["kind"] == "molding" and pb["frames"]["time_s"]
    assert (tmp_path / "out" / "playback.json").stat().st_size < 300_000
