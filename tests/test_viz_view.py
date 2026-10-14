"""simulate's page plays every step's scene under the question, and wobbl view opens any run's view."""
import json

import numpy as np
from click.testing import CliRunner

from wobbl.viz import Scene, simulation_view


def step(d):
    d.mkdir(parents=True)
    s = Scene("a step", domain="test")
    s.timeline([0.0, 1.0])
    s.body("box", [1, 1, 1], pos=np.zeros((2, 3)))
    s.save(d)


def test_simulation_page_holds_each_step(tmp_path):
    step(tmp_path / "stress")
    step(tmp_path / "heat")
    (tmp_path / "simulation.json").write_text(json.dumps({
        "thing": "bracket.step", "question": "will it hold?", "verdict": "pass",
        "situation": {"supports": ["wall x=min"], "loads": ["40 N down @ top"]},
        "steps": [{"analysis": "stress", "result_dir": str(tmp_path / "stress"), "headline": "holds"},
                  {"analysis": "heat", "result_dir": "heat", "headline": "stays cool"},
                  {"analysis": "broken", "error": "no solver"}]}))
    page = simulation_view(tmp_path / "simulation.json")
    text = page.read_text()
    assert page.name == "simulation.html"
    assert text.count("data-wobbl-scene=") == 2 and "Held by wall x=min." in text and "will it hold?" in text


def test_nothing_to_play(tmp_path):
    (tmp_path / "simulation.json").write_text(json.dumps({"steps": [{"analysis": "x", "result_dir": "nowhere"}]}))
    assert simulation_view(tmp_path / "simulation.json") is None


def test_view_command(tmp_path):
    from wobbl.cli.main import cli

    step(tmp_path / "run")
    r = CliRunner().invoke(cli, ["view", str(tmp_path / "run"), "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["view"].endswith("view.html")
    r = CliRunner().invoke(cli, ["view", str(tmp_path), "--json"])
    assert r.exit_code == 2 and "no 3D view" in r.output
