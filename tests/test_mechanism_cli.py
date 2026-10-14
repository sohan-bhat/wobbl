"""wobbl mechanism, plain questions and the solve_mechanism MCP tool."""
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.ask import detect_domain
from wobbl.cli.main import cli
from wobbl.mechanism import install
from wobbl.mechanism.tools import read_mechanism_question

needs_mujoco = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed")


@pytest.mark.parametrize("q,domain", [
    ("can two mg996r servos lift 500 g with my robot arm?", "mechanism"),
    ("will this four bar linkage bind?", "mechanism"),
    ("what speed does this gearbox give?", "mechanism"),
    ("will the camera stand tip over with a 2 kg lens?", "mechanism"),
    ("will the gear housing crack under 200 N?", "strength"),
    ("will a blue LED on 5V with a 47 ohm resistor fry?", "circuit"),
])
def test_questions_reach_the_mechanism_domain(q, domain):
    assert detect_domain(q) == domain


def test_the_question_reader_picks_the_template_and_says_what_it_guessed():
    t, p, g = read_mechanism_question("Can DS3218 servos lift 300 g at 250 mm reach in a pick and place?")
    assert t == "arm" and p["servo"] == "ds3218" and p["payload"] == "300 g" and p["pick"] == "250,0,40"
    t, p, g = read_mechanism_question("does my wiper crank at 45 rpm sweep far enough?")
    assert t == "fourbar" and p["speed"] == "45 rpm" and any("link lengths" in x for x in g)
    t, p, g = read_mechanism_question("a gearbox with 12:48 and 10:30 driving 0.02 N m")
    assert t == "gears" and p["teeth"] == "12:48,10:30" and p["load"] == "0.02 N*m"
    t, p, g = read_mechanism_question("what peak force when a 200 g part is dropped from 1 m onto steel?")
    assert t == "impact" and p == {"mass": "200 g", "drop": "1 m"}
    with pytest.raises(ValueError, match="which mechanism"):
        read_mechanism_question("how strong is this?")


def test_help_lists_the_command_and_its_examples():
    r = CliRunner().invoke(cli, ["mechanism", "--help"])
    assert r.exit_code == 0 and "wobbl mechanism arm" in r.output and "fourbar" in r.output
    assert "mechanism" in cli.commands and cli.commands["mechanism"].hidden   # reached through wobbl simulate
    r = CliRunner().invoke(cli, ["mechanism", "arm", "--set", "payload"])
    assert r.exit_code != 0 and "key=value" in r.output


def test_the_server_offers_solve_mechanism():
    import asyncio

    from wobbl.mcp.server import build_server

    tools = {t.name: t for t in asyncio.run(build_server().list_tools())}
    assert "solve_mechanism" in tools and "fourbar" in tools["solve_mechanism"].description


@needs_mujoco
def test_the_cli_runs_a_template_and_check_and_report_read_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["mechanism", "pendulum", "--set", "periods=1", "--json", "-o", "out"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["domain"] == "mechanism" and d["verdict"] == "ok" and d["events"]
    r = CliRunner().invoke(cli, ["check", "out"])
    assert r.exit_code == 0 and "hand calculation" in r.output
    r = CliRunner().invoke(cli, ["mechanism", "pendulum", "--set", "colour=red"])
    assert r.exit_code != 0 and "unknown values colour" in r.output


@needs_mujoco
def test_the_mcp_tool_takes_a_spec_as_text_and_its_fixes_point_back_to_it(tmp_path, monkeypatch):
    from wobbl.mcp import tools
    from wobbl.mechanism.tools import solve_mechanism_tool

    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    from test_mechanism_study import WEAK_ARM

    r = solve_mechanism_tool(spec=WEAK_ARM)
    assert "error" not in r, r
    assert r["verdict"] == "fails" and Path(r["trajectory"]).exists() and r["fixes"]
    fix = tools.suggest_fix(r["result_dir"])
    assert fix["domain"] == "mechanism" and "solve_mechanism" in fix["note"]
    q = solve_mechanism_tool(question="does a 1 m pendulum from 30 degrees keep time?")
    assert q["verdict"] == "ok" and q["values"]["period_s"] > 2
    assert "error" in solve_mechanism_tool(template="catapult")
    shutil.rmtree(tmp_path / "results", ignore_errors=True)
