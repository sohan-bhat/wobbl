"""wobbl simulate end to end: the command, the MCP tool, the rerun contract and a domain plugged in from outside."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.simulate import Analysis, Domain, Option, get, registry, rerun, route

ROOT = Path(__file__).resolve().parents[1]
PART = str(ROOT / "examples" / "shelf-bracket" / "bracket.py")
ROCK = "a 9 cm rock dropped from 300 m"


def test_plan_shows_route_and_overrides_without_running():
    r = CliRunner().invoke(cli, ["simulate", PART, "will it hold 200 N and stay under 60 C at 5 W?", "--plan"])
    assert r.exit_code == 0, r.output
    assert "override with --physics stress,heat" in r.output and "override with --domain part" in r.output
    r = CliRunner().invoke(cli, ["simulate", PART, "--physics", "heat", "--plan", "--json"])
    assert json.loads(r.output)["analyses"] == ["heat"]


def test_unknown_options_and_physics_are_caught():
    r = CliRunner().invoke(cli, ["simulate", PART, "--physics", "heat", "--rpm", "6000", "--plan"])
    assert r.exit_code != 0 and "no analysis in this plan reads --rpm" in r.output
    r = CliRunner().invoke(cli, ["simulate", PART, "--physics", "teleport", "--plan"])
    assert r.exit_code != 0 and "no physics 'teleport'" in r.output
    r = CliRunner().invoke(cli, ["simulate", "missing.step", "will it hold?"])
    assert r.exit_code != 0 and "file not found" in r.output


def test_list_names_every_domain():
    r = CliRunner().invoke(cli, ["simulate", "--list"])
    assert r.exit_code == 0
    for word in ("part", "circuit", "mechanism", "flight", "fluid", "field", "chemistry", "stress", "trajectory"):
        assert word in r.output


def test_a_flight_runs_end_to_end_and_reruns_from_its_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["simulate", ROCK, "how fast does it land?", "--json", "--no-open", "-o", "rock"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["plan"]["domain"] == "flight" and d["steps"][0]["verdict"] == "ok"
    assert "--height '300 m'" in d["explicit"] and "--diameter '9 cm'" in d["explicit"]
    req = json.loads((tmp_path / "rock" / "request.json").read_text())
    assert req["domain"] == "flight" and req["analysis"] == "trajectory"
    assert req["spec"]["options"]["height"] == ["300 m"]
    sim = json.loads((tmp_path / "rock" / "simulation.json").read_text())
    assert sim["situation"]["objects"] == ["ground"] and sim["steps"][0]["analysis"] == "trajectory"
    again = get("flight", "trajectory").run(req["spec"], tmp_path / "again", quick=True)
    assert again["headline"] == json.loads((tmp_path / "rock" / "outcome.json").read_text())["headline"]
    higher = rerun(tmp_path / "rock", tmp_path / "higher", {"height": "600 m"}, quick=True)
    assert higher["values"]["flight_time_s"] > again["values"]["flight_time_s"]


def test_the_bare_form_goes_through_simulate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, [ROCK, "--json", "--no-open"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["plan"]["domain"] == "flight"


def test_mcp_simulate(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    from wobbl.mcp.simulate import physics_tool, simulate_tool

    r = simulate_tool(PART, "will it hold 200 N?", plan_only=True)
    assert r["plan"]["analyses"] == ["stress"] and "material" in r["options_read"]
    r = simulate_tool("", ROCK, options={"cd": 0.6})
    assert r["verdict"] == "pass" and r["explicit"]["options"]["cd"] == "0.6" and "view" in r
    assert r["steps"][0]["physics"] == "trajectory" and r["steps"][0]["confidence"]
    assert "error" in simulate_tool(PART, "", options={"warp": 9})
    names = {d["name"] for d in physics_tool()["domains"]}
    assert {"part", "flight", "chemistry"} <= names


def test_mcp_lists_simulate_first_and_marks_the_scenario_tools():
    import asyncio

    from wobbl.mcp.server import build_server

    tools = asyncio.run(build_server().list_tools())
    assert [t.name for t in tools][:2] == ["simulate", "physics"]
    by = {t.name: t.description for t in tools}
    assert by["solve_thermal"].startswith("Deprecated alias of simulate")
    assert not by["suggest_fix"].startswith("Deprecated")


@pytest.fixture
def optics():
    """A domain and an analysis registered from outside, the way any package's domain.py does it."""
    saved = (dict(registry._DOMAINS), dict(registry._ANALYSES))

    def focus(job):
        return {"verdict": "ok", "headline": f"focus at {job.value('distance', '50 mm', 'distance not given')}",
                "confidence": "high", "workdir": str(job.out)}

    registry.register(Domain("optics", "A lens", extensions={".lens": "lens"}, words=(r"\blens\b",),
                             kinds={"lens": r"\blens\b"}))
    registry.register(Analysis("optics", "focus", "Where a lens focuses", focus, keywords=(r"\bfocus",),
                               default=True, options=(Option("distance", "Object distance", "50 mm"),)))
    yield
    registry._DOMAINS.clear()
    registry._DOMAINS.update(saved[0])
    registry._ANALYSES.clear()
    registry._ANALYSES.update(saved[1])


def test_new_physics_plugs_in_without_a_command(optics, tmp_path, monkeypatch):
    plan = route("", "where does a 25 mm lens focus?")
    assert plan.domain == "optics" and plan.names == ["focus"]
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["simulate", "", "where does a 25 mm lens focus?", "--distance", "80 mm", "--json",
                                 "-o", "lens"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["steps"][0]["headline"] == "focus at 80 mm" and "--distance '80 mm'" in d["explicit"]
    assert json.loads((tmp_path / "lens" / "request.json").read_text())["spec"]["options"] == {"distance": ["80 mm"]}
