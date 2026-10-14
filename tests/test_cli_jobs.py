"""wobbl estimate, wobbl jobs and wobbl remote through the real CLI and the MCP tools, with nothing solved."""
from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.jobs import tools


def deck(path, n=6, steps=1):
    from test_jobs import heat_deck
    return heat_deck(path, n=n, steps=steps)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    return tmp_path


def test_estimate_a_deck_and_a_run_given_by_size(home):
    d = deck(home / "cube.inp")
    r = CliRunner().invoke(cli, ["estimate", str(d), "--json"])
    assert r.exit_code == 0, r.output
    e = json.loads(r.output)
    assert e["solver"] == "calculix" and e["verdict"] == "fits" and e["size"]["nodes"] == 343
    r = CliRunner().invoke(cli, ["estimate", "--solver", "openradioss", "--elements", "1.52e6", "--nodes", "1.49e6",
                                 "--end-time", "0.2", "--threads", "4"])
    assert r.exit_code == 0, r.output
    assert "200,000 cycles" in r.output and "Toyota Yaris" in r.output and "1 microsecond" in r.output
    r = CliRunner().invoke(cli, ["estimate"])
    assert r.exit_code != 0 and "--solver" in r.output


def test_estimate_any_command_as_typed(home, parts):
    r = CliRunner().invoke(cli, ["estimate", "--json", "run", parts["plate_hole"], "-m", "pla", "--quality", "fast",
                                 "--fix", "x=min", "--load", "10 N down @ x=max"])
    assert r.exit_code == 0, r.output
    e = json.loads(r.output)
    assert e["solver"] == "wobbl run" and e["procedure"] == "strength" and e["size"]["levels"][0] > 1000
    assert e["verdict"] == "fits" and e["time_s"] > 0
    r = CliRunner().invoke(cli, ["estimate", "mechanism", "fourbar", "--set", "crank=80"])
    assert r.exit_code == 0 and "bundled mechanism example" in r.output, r.output
    r = CliRunner().invoke(cli, ["estimate", "magnetic", "solenoid"])
    assert r.exit_code == 0 and "UNKNOWN" in r.output and "wobbl jobs start" in r.output, r.output


def test_options_after_the_wrapped_command_stay_with_it():
    from wobbl.cli.jobs import start

    ctx = start.make_context("start", ["--threads", "2", "run", "part.step", "-o", "res", "--json", "--threads", "8"])
    assert ctx.params["threads"] == 2 and not ctx.params["as_json"]
    assert ctx.params["target"] == "run" and ctx.params["args"] == ("part.step", "-o", "res", "--json", "--threads", "8")
    deck = start.make_context("start", ["deck.inp", "--threads", "3"])
    assert deck.params["target"] == "deck.inp" and deck.params["threads"] == 3


def test_jobs_list_and_unknown_job(home):
    r = CliRunner().invoke(cli, ["jobs", "list"])
    assert r.exit_code == 0 and "no jobs yet" in r.output
    r = CliRunner().invoke(cli, ["jobs", "status", "nope-1234"])
    assert r.exit_code != 0 and "no job" in r.output


def test_a_cloud_start_only_shows_the_plan(home, monkeypatch):
    from wobbl.jobs import cloud

    def never(*a, **k):
        raise AssertionError("nothing may be launched without the plan's token")
    monkeypatch.setattr(cloud, "launch", never)
    d = deck(home / "cube.inp")
    r = CliRunner().invoke(cli, ["jobs", "start", str(d), "--cloud", "aws", "--instance", "c7i.4xlarge",
                                 "--key-name", "me", "--security-group", "sg-1"])
    assert r.exit_code == 0, r.output
    assert "nothing was started" in r.output and "--confirm" in r.output and "shuts itself down" in r.output
    out = tools.start_job(str(d), cloud="aws", instance="c7i.4xlarge", key_name="me", security_group="sg-1")
    assert out["started"] is False and out["confirm_token"] and out["plan"]["cost_usd"] >= 0
    wrong = tools.start_job(str(d), cloud="aws", instance="c7i.4xlarge", key_name="me", security_group="sg-1",
                            confirm="yes")
    assert wrong["started"] is False
    assert tools.job_status("") .get("error")


@pytest.mark.skipif(sys.platform == "win32", reason="the remote side is a POSIX shell")
def test_remote_add_list_remove(home):
    r = CliRunner().invoke(cli, ["remote", "add", "box", f"local:{home / 'far'}"])
    assert r.exit_code == 0 and "added box with" in r.output, r.output
    assert "box" in CliRunner().invoke(cli, ["remote", "list"]).output
    e = tools.estimate_cost(path=str(deck(home / "cube.inp")), remote="box")
    assert e["machine"]["name"] == "box"
    assert "removed box" in CliRunner().invoke(cli, ["remote", "remove", "box"]).output


def test_mcp_lists_the_job_tools():
    import asyncio

    from wobbl.mcp.server import build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert {"estimate_cost", "start_job", "job_status", "job_logs", "cancel_job", "resume_job"} <= names
