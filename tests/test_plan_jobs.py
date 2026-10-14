"""wobbl simulate plans through the shared verbs: priced before they run, analysis by analysis, run in the
background exactly as planned, and remembered under each analysis so the next estimate learns from them."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from conftest import needs_ccx
from wobbl import estimate as E
from wobbl.jobs import manage
from wobbl.jobs import workload as WL
from wobbl.simulate import route

M = E.Machine("test box", 6, 4, 16e9, 500e9)
SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    """Numbers on this machine are compared as if it were idle, whatever else is running."""
    monkeypatch.setattr(E, "load", lambda: 0.0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    from wobbl.solvers import base

    ccx = base.get("calculix").locate()        # found before the home moves, so the job's own process finds it too
    if ccx:
        monkeypatch.setenv("WOBBL_CCX", str(ccx))
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    return tmp_path


def test_a_plan_is_priced_analysis_by_analysis(home, parts):
    one = route(parts["plate_hole"], "will it hold 100 N")
    two = route(parts["plate_hole"], "will it hold 100 N and what are its natural frequencies")
    assert one.names == ["stress"] and set(two.names) >= {"stress", "vibration"}
    a = E.estimate_plan(one, {"quality": ["fast"]}, M)
    b = E.estimate_plan(two, {"quality": ["fast"]}, M)
    assert a.verdict == "fits" and a.cost.size["analyses"] == ["part.stress"] and a.cost.size["levels"][0] > 500
    assert any(x.startswith("part.stress: A strength study") for x in a.basis)
    assert b.seconds > a.seconds and any(x.startswith("part.vibration:") for x in b.basis)
    # part.stress shares its history with wobbl run, the command it replaced
    assert WL.analysis_command(one.analyses[0]) == "run"


def test_an_analysis_with_nothing_measured_is_left_out_and_said_so(home, parts):
    from wobbl.simulate import Analysis

    plan = route(parts["plate_hole"], "will it hold 100 N")
    new = Analysis("part", "novel", "Something never run here", entry=lambda job: None)
    plan.analyses.append(new)
    e = E.estimate_plan(plan, {}, M)
    assert e.known and any("part.novel is not counted" in x for x in e.basis)
    plan.analyses = [new]
    assert E.estimate_plan(plan, {}, M).verdict == "unknown"


def test_the_plan_command_line_gives_back_the_same_plan(home, parts):
    plan = route(parts["plate_hole"], "will it hold 100 N")
    opts = {"material": ["pla"], "load": ["100 N down @ x=max"], "fix": ["x=min", "y=min"]}
    argv = WL.plan_argv(plan, opts, out="res", quick=True)
    w = WL.parse(argv)
    assert w.command == "jobs run-plan" and w.inputs == [Path(parts["plate_hole"])] and w.out == "res"
    back, options = WL.plan_of(w)
    assert back.names == plan.names and back.domain == plan.domain and options == opts
    assert E.estimate_command(argv, M).cost.size["analyses"] == ["part.stress"]


def test_a_simulate_run_is_remembered_under_its_analysis(home, tmp_path, monkeypatch):
    out = tmp_path / "r"
    out.mkdir()
    part = tmp_path / "p.step"
    part.write_text("solid")
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/wobbl", "mcp"])       # a run from the MCP server
    WL.record_outcome({"domain": "strength", "t_total": 42.0}, out)
    assert E.load_calibration().get("runs", []) == []                       # kept until request.json is there
    (out / "request.json").write_text(json.dumps({"domain": "part", "analysis": "stress",
                                                  "spec": {"thing": str(part), "options": {"quality": ["fast"]}}}))
    WL._flush()
    run = E.load_calibration()["runs"][-1]
    assert run["solver"] == "wobbl run" and run["measured_time"] == 42.0 and run["size"]["signature"]
    WL.record_outcome({"domain": "thermal", "t_total": 9.0}, tmp_path / "none")     # no request.json at all
    WL._flush()
    assert E.load_calibration()["runs"][-1]["solver"] == "wobbl thermal"


@pytest.mark.solver
@needs_ccx
def test_a_plan_runs_in_the_background_exactly_as_planned(home, parts):
    plan = route(parts["plate_hole"], "will a 100 N pull hold")
    opts = {"material": ["6061-t6"], "load": ["100 N +x @ x=max"], "fix": ["x=min"], "quality": ["fast"]}
    r = manage.start_plan(plan, opts, quick=True)
    t0 = time.time()
    while time.time() - t0 < 300:
        st = manage.status(r["job_id"])
        if st["state"] not in ("starting", "running"):
            break
        time.sleep(1)
    assert st["state"] == "done", manage.logs(r["job_id"])
    assert st["results"]["verdict"] and "stress" in manage.logs(r["job_id"], 200)
    job = manage.find(r["job_id"])
    res = Path(job.args[job.args.index("-o") + 1])
    assert res == job.folder / "result"                 # not a folder named result wherever it was started
    req = json.loads((res / "request.json").read_text())
    assert req["analysis"] == "stress" and req["spec"]["options"]["load"] == ["100 N +x @ x=max"]
    runs = E.load_calibration()["runs"]
    assert [x["solver"] for x in runs] == ["wobbl run"] and runs[0]["predicted_time"] > 0


@pytest.mark.skipif(sys.platform == "win32", reason="the remote side is a POSIX shell")
def test_a_plan_goes_to_a_remote_with_its_part(home, parts):
    from wobbl.jobs import remote

    fake = home / "fakewobbl.sh"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > args.txt\n')
    fake.chmod(0o755)
    remote.add("box", f"local:{home / 'far'}", wobbl=str(fake))
    plan = route(parts["plate_hole"], "will it hold 100 N")
    r = manage.start_plan(plan, {"load": ["100 N +x @ x=max"]}, on="box")
    far = home / "far" / r["job_id"]
    for _ in range(100):
        if (far / "exit_code").exists():
            break
        time.sleep(0.1)
    args = (far / "args.txt").read_text().splitlines()
    name = Path(parts["plate_hole"]).name
    assert args[:3] == ["jobs", "run-plan", f"inputs/1_{name}"] and (far / "inputs" / f"1_{name}").exists()
    assert args[args.index("--physics") + 1] == "part.stress" and args[-2:] == ["-o", "result"]
    assert args[args.index("--opt") + 1:args.index("--opt") + 3] == ["load", "100 N +x @ x=max"]
