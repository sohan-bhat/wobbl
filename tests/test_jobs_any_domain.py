"""Jobs of any domain: a Wobbl command runs in the background here or on a remote, from any folder, is predicted
before it starts and refused when it will not fit, and its finished run teaches the next estimate."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from wobbl import estimate as E
from wobbl.jobs import manage, remote

posix = pytest.mark.skipif(sys.platform == "win32", reason="the remote side is a POSIX shell")
ROCKET = Path(__file__).resolve().parents[1] / "examples" / "rocket-launch" / "rocket.toml"
SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    """Numbers on this machine are compared as if it were idle, whatever else is running."""
    monkeypatch.setattr(E, "load", lambda: 0.0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    monkeypatch.chdir(tmp_path)
    shutil.copy2(ROCKET, tmp_path / "rocket.toml")
    return tmp_path


def wait(job_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = manage.status(job_id)
        if st["state"] not in ("starting", "running"):
            return st
        time.sleep(0.5)
    raise AssertionError(f"{job_id} still running after {timeout} s")


def test_a_command_the_cli_cannot_read_never_becomes_a_job(home):
    from wobbl.jobs.workload import WorkloadError

    with pytest.raises(WorkloadError):
        manage.start("", ["flow"])
    assert manage.list_jobs() == []


def test_a_command_that_will_not_fit_is_refused(home, monkeypatch):
    E.record("wobbl motion", 0.0, 30.0, 0.0, 9e9, {"signature": "x", "disk": 1e6}, "run")
    monkeypatch.setattr(E, "this_machine", lambda *a, **k: E.Machine("tiny", 2, 1, 2e9, 1e12))
    with pytest.raises(manage.JobError, match="wobbl motion will not run on tiny") as e:
        manage.start("", ["motion", "rocket.toml"])
    assert e.value.estimate["verdict"] == "does not fit" and manage.list_jobs() == []


def test_a_motion_job_runs_from_any_folder_and_teaches_the_next_estimate(home, monkeypatch):
    r = manage.start("", ["motion", "rocket.toml", "-o", "flight"])
    job = manage.find(r["job_id"])
    assert job.args[1] == str((home / "rocket.toml").resolve()) and job.args[-2:] == ["-o", str(home / "flight")]
    monkeypatch.chdir(home.parent)                     # following it from elsewhere changes nothing
    st = wait(r["job_id"])
    assert st["state"] == "done", manage.logs(r["job_id"])
    assert st["results"]["verdict"] and st["results"]["headline"]
    assert (home / "flight" / "outcome.json").exists()
    runs = E.load_calibration()["runs"]
    assert [x["solver"] for x in runs] == ["wobbl motion"] and runs[0]["size"]["domain"] == "motion"
    monkeypatch.chdir(home)
    again = E.estimate_command(["motion", "rocket.toml"], E.Machine("t", 6, 4, 16e9, 1e12))
    assert "very motion study" in again.basis[0] and again.seconds == pytest.approx(runs[0]["measured_time"])


@posix
def test_a_command_runs_on_a_remote_with_its_inputs_and_comes_back(home):
    remote.add("box", f"local:{home / 'far'}", wobbl=f"{sys.executable} -m wobbl")
    r = manage.start("", ["motion", "rocket.toml"], on="box")
    assert r["runner"] == "box" and r["input"] == "wobbl motion rocket.toml"
    for _ in range(480):            # listing the jobs is enough to learn that a remote one finished
        if manage.list_jobs()[0]["state"] not in ("starting", "running"):
            break
        time.sleep(0.5)
    assert manage.list_jobs()[0]["state"] == "done"
    st = wait(r["job_id"])
    assert st["state"] == "done", manage.logs(r["job_id"])
    far = home / "far" / r["job_id"]
    assert (far / "inputs" / "1_rocket.toml").exists() and (far / "result" / "outcome.json").exists()
    back = home / "home" / "jobs" / r["job_id"] / "results"
    assert json.loads((back / "result" / "outcome.json").read_text())["domain"] == "motion"
    assert st["results"]["headline"] and (back / "solver.log").exists()


@posix
def test_cancelling_a_remote_command_stops_its_whole_process_group(home):
    fake = home / "fakewobbl.sh"
    fake.write_text("#!/bin/sh\nsleep 120 &\nsleep 120\n")
    fake.chmod(0o755)
    remote.add("box", f"local:{home / 'far'}", wobbl=str(fake))
    t0 = time.time()
    r = manage.start("", ["motion", "rocket.toml"], on="box")
    assert time.time() - t0 < 20                     # the launch returns at once, the run goes on there
    far = home / "far" / r["job_id"]
    for _ in range(50):
        if (far / "pgid").exists() and (far / "pgid").read_text().strip():
            break
        time.sleep(0.1)
    assert manage.status(r["job_id"])["state"] == "running"
    pgid = int((far / "pgid").read_text())
    assert manage.cancel(r["job_id"])["state"] == "cancelled"
    import os
    for _ in range(50):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the remote run's processes are still alive")


def test_a_job_stops_before_the_disk_fills(home, monkeypatch):
    monkeypatch.setenv("WOBBL_DISK_RESERVE_GB", "1e9")          # more than any disk, so the watcher stops it at once
    r = manage.start("", ["motion", "rocket.toml"], force=True)
    st = wait(r["job_id"])
    assert st["state"] == "failed" and "left on disk, to keep" in st["detail"] and "wobbl jobs resume" in st["detail"]


@posix
def test_a_remote_run_stops_before_its_disk_fills(home, monkeypatch):
    fake = home / "fakewobbl.sh"
    fake.write_text("#!/bin/sh\nsleep 60\n")
    fake.chmod(0o755)
    remote.add("box", f"local:{home / 'far'}", wobbl=str(fake))
    monkeypatch.setattr(E, "DISK_RESERVE", 1e18)
    r = manage.start("", ["motion", "rocket.toml"], on="box", force=True)
    st = wait(r["job_id"], timeout=60)
    assert st["state"] == "failed" and "Stopped on box with" in st["detail"]


def test_a_remote_without_wobbl_is_refused_for_commands(home):
    cfg = remote.load()
    cfg["bare"] = {"host": f"local:{home / 'far'}", "root": "wobbl-jobs", "cores": 2, "ram": 4e9}
    remote.save(cfg)
    with pytest.raises(manage.JobError, match="Wobbl is not installed on bare"):
        manage.start("", ["motion", "rocket.toml"], on="bare")
