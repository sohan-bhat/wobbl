"""Long jobs: staging decks with checkpoints, reading progress, the live summary of a growing .frd, and with
CalculiX a job that runs in the background, one that is cancelled after its first step and resumed from its
checkpoint to the same answer as an unbroken run, and the same on a remote reached through LocalTransport."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_ccx
from wobbl.jobs import drivers, frdsummary
from wobbl.jobs.watch import eta


def heat_deck(path: Path, n: int = 8, steps: int = 3, period: float = 0.3, max_inc: float = 0.02) -> Path:
    """A hex8 aluminium cube, one face held at 120 C from 20 C, over several transient steps."""
    L = 10.0
    idx, lines = {}, ["*HEADING", "jobs test", "*NODE, NSET=NALL"]
    k = 1
    for kz in range(n + 1):
        for ky in range(n + 1):
            for kx in range(n + 1):
                idx[(kx, ky, kz)] = k
                lines.append(f"{k},{kx * L / n:g},{ky * L / n:g},{kz * L / n:g}")
                k += 1
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    e = 1
    for kz in range(n):
        for ky in range(n):
            for kx in range(n):
                c = [idx[(kx, ky, kz)], idx[(kx + 1, ky, kz)], idx[(kx + 1, ky + 1, kz)], idx[(kx, ky + 1, kz)],
                     idx[(kx, ky, kz + 1)], idx[(kx + 1, ky, kz + 1)], idx[(kx + 1, ky + 1, kz + 1)],
                     idx[(kx, ky + 1, kz + 1)]]
                lines.append(f"{e}," + ",".join(map(str, c)))
                e += 1
    lines.append("*NSET, NSET=HOT")
    lines += [str(idx[(0, ky, kz)]) for ky in range(n + 1) for kz in range(n + 1)]
    lines += ["*MATERIAL, NAME=AL", "*CONDUCTIVITY", "167.", "*SPECIFIC HEAT", "896e6", "*DENSITY", "2.7e-9",
              "*SOLID SECTION, ELSET=EALL, MATERIAL=AL", "*INITIAL CONDITIONS, TYPE=TEMPERATURE", "NALL,20."]
    for s in range(steps):
        lines += ["*STEP, INC=100000", "*HEAT TRANSFER, DELTMX=2.", f"0.01,{period},1e-6,{max_inc}"]
        if s == 0:
            lines += ["*BOUNDARY", "HOT,11,11,120."]
        lines += ["*NODE FILE", "NT", "*END STEP"]
    path.write_text("\n".join(lines) + "\n")
    return path


# ------------------------------------------------------------------------------------------------ no solver
def test_staging_adds_overlaid_restarts_and_resolves_includes(tmp_path):
    (tmp_path / "mesh").mkdir()
    (tmp_path / "mesh" / "nodes.inp").write_text("*NODE\n1,0,0,0\n")
    (tmp_path / "deck.inp").write_text("*INCLUDE, INPUT=mesh/nodes.inp\n*STEP\n*STATIC\n*END STEP\n"
                                       "*STEP\n*STATIC\n*RESTART, WRITE, FREQUENCY=2\n*END STEP\n")
    out = tmp_path / "work"
    out.mkdir()
    drivers.stage_inp(tmp_path / "deck.inp", out / "job.inp", restart=True, copy_includes=False)
    text = (out / "job.inp").read_text()
    assert f"*INCLUDE, INPUT={(tmp_path / 'mesh' / 'nodes.inp').resolve()}" in text
    assert text.count("*RESTART") == 2 and drivers.RESTART_WRITE in text      # the deck's own one is kept
    far = tmp_path / "far"
    far.mkdir()
    files = drivers.stage_inp(tmp_path / "deck.inp", far / "job.inp", restart=False, copy_includes=True)
    assert "*INCLUDE, INPUT=inc1_nodes.inp" in (far / "job.inp").read_text() and len(files) == 2
    assert (far / "inc1_nodes.inp").read_text() == "*NODE\n1,0,0,0\n"
    assert [b.splitlines()[1] for b in drivers.iter_steps(tmp_path / "deck.inp")] == ["*STATIC", "*STATIC"]


def test_progress_from_the_status_file():
    sta = ("SUMMARY OF JOB INFORMATION\n  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME\n"
           "     1          1     1U    1  0.000000E+00  0.000000E+00  0.100000E-01\n"
           "     1          9     1     2  0.300000E+00  0.300000E+00  0.200000E-01\n"
           "     2          3     1     2  0.360000E+00  0.600000E-01  0.200000E-01\n")
    steps = [{"procedure": "heat", "period": 0.3}] * 3
    frac, detail, done = drivers.ccx_progress(sta, steps)
    assert frac == pytest.approx((0.3 + 0.06) / 0.9) and done == 1 and "step 2 of 3" in detail
    assert drivers.ccx_progress("", steps) == (None, "", 0)
    assert drivers.ccx_progress("", steps, 2)[0] == pytest.approx(2 / 3)


def test_eta_hands_over_from_the_model_to_the_pace():
    assert eta(10, None, 100) == 90
    assert eta(10, 0.5, 100) == pytest.approx(10)                 # past a tenth, the pace alone
    assert eta(10, 0.05, 100) == pytest.approx(0.5 * 190 + 0.5 * 90)
    assert eta(10, 0.5, None) == pytest.approx(10)


def test_frd_summary_reads_a_growing_file_once(tmp_path):
    def block(t, step, vals):
        rows = "".join(f" -1{i + 1:10d}{v:12.5E}\n" for i, v in enumerate(vals))
        return (f"  100CL  101{t:12.5E}{len(vals):12d}                     0{step:5d}           1\n"
                f" -4  NDTEMP      1    1\n -5  T           1    1    0    0\n{rows} -3\n")
    head = "    2C                             3                                     1\n" + \
        "".join(f" -1{i + 1:10d}{float(i):12.5E}{0.0:12.5E}{0.0:12.5E}\n" for i in range(3)) + " -3\n"
    f = tmp_path / "r.frd"
    f.write_text(head + block(0.1, 1, [20.0, 30.0, 25.0]) + "  100CL  101 0.2")    # the last block is still coming
    state = {}
    s = frdsummary.summary(str(f), state)
    assert s["blocks"] == 1 and s["peaks"][0]["max"] == 30.0 and s["peaks"][0]["xyz"] == [1.0, 0.0, 0.0]
    with open(f, "a") as fh:
        fh.write("0000E+00           3                     0    2           1\n")
    f.write_text(head + block(0.1, 1, [20.0, 30.0, 25.0]) + block(0.2, 2, [20.0, 31.0, 44.0]) + "9999\n")
    s = frdsummary.summary(str(f), state)
    p = s["peaks"][0]
    assert s["blocks"] == 2 and p["max"] == 44.0 and p["time"] == pytest.approx(0.2) and p["xyz"] == [2.0, 0, 0]
    assert p["min"] == 20.0 and s["history"]["temperature"] == [[0.1, 30.0], [0.2, 44.0]]


def test_a_job_that_will_not_fit_is_refused_before_it_starts(tmp_path, monkeypatch):
    from wobbl import estimate as E
    from wobbl.jobs import manage

    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    deck = heat_deck(tmp_path / "cube.inp", n=4, steps=1)
    monkeypatch.setattr(E, "this_machine", lambda *a, **k: E.Machine("tiny", 1, 1, 1e6, 1e12))
    with pytest.raises(manage.JobError) as e:
        manage.start(str(deck))
    assert "will not run on tiny" in str(e.value) and e.value.estimate["verdict"] == "does not fit"
    assert manage.list_jobs() == []


# ----------------------------------------------------------------------------------------------- CalculiX
def wait(manage, job_id, until=("done", "failed", "cancelled", "interrupted"), timeout=300, check=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = manage.status(job_id)
        if st["state"] in until or (check and check(st)):
            return st
        time.sleep(0.3)
    raise AssertionError(f"{job_id} did not reach {until} in {timeout} s ({st})")


def final_temperatures(path: Path, n_nodes: int) -> np.ndarray:
    from wobbl.solvers.calculix import read_frd

    return read_frd(path, n_nodes)["NDTEMP"][-1][1][:, 0]


@pytest.fixture
def jobs_home(tmp_path, monkeypatch):
    from wobbl.solvers import base

    ccx = base.get("calculix").locate()
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WOBBL_CCX", str(ccx))
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    return tmp_path


@pytest.mark.solver
@needs_ccx
def test_a_background_job_runs_reports_and_calibrates(jobs_home):
    from wobbl import estimate as E
    from wobbl.jobs import manage

    deck = heat_deck(jobs_home / "cube.inp")
    r = manage.start(str(deck), threads=2)
    assert r["state"] in ("starting", "running") and r["predicted"]
    st = wait(manage, r["job_id"])
    assert st["state"] == "done" and st["progress"] == 1.0
    temp = next(p for p in st["peaks"] if p["quantity"] == "temperature")
    assert temp["max"] == pytest.approx(120.0) and 20.0 > temp["min"] - 2.0
    assert "DONE" in Path(st["report"]).read_text() and "Results so far" in Path(st["report"]).read_text()
    assert E.factor("calculix")[1] == 1


@pytest.mark.solver
@needs_ccx
@pytest.mark.skipif(sys.platform == "win32", reason="cancelling signals the watcher")
def test_cancel_after_a_step_and_resume_from_the_checkpoint(jobs_home):
    import os
    import subprocess

    from wobbl.jobs import manage
    from wobbl.solvers import base
    from wobbl.solvers.locate import tool_env

    deck = heat_deck(jobs_home / "cube.inp", n=10, max_inc=0.015)
    ref = jobs_home / "ref"
    ref.mkdir()
    (ref / "cube.inp").write_text(deck.read_text())
    ccx = base.get("calculix").locate()
    subprocess.run([str(ccx), "-i", "cube"], cwd=ref, env=tool_env(ccx, 2), capture_output=True, check=True)
    r = manage.start(str(deck), threads=2)
    job = r["job_id"]
    wait(manage, job, check=lambda st: "step 2 of 3" in st.get("detail", ""))
    st = manage.cancel(job)
    assert st["state"] == "cancelled"
    r2 = manage.resume(job)
    assert r2["resumed_from"] == "after step 1 of 3" and r2["restarts"] == 1
    st = wait(manage, job)
    assert st["state"] == "done", st
    work = jobs_home / "home" / "jobs" / job / "work"
    assert (work / "job_r1.inp").read_text().startswith("*RESTART, READ, STEP=1\n")
    assert not (work / "job.rout").exists() and (work / "job_r1.rout").exists()
    got, want = final_temperatures(work / "job_r1.frd", 1331), final_temperatures(ref / "cube.frd", 1331)
    assert want.max() - want.min() > 15          # still heating up, so the comparison means something
    assert np.abs(got - want).max() < 1e-6
    assert os.path.getsize(work / "job_r1.rout") < 1000 * 1331     # OVERLAY keeps one step, about 720 a node


@pytest.mark.solver
@needs_ccx
@pytest.mark.skipif(sys.platform == "win32", reason="the remote side is a POSIX shell")
def test_a_remote_job_sends_input_and_brings_back_compact_results(jobs_home):
    from wobbl.jobs import manage, remote
    from wobbl.solvers import base

    remote.add("box", f"local:{jobs_home / 'far'}", ccx=str(base.get("calculix").locate()))
    (jobs_home / "mesh").mkdir()
    deck = heat_deck(jobs_home / "mesh" / "cube_mesh.inp", steps=0)
    main = jobs_home / "cube.inp"
    body = heat_deck(jobs_home / "full.inp", steps=2).read_text()
    steps = body[body.index("*MATERIAL"):]
    main.write_text("*HEADING\nremote\n*INCLUDE, INPUT=mesh/cube_mesh_nodes.inp\n" + steps)
    mesh = deck.read_text()
    (jobs_home / "mesh" / "cube_mesh_nodes.inp").write_text(mesh[mesh.index("*NODE"):mesh.index("*MATERIAL")])
    r = manage.start(str(main), on="box")
    assert r["runner"] == "box"
    st = wait(manage, r["job_id"])
    assert st["state"] == "done", st
    res = jobs_home / "home" / "jobs" / r["job_id"] / "results"
    assert (res / "solver.log").exists() and (res / "job.sta").exists() and not (res / "job.frd").exists()
    summary = json.loads((res / "summary_job.json").read_text())
    assert summary["peaks"][0]["quantity"] == "temperature" and st["peaks"][0]["max"] == pytest.approx(120.0)
    far = jobs_home / "far" / r["job_id"]
    assert (far / "work" / "job.frd").exists() and (far / "work" / "inc1_cube_mesh_nodes.inp").exists()
    full = manage.pull(r["job_id"], full=True)
    assert "job.frd" in full["files"]
