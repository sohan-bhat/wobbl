"""The watcher: one detached process per local job that runs the solver, follows it and keeps the status and the
live report current.

'wobbl jobs start' launches it in a session of its own, so closing the terminal does not stop the job. On a Mac
it also holds an idle sleep assertion (caffeinate) for as long as the solver runs. Closing the lid still sleeps
the machine, which pauses the job, and it carries on when the machine wakes. A restart or a crash ends it, and
'wobbl jobs resume' picks it up again from its last checkpoint.

    python -m wobbl.jobs.watch JOB_ID
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .store import Job, write_json

INTERVAL = 2.0          # seconds between looks at the solver
REPORT_EVERY = 10.0     # seconds between report refreshes
_cancelled = False
_proc: subprocess.Popen | None = None


def eta(elapsed: float, frac: float | None, predicted: float | None) -> float | None:
    """Seconds left. The cost model's prediction at first, handing over to the solver's own pace as progress
    builds, fully once a tenth of the run is done."""
    model = max(0.0, predicted - elapsed) if predicted else None
    if frac is None or frac <= 0.005:
        return model
    pace = elapsed * (1.0 - frac) / frac
    if model is None:
        return pace
    w = min(1.0, frac / 0.1)
    return w * pace + (1.0 - w) * model


def keep_awake(pid: int):
    """Hold off idle sleep on a Mac while the solver runs."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        try:
            return subprocess.Popen(["caffeinate", "-i", "-w", str(pid)], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError:
            return None
    return None


def poll(p: subprocess.Popen):
    """(exit code, resource usage) once the solver has finished, else (None, None)."""
    if sys.platform == "win32":
        code = p.poll()
        return code, None
    pid, status, ru = os.wait4(p.pid, os.WNOHANG)
    if pid == 0:
        return None, None
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, ru


def _on_signal(signum, frame):
    """Stop the solver at once, so a cancel never waits for the next look at it."""
    global _cancelled
    _cancelled = True
    if _proc is not None and _proc.returncode is None:
        try:
            os.killpg(_proc.pid, signal.SIGTERM) if sys.platform != "win32" else _proc.terminate()
        except OSError:
            pass


def _stop(p: subprocess.Popen):
    try:
        if sys.platform == "win32":
            p.terminate()
        else:
            os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(p.pid, signal.SIGKILL) if sys.platform != "win32" else p.kill()
        except Exception:  # noqa: BLE001
            pass


def solver_env(argv0: str, threads: int) -> dict:
    from ..solvers.locate import tool_env

    env = tool_env(Path(argv0), threads) if Path(argv0).is_file() else dict(os.environ)
    for k in ("OMP_NUM_THREADS", "CCX_NPROC_EQUATION_SOLVER", "CCX_NPROC_RESULTS", "CCX_NPROC_STIFFNESS",
              "OPENBLAS_NUM_THREADS"):
        env[k] = str(threads)
    return env


def run(job_id: str) -> int:
    from .manage import progress_of, results_of, finish
    from .report import write_report

    job = Job.load(job_id)
    launch = json.loads((job.folder / "launch.json").read_text())
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    t0 = time.time()
    st = {"state": "running", "progress": 0.0, "eta_s": None, "elapsed_s": 0.0, "detail": "starting",
          "watcher_pid": os.getpid(), "started": t0, "updated": t0, "run": job.restarts}
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else \
        {"start_new_session": True}
    with open(job.folder / "solver.log", "ab") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} run {job.restarts}: {' '.join(launch['argv'])}\n"
                  .encode())
        log.flush()
        p = subprocess.Popen(launch["argv"], cwd=launch["cwd"], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=solver_env(launch["argv"][0], job.threads), **kw)
    global _proc
    _proc = p
    st["solver_pid"] = p.pid
    awake = keep_awake(p.pid)
    predicted = (job.estimate or {}).get("time_s")
    cache: dict = {}
    last_report = 0.0
    code, ru = None, None
    while True:
        code, ru = poll(p)
        if _cancelled and code is None:
            _stop(p)
            code, ru = (p.returncode if p.returncode is not None else -15), None
        now = time.time()
        frac, detail = progress_of(job, now - t0)
        st.update(progress=frac, detail=detail or st["detail"], elapsed_s=round(now - t0, 1), updated=now,
                  eta_s=None if code is not None else eta(now - t0, frac, predicted))
        if now - last_report >= REPORT_EVERY or code is not None:
            st["results"] = results_of(job, cache)
            write_report(job, st)
            last_report = now
        write_json(job.folder / "status.json", st)
        if code is not None:
            break
        time.sleep(INTERVAL)
    if awake is not None:
        awake.terminate()
    finish(job, st, code, ru, cancelled=_cancelled)
    write_json(job.folder / "status.json", st)
    write_report(job, st)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
