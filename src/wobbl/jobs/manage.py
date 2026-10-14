"""Start, follow, stop and resume long jobs, here or on a machine the user owns.

Every start predicts the run first and refuses one that will not fit in memory or on disk, with the options
the estimate offers, rather than letting it fail halfway. A local job runs under its own watcher process
(``wobbl.jobs.watch``). A remote job runs under a small shell script that needs nothing but the solver on the
other side, and its status is read back whenever it is asked for, so nothing has to stay connected.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import remote as R
from .drivers import CalculixDriver, ccx_progress, stage_inp, sta_rows
from .store import ACTIVE, Job, all_jobs, latest, new_id, write_json

COMPACT = (".sta", ".dat", ".cvg")      # small result files a remote job sends back, as the .frd stays unless asked
CLOUD_CCX = "/home/ubuntu/.wobbl/env/bin/ccx"    # where the cloud machine's start up script installs CalculiX
# Only the run goes to the background, in a subshell of its own that returns at once. Backgrounding a whole
# "cd && nohup ... &" list would keep the connection open until the run ended.
LAUNCH = "(nohup sh run.sh > /dev/null 2>&1 < /dev/null &)"
# a run lives as long as its script, which writes the exit code last, and one whose script has not written its
# pid yet is starting, for STARTING seconds after the launch
ALIVE = ("if [ ! -f watcher.pid ]; then echo starting; elif kill -0 $(cat watcher.pid) 2>/dev/null; then echo alive; "
         "else echo gone; fi")
STARTING = 120.0


class JobError(RuntimeError):
    def __init__(self, message: str, hint: str = "", estimate: dict | None = None):
        super().__init__(message)
        self.hint = hint
        self.estimate = estimate


def kind_of(target: str, args: list[str]) -> str:
    if args:
        return "wobbl"
    p = Path(target)
    if p.is_dir():
        return "openfoam"
    ext = p.suffix.lower()
    if ext == ".inp":
        return "calculix"
    if ext in (".k", ".key", ".dyn", ".rad"):
        return "openradioss"
    raise JobError(f"Wobbl cannot run '{target}' as a job", "Give a CalculiX .inp deck, or a Wobbl command such "
                   "as wobbl jobs start run part.step -m pla --fix 'x=min' --load '50 N down @ top'")


def find(job_id: str = "") -> Job:
    if job_id:
        return Job.load(job_id)
    j = latest()
    if j is None:
        raise JobError("there are no jobs yet", "Start one with wobbl jobs start deck.inp")
    return j


# --------------------------------------------------------------------------------------------------- start
def start(target: str = "", args: list[str] | None = None, on: str = "", threads: int = 0, checkpoint: bool = True,
          force: bool = False, max_hours: float | None = None, cloud: dict | None = None) -> dict:
    """Predict, then start a job. ``cloud`` is a launched machine from ``jobs.cloud`` to run on."""
    from ..estimate import Machine, default_threads, estimate_command, estimate_deck, this_machine
    from . import workload as WL

    args = list(args or [])
    kind = kind_of(target, args)
    if kind in ("openfoam", "openradioss"):
        raise JobError(f"{'OpenFOAM' if kind == 'openfoam' else 'OpenRadioss'} jobs are not available yet",
                       f"wobbl estimate {target} predicts its run time, memory and disk")
    entry = R.get(on) if on else None
    if cloud and kind == "wobbl":
        raise JobError("a rented cloud machine runs CalculiX decks, not Wobbl commands",
                       "Run the command on a machine of yours with --on, or start its deck on the cloud")
    if entry and kind == "wobbl" and not entry.get("wobbl"):
        raise JobError(f"Wobbl is not installed on {on}, which a Wobbl command needs there",
                       f"Install it there, or point at it with wobbl remote add {on} {entry['host']} --wobbl PATH")
    if cloud:
        entry = {"host": cloud["host"], "root": "wobbl-jobs", "ccx": cloud.get("ccx", CLOUD_CCX), "cloud": True,
                 "cores": cloud.get("cores"), "ram": cloud.get("ram")}
    job = Job(new_id(target if kind != "wobbl" else (args[1] if len(args) > 1 else "study")), kind,
              str(Path(target).resolve()) if kind != "wobbl" else " ".join(args), on or ("cloud" if cloud else "local"),
              checkpoint=checkpoint and kind == "calculix", args=args, cloud=cloud or {})
    if entry:
        cores = int(entry.get("cores") or 1)
        machine = Machine(on or "the cloud machine", cores, threads or cores, float(entry.get("ram") or 8e9),
                          float(entry.get("free") or float("inf")))
    else:
        machine = this_machine(job.folder.parent, threads or min(default_threads(), physical()))
    if kind == "calculix":
        est = estimate_deck(target, machine, max_hours, checkpoint=job.checkpoint)
        name = Path(target).name
    else:
        w = WL.parse(args)                      # a command the CLI cannot read stops here, before anything is written
        est = estimate_command(args, machine if entry else None, max_hours)
        job.args = WL.absolute(w, str(job.folder / "result"))
        name = f"wobbl {w.command}"
    if est.known and not est.fits and not force:
        raise JobError(f"{name} will not run on {machine.name}. " + " ".join(est.problems),
                       "\n".join(o["summary"] for o in est.options), est.to_dict())
    if kind == "calculix":
        from ..decks import read_inp

        job.steps = [{"procedure": s.procedure, "period": s.period} for s in read_inp(target).steps]
    job.threads = machine.threads
    if est.known:
        job.estimate = {**est.to_dict(), "core_seconds": est.cost.core_seconds,
                        "parallel": est.cost.parallel, "peak_bytes": est.cost.peak_bytes}
    job.save()
    if entry:
        _launch_remote(job, entry)
    else:
        _launch_local(job)
    return describe(job)


def physical() -> int:
    from ..estimate import physical_cores
    return physical_cores()


def _launch_local(job: Job, resume_deck: str = ""):
    job.work.mkdir(parents=True, exist_ok=True)
    if job.kind == "calculix":
        from ..solvers import base as solvers

        ccx = str(solvers.get("calculix").require())
        drv = CalculixDriver(job)
        if resume_deck:
            (job.work / f"{drv.name}.inp").write_text(resume_deck)
        elif not (job.work / "job.inp").exists():
            drv.stage(job.work)
        argv, cwd = drv.command(ccx), str(job.work)
    else:
        from .drivers import wobbl_command

        argv, cwd = wobbl_command(job), str(job.folder)
    write_json(job.folder / "launch.json", {"argv": argv, "cwd": cwd})
    write_json(job.folder / "status.json", {"state": "starting", "progress": 0.0, "updated": time.time(),
                                            "detail": "starting", "run": job.restarts})
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])        # the folder holding the wobbl package
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    kw = {"creationflags": 0x00000008 | 0x00000200} if sys.platform == "win32" else {"start_new_session": True}
    with open(job.folder / "watcher.log", "ab") as err:
        p = subprocess.Popen([sys.executable, "-m", "wobbl.jobs.watch", job.id], cwd=job.folder, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, close_fds=True, **kw)
    st = json.loads((job.folder / "status.json").read_text())
    st["watcher_pid"] = p.pid
    write_json(job.folder / "status.json", st)


# ------------------------------------------------------------------------------------------- local follow
def _reader(job: Job):
    def read(name: str, tail: int = 0) -> str:
        p = job.work / name
        try:
            with open(p, "rb") as fh:
                if tail:
                    fh.seek(max(0, p.stat().st_size - tail))
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
    return read


def progress_of(job: Job, elapsed: float) -> tuple[float | None, str]:
    """Fraction done and a detail line for a local job. A single increment step reports nothing until it ends,
    so in the meantime the fraction comes from the time the cost model predicted."""
    predicted = (job.estimate or {}).get("time_s")
    if job.kind == "calculix":
        frac, detail = CalculixDriver(job).progress(_reader(job))
        if frac is None and predicted:
            return min(0.95, elapsed / predicted), "solving (progress estimated from the cost model)"
        return frac, detail
    from .drivers import wobbl_progress

    log = (job.folder / "solver.log").read_text(errors="replace")[-20000:] if (job.folder / "solver.log").exists() \
        else ""
    frac, detail = wobbl_progress(log)
    if frac is None and predicted:
        frac = min(0.95, elapsed / predicted)
    return frac, detail


def results_of(job: Job, cache: dict) -> dict:
    """The peaks so far, read from the solver's results as they arrive."""
    if job.kind != "calculix":
        return {}
    from .frdsummary import summary

    out = {}
    for r in range(job.restarts + 1):
        name = "job" if r == 0 else f"job_r{r}"
        f = job.work / f"{name}.frd"
        if f.exists():
            out[name] = summary(str(f), cache.setdefault(name, {}))
    return merge(out)


def merge(parts: dict) -> dict:
    """One summary from the summaries of a job's runs, in run order."""
    peaks, history, blocks = {}, {}, 0
    for s in parts.values():
        blocks += s.get("blocks", 0)
        for p in s.get("peaks", []):
            q = p["quantity"]
            if q not in peaks or (p.get("max") or 0) > (peaks[q].get("max") or 0):
                peaks[q] = p
        for q, h in s.get("history", {}).items():
            history.setdefault(q, []).extend(h)
    return {"blocks": blocks, "peaks": list(peaks.values()), "history": history} if parts else {}


def finish(job: Job, st: dict, code, ru, cancelled: bool = False):
    """Settle the final state, keep the measured run for calibration and say what happened."""
    from ..estimate import amdahl, duration, record

    log = (job.folder / "solver.log").read_text(errors="replace")[-6000:] if (job.folder / "solver.log").exists() \
        else ""
    errors = [ln.strip() for ln in log.splitlines() if "*ERROR" in ln]
    st["exit_code"] = code
    st["finished"] = time.time()
    wall = st.get("elapsed_s") or 0.0
    if ru is not None:
        st["cpu_s"] = round(ru.ru_utime + ru.ru_stime, 1)
        st["peak_memory_gb"] = round(ru.ru_maxrss * (1.0 if sys.platform == "darwin" else 1024.0) / 1e9, 3)
    if cancelled:
        st["state"], st["detail"] = "cancelled", f"Cancelled. Resume it with wobbl jobs resume {job.id}"
    elif code == 0 and not errors:
        st["state"], st["progress"], st["eta_s"] = "done", 1.0, 0
        st["detail"] = f"Finished in {duration(wall)}."
        est = job.estimate or {}
        if job.kind == "calculix" and job.restarts == 0 and est.get("core_seconds") and wall > 0:
            core = wall * amdahl(job.threads, est.get("parallel", 0.6))
            record("calculix", est["core_seconds"], core, est.get("peak_bytes", 0.0),
                   st.get("peak_memory_gb", 0.0) * 1e9, est.get("size", {}), "job")
    else:
        st["state"] = "failed"
        st["detail"] = (" ".join(errors[:2]) if errors else f"The solver stopped with exit code {code}.")[:400] + \
            (f" Resume it with wobbl jobs resume {job.id}" if job.checkpoint else "")
    if job.kind == "wobbl":
        from .drivers import result_folder

        out = result_folder(job) / "outcome.json"
        if out.exists():
            d = json.loads(out.read_text())
            st["results"] = {"verdict": d.get("verdict"), "headline": d.get("headline"), "report": d.get("report")}


# ------------------------------------------------------------------------------------------------- remote
RUN_SH = """#!/bin/sh
# Wobbl job {id}, run {run}. Needs only a POSIX shell and CalculiX.
cd "$(dirname "$0")/work" || exit 1
export OMP_NUM_THREADS={t} CCX_NPROC_EQUATION_SOLVER={t} CCX_NPROC_RESULTS={t} CCX_NPROC_STIFFNESS={t}
{wait}rm -f ../exit_code
ps -o pgid= -p $$ | tr -d ' ' > ../pgid
echo $$ > ../watcher.pid
date +%s > ../started
{ccx} -i {name} >> ../solver.log 2>&1 &
echo $! > ../solver.pid
wait $!
code=$?
if command -v python3 > /dev/null 2>&1 && [ -f {name}.frd ]; then
  python3 ../frdsummary.py {name}.frd > ../summary_{name}.json 2> /dev/null
fi
echo $code > ../exit_code
"""


RUN_WOBBL_SH = """#!/bin/sh
# Wobbl job {id}, run {run}. Needs only a POSIX shell and Wobbl.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS={t}
rm -f exit_code
ps -o pgid= -p $$ | tr -d ' ' > pgid
echo $$ > watcher.pid
date +%s > started
{wobbl} {args} >> solver.log 2>&1 < /dev/null &
echo $! > solver.pid
wait $!
echo $? > exit_code
"""


def _launch_remote_wobbl(job: Job, entry: dict):
    """Send a Wobbl command's input files to a remote and start it there under nohup."""
    from . import workload as WL

    t = R.transport(entry)
    job.remote_dir = job.remote_dir or job.id
    w = WL.parse(job.args)
    stage = job.folder / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "inputs").mkdir(parents=True)
    mapping = {}
    for k, (tok, p) in enumerate(zip(w.tokens, w.inputs)):
        name = f"{k + 1}_{p.name}"
        if p.is_dir():
            shutil.copytree(p, stage / "inputs" / name, ignore=shutil.ignore_patterns("*.frd", "*.rout", "stage"))
        else:
            shutil.copy2(p, stage / "inputs" / name)
        mapping[tok] = f"inputs/{name}"
    args = WL.rewrite(job.args, mapping, out="result")
    (stage / "run.sh").write_text(RUN_WOBBL_SH.format(id=job.id, run=job.restarts, t=job.threads,
                                                      wobbl=entry["wobbl"], args=shlex.join(args)))
    t.put(stage, job.remote_dir)
    shutil.rmtree(stage, ignore_errors=True)
    d = shlex.quote(job.remote_dir)
    t.check(f"cd {d} && rm -rf result exit_code solver.pid watcher.pid started && {LAUNCH}")
    job.pulled = False
    job.save()
    write_json(job.folder / "status.json", {"state": "running", "progress": 0.0, "updated": time.time(),
                                            "started": time.time(), "launched": time.time(),
                                            "detail": f"started on {t.name}", "run": job.restarts})


def _remote_status_wobbl(job: Job, st: dict, t: R.Transport) -> dict:
    from ..estimate import duration
    from .drivers import wobbl_progress
    from .watch import eta

    d = shlex.quote(job.remote_dir)
    cmd = (f"cd {d} && cat exit_code 2>/dev/null; echo '@@'; tail -c 6000 solver.log 2>/dev/null; echo '@@'; "
           f"{ALIVE}; echo '@@'; cat started 2>/dev/null")
    try:
        out = t.run(cmd, timeout=60).stdout.split("@@")
    except R.RemoteError as e:
        st["reachable"] = False
        st["detail"] = f"{job.runner} cannot be reached right now ({e}). The job keeps running there."
        return st
    code_text, log, alive_text = out[0].strip(), out[1], out[2].strip()
    started = float(out[-1].strip() or st.get("started") or time.time())
    elapsed = time.time() - started
    predicted = (job.estimate or {}).get("time_s")
    frac, detail = wobbl_progress(log)
    if frac is None and predicted:
        frac = min(0.95, elapsed / predicted)
    st.update(reachable=True, elapsed_s=round(elapsed, 1), updated=time.time(), run=job.restarts)
    if code_text:
        code = int(code_text) if code_text.lstrip("-").isdigit() else 1
        st["exit_code"] = code
        st["state"] = "done" if code == 0 else "failed"
        st["progress"] = 1.0 if code == 0 else frac
        st["detail"] = (f"Finished in {duration(elapsed)}." if code == 0 else
                        f"Wobbl stopped with exit code {code}. " + (detail or ""))
        pull(job.id, st=st)
    elif alive_text == "alive" or (alive_text == "starting" and time.time() - st.get("launched", 0) < STARTING):
        st.update(state="running", progress=frac, detail=detail or "running", eta_s=eta(elapsed, frac, predicted))
    elif st.get("state") != "cancelled":
        st["state"] = "interrupted"
        st["detail"] = f"The run stopped on {job.runner} without finishing. Run it again with wobbl jobs resume {job.id}"
    write_json(job.folder / "status.json", st)
    return st


def _transport(job: Job) -> R.Transport:
    if job.cloud:
        return R.transport({"host": job.cloud["host"], "root": "wobbl-jobs"})
    return R.transport(R.get(job.runner))


def _launch_remote(job: Job, entry: dict, resume_deck: str = "", rin_from: str = ""):
    if job.kind == "wobbl":
        return _launch_remote_wobbl(job, entry)
    t = R.transport(entry)
    job.remote_dir = job.remote_dir or job.id
    d = shlex.quote(job.remote_dir)
    drv = CalculixDriver(job)
    stage = job.folder / "stage"
    (stage / "work").mkdir(parents=True, exist_ok=True)
    if resume_deck:
        (stage / "work" / f"{drv.name}.inp").write_text(resume_deck)
    else:
        stage_inp(Path(job.input), stage / "work" / "job.inp", job.checkpoint, copy_includes=True)
        shutil.copy2(Path(__file__).with_name("frdsummary.py"), stage / "frdsummary.py")
    ccx = entry.get("ccx") or "ccx"
    wait = 'while [ ! -f "$HOME/.wobbl-ready" ]; do sleep 10; done\n' if entry.get("cloud") else ""
    (stage / "run.sh").write_text(RUN_SH.format(id=job.id, run=job.restarts, t=job.threads, name=drv.name,
                                                ccx=shlex.quote(ccx), wait=wait))
    if entry.get("cloud"):
        _wait_for_ssh(t)
    t.put(stage, job.remote_dir)
    shutil.rmtree(stage, ignore_errors=True)          # the remote holds the input now
    if rin_from:
        t.check(f"cd {d}/work && mv {shlex.quote(rin_from)} {shlex.quote(drv.name + '.rin')}")
    t.check(f"cd {d} && rm -f exit_code solver.pid watcher.pid started && {LAUNCH}")
    job.save()
    write_json(job.folder / "status.json", {"state": "running", "progress": 0.0, "updated": time.time(),
                                            "started": time.time(), "launched": time.time(),
                                            "detail": f"started on {t.name}", "run": job.restarts})


def _wait_for_ssh(t: R.Transport, timeout: float = 600.0):
    """A freshly rented machine takes a minute or two before it answers SSH."""
    t0 = time.time()
    while True:
        try:
            if t.run("true", timeout=30).returncode == 0:
                return
        except R.RemoteError:
            pass
        if time.time() - t0 > timeout:
            raise JobError(f"{t.name} did not answer SSH within {timeout / 60:.0f} minutes",
                           "Check that the security group lets SSH in from this machine")
        time.sleep(10)


def _remote_status(job: Job) -> dict:
    """Read the small files of a remote job back and settle its state, pulling the compact results once done."""
    from ..estimate import duration
    from .watch import eta

    st = json.loads((job.folder / "status.json").read_text()) if (job.folder / "status.json").exists() else {}
    if st.get("state") in ("done", "failed", "cancelled") and job.pulled:
        return st
    t = _transport(job)
    if job.kind == "wobbl":
        return _remote_status_wobbl(job, st, t)
    d = shlex.quote(job.remote_dir)
    names = ["job"] + [f"job_r{r}" for r in range(1, job.restarts + 1)]
    cmd = f"cd {d} && cat exit_code 2>/dev/null; echo '@@'; " + " ".join(
        f"tail -c 6000 work/{n}.sta 2>/dev/null; echo '@@';" for n in names) + \
        f" {ALIVE}; echo '@@'; cat started 2>/dev/null"
    try:
        out = t.run(cmd, timeout=60).stdout.split("@@")
    except R.RemoteError as e:
        st["reachable"] = False
        st["detail"] = f"{job.runner} cannot be reached right now ({e}). The job keeps running there."
        return st
    code_text, stas, alive_text = out[0].strip(), out[1:1 + len(names)], out[1 + len(names)].strip()
    started = float(out[-1].strip() or st.get("started") or time.time())
    done_before = 0
    for sta in stas[:-1]:
        done_before = ccx_progress(sta, job.steps, done_before)[2]
    frac, detail, _ = ccx_progress(stas[-1], job.steps, done_before)
    elapsed = time.time() - started
    st.update(reachable=True, elapsed_s=round(elapsed, 1), updated=time.time(), run=job.restarts)
    if code_text:
        code = int(code_text) if code_text.lstrip("-").isdigit() else 1
        st["exit_code"] = code
        st["state"] = "done" if code == 0 else "failed"
        st["progress"] = 1.0 if code == 0 else frac
        st["detail"] = f"Finished in {duration(elapsed)}." if code == 0 else f"The solver stopped with exit code {code}."
        pull(job.id, st=st)
    elif alive_text == "alive" or (alive_text == "starting" and time.time() - st.get("launched", 0) < STARTING):
        predicted = (job.estimate or {}).get("time_s")
        if frac is None and predicted:
            frac, detail = min(0.95, elapsed / predicted), "solving (progress estimated from the cost model)"
        st.update(state="running", progress=frac, detail=detail or "running", eta_s=eta(elapsed, frac, predicted))
    elif st.get("state") != "cancelled":
        st["state"] = "interrupted"
        st["detail"] = f"The run stopped on {job.runner} without finishing. Resume it with wobbl jobs resume {job.id}"
    write_json(job.folder / "status.json", st)
    return st


def pull(job_id: str = "", full: bool = False, st: dict | None = None) -> dict:
    """Bring a remote job's compact results back: the summary, the log and the small solver files. With
    ``full`` the .frd files come too."""
    job = find(job_id)
    if job.runner == "local":
        return {"job_id": job.id, "results": str(job.work)}
    t = _transport(job)
    dest = job.folder / "results"
    if job.kind == "wobbl":
        return _pull_wobbl(job, t, dest, full, st)
    names = ["job"] + [f"job_r{r}" for r in range(1, job.restarts + 1)]
    wanted = ["solver.log"] + [f"summary_{n}.json" for n in names]
    wanted += [f"work/{n}{e}" for n in names for e in COMPACT + ((".frd",) if full else ())]
    got = []
    for w in wanted:
        if t.run(f"test -f {shlex.quote(job.remote_dir + '/' + w)}").returncode == 0:
            t.get(f"{job.remote_dir}/{w}", dest / Path(w).name)
            got.append(Path(w).name)
    parts = {}
    for n in names:
        f = dest / f"summary_{n}.json"
        if f.exists():
            parts[n] = json.loads(f.read_text())
    if st is not None and parts:
        st["results"] = merge(parts)
    job.pulled = True
    job.save()
    if job.cloud and not job.cloud.get("terminated") and st is not None and st.get("state") in ("done", "failed"):
        _end_cloud(job)
    return {"job_id": job.id, "results": str(dest), "files": got}


def _pull_wobbl(job: Job, t: R.Transport, dest: Path, full: bool, st: dict | None) -> dict:
    """A remote Wobbl job's result folder, without its work files unless ``full``, and its log."""
    t.get_dir(f"{job.remote_dir}/result", dest / "result", () if full else ("work",))
    t.get(f"{job.remote_dir}/solver.log", dest / "solver.log")
    o = dest / "result" / "outcome.json"
    if st is not None and o.exists():
        d = json.loads(o.read_text())
        rep = dest / "result" / "report.html"
        st["results"] = {"verdict": d.get("verdict"), "headline": d.get("headline"),
                         "report": str(rep) if rep.exists() else ""}
    job.pulled = True
    job.save()
    return {"job_id": job.id, "results": str(dest / "result"),
            "files": sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())}


def _end_cloud(job: Job):
    from .cloud import terminate

    if terminate(job.cloud["instance_id"], job.cloud["region"]):
        job.cloud["terminated"] = time.time()
        job.save()


# --------------------------------------------------------------------------------------------- follow, stop
def status(job_id: str = "") -> dict:
    job = find(job_id)
    st = _remote_status(job) if job.runner != "local" else job.status()
    if job.runner != "local":
        from .report import write_report
        write_report(job, st)
    return describe(job, st)


def describe(job: Job, st: dict | None = None) -> dict:
    from ..estimate import duration

    st = st if st is not None else (job.status() if job.runner == "local" else
                                    json.loads((job.folder / "status.json").read_text()))
    est = job.estimate or {}
    out = {"job_id": job.id, "kind": job.kind, "input": job.input, "runner": job.runner, "threads": job.threads,
           "state": st.get("state"), "progress": st.get("progress"), "detail": st.get("detail", ""),
           "elapsed": duration(st.get("elapsed_s") or 0.0), "report": str(job.folder / "report.html"),
           "log": str(job.folder / "solver.log"), "restarts": job.restarts}
    if st.get("eta_s") is not None and st.get("state") in ACTIVE:
        out["eta"] = duration(st["eta_s"])
    if est:
        out["predicted"] = est.get("time")
        out["predicted_peak_memory_gb"] = est.get("peak_memory_gb")
    for k in ("peak_memory_gb", "cpu_s", "exit_code"):
        if st.get(k) is not None:
            out[k] = st[k]
    res = st.get("results") or {}
    if res.get("peaks"):
        out["peaks"] = [{k: p[k] for k in ("quantity", "max", "min", "time", "step", "xyz") if k in p}
                        for p in res["peaks"]]
        out["result_sets"] = res.get("blocks", 0)
    elif res:
        out["results"] = res
    if job.cloud:
        out["cloud"] = {k: job.cloud.get(k) for k in ("instance_id", "region", "price_per_hour", "terminated")}
    return out


def logs(job_id: str = "", lines: int = 40) -> str:
    job = find(job_id)
    if job.runner != "local":
        return _transport(job).read(f"{job.remote_dir}/solver.log", tail=200 * lines)
    p = job.folder / "solver.log"
    text = p.read_text(errors="replace") if p.exists() else ""
    return "\n".join(text.splitlines()[-lines:])


def cancel(job_id: str = "") -> dict:
    import signal

    job = find(job_id)
    if job.runner != "local":
        d = shlex.quote(job.remote_dir)
        try:
            # the whole process group, so a Wobbl command's own solver children stop with it
            _transport(job).run(f"cd {d} && g=$(cat pgid 2>/dev/null); if [ -n \"$g\" ]; then kill -TERM -\"$g\"; "
                                "else kill $(cat solver.pid) $(cat watcher.pid); fi 2>/dev/null; true")
        except R.RemoteError:
            pass
        st = json.loads((job.folder / "status.json").read_text())
        st.update(state="cancelled", detail=f"Cancelled. Resume it with wobbl jobs resume {job.id}",
                  updated=time.time())
        write_json(job.folder / "status.json", st)
        if job.cloud and not job.cloud.get("terminated"):
            _end_cloud(job)
        return describe(job, st)
    st = job.status()
    if st.get("state") not in ACTIVE:
        return describe(job, st)
    pid = st.get("watcher_pid")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, TypeError, ValueError):
        pass
    for _ in range(60):
        st = job.status()
        if st.get("state") not in ACTIVE:
            break
        time.sleep(0.5)
    return describe(job, st)


def resume(job_id: str = "") -> dict:
    """Carry on a stopped job from its last checkpoint, or run it again from the start when it has none."""
    job = find(job_id)
    st = status(job.id)
    if st["state"] in ACTIVE:
        raise JobError(f"{job.id} is still running", "Follow it with wobbl jobs status")
    if st["state"] == "done":
        raise JobError(f"{job.id} already finished", f"Its report is {st['report']}")
    if job.kind != "calculix":
        job.restarts += 1
        job.save()
        if job.runner != "local":
            _launch_remote_wobbl(job, R.get(job.runner))
        else:
            _launch_local(job)
        return {**describe(job), "resumed_from": "the start, because a Wobbl command has no checkpoint"}
    remote = job.runner != "local"
    read = (lambda n, tail=0: _transport(job).read(f"{job.remote_dir}/work/{n}", tail)) if remote else _reader(job)
    drv = CalculixDriver(job)
    ok, why = drv.can_resume(read)
    entry = ({"host": job.cloud["host"], "root": "wobbl-jobs", "ccx": CLOUD_CCX, "cloud": True} if job.cloud
             else R.get(job.runner)) if remote else None
    if not ok:
        if remote:
            _launch_remote(job, entry)
        else:
            _launch_local(job)
        return {**describe(job), "resumed_from": f"the start, because {why}"}
    done = drv.steps_done(read)
    prev = drv.name
    job.restarts += 1
    deck = f"*RESTART, READ, STEP={done}\n" + "".join(drv.remaining_steps(done))
    if remote:
        _launch_remote(job, entry, deck, rin_from=f"{prev}.rout")
    else:
        rout = job.work / f"{prev}.rout"
        if not rout.exists():
            raise JobError(f"the checkpoint {rout.name} is missing, so {job.id} cannot continue")
        rout.replace(job.work / f"{CalculixDriver(job).name}.rin")
        job.save()
        _launch_local(job, deck)
    return {**describe(job), "resumed_from": f"after step {done} of {len(job.steps)}"}


def list_jobs() -> list[dict]:
    out = []
    for j in all_jobs():
        st = j.status() if j.runner == "local" else (json.loads((j.folder / "status.json").read_text())
                                                     if (j.folder / "status.json").exists() else {})
        out.append({"job_id": j.id, "state": st.get("state"), "progress": st.get("progress"), "runner": j.runner,
                    "input": Path(j.input).name if j.kind != "wobbl" else j.input, "created": j.created})
    return out


__all__ = ["start", "status", "logs", "cancel", "resume", "pull", "list_jobs", "JobError", "sta_rows"]
