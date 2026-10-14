"""Where jobs live: one folder each under ~/.wobbl/jobs, holding the job, its status, its logs and its report.

    job.json      what to run, where, with how many threads, and the estimate made before it started
    status.json   written by whatever watches the run: state, progress, ETA, peaks so far
    solver.log    everything the solver printed
    report.html   the live report, refreshed while the job runs
    work/         the solver's own folder (local jobs)
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..solvers.locate import wobbl_home

ACTIVE = ("starting", "running")
STALE = 120.0     # seconds without a status update before a job whose watcher is gone counts as interrupted


def jobs_root() -> Path:
    p = wobbl_home() / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Job:
    id: str
    kind: str                      # calculix, openfoam, openradioss, wobbl
    input: str                     # the deck, the case folder, or the wobbl command line
    runner: str = "local"          # local, or the name of a remote
    threads: int = 1
    checkpoint: bool = True
    created: float = field(default_factory=time.time)
    estimate: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)   # CalculiX: each step's procedure and period
    args: list[str] = field(default_factory=list)     # wobbl jobs: the command line after 'wobbl'
    restarts: int = 0              # how many times it was resumed
    remote_dir: str = ""           # remote jobs: the job's folder under the remote root
    cloud: dict = field(default_factory=dict)         # cloud jobs: provider, region and instance
    pulled: bool = False

    @property
    def folder(self) -> Path:
        return jobs_root() / self.id

    @property
    def work(self) -> Path:
        return self.folder / "work"

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json(self.folder / "job.json", asdict(self))

    @classmethod
    def load(cls, job_id: str) -> "Job":
        p = jobs_root() / job_id / "job.json"
        if not p.exists():
            raise KeyError(f"no job '{job_id}'. List them with wobbl jobs list")
        d = json.loads(p.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def status(self) -> dict:
        try:
            st = json.loads((self.folder / "status.json").read_text())
        except (OSError, ValueError):
            st = {"state": "starting", "progress": 0.0, "updated": self.created}
        if st.get("state") in ACTIVE and self.runner == "local":
            pid = st.get("watcher_pid")
            quiet = time.time() - float(st.get("updated", self.created))
            if (pid and not alive(int(pid))) or (not pid and quiet > STALE):
                st["state"] = "interrupted"
                st["detail"] = ("The run stopped without finishing, most likely a restart or a crash. Resume it "
                                f"with wobbl jobs resume {self.id}")
        return st


def new_id(stem: str) -> str:
    clean = "".join(c if c.isalnum() else "-" for c in Path(stem).stem.lower()).strip("-")[:24] or "job"
    return f"{clean}-{uuid.uuid4().hex[:4]}"


def write_json(path: Path, data: dict):
    """Write through a temporary file so a reader never sees half a file."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)


def all_jobs() -> list[Job]:
    out = []
    for p in jobs_root().iterdir():
        if (p / "job.json").exists():
            try:
                out.append(Job.load(p.name))
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(out, key=lambda j: j.created)


def latest() -> Job | None:
    jobs = all_jobs()
    return jobs[-1] if jobs else None


def alive(pid: int) -> bool:
    """Whether a process is still running, without disturbing it."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                       # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a finished child of this very process lingers as a zombie until it is reaped
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True
