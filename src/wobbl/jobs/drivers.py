"""What each solver needs to run as a long job: its staged input, its command, its progress and its restart.

A driver never touches the solver's folder directly. It reads files through a ``read(name, tail)`` function and
writes through ``write(name, text)``, so the same driver serves a job in a folder here and a job on a remote
machine reached through a transport.

    CalculiX     the deck gets *RESTART, WRITE, OVERLAY in every step, so a stopped run resumes after its last
                 finished step from a deck that reads the restart file and holds only the remaining steps.
                 OVERLAY keeps one step's restart data on disk, not one per step. Progress comes from the .sta
                 file, which CalculiX writes after every increment.
    wobbl        a whole Wobbl study run in the background. It has no checkpoint, so it starts over.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

from .store import Job

Read = Callable[..., str]
Write = Callable[[str, str], None]
RESTART_WRITE = "*RESTART, WRITE, FREQUENCY=1, OVERLAY\n"   # OVERLAY keeps only the newest step


# ---------------------------------------------------------------------------------------------- CalculiX
def stage_inp(src: Path, dest: Path, restart: bool, copy_includes: bool) -> list[Path]:
    """Copy a deck line by line to ``dest`` with a restart write in every step. Includes are either pointed at by
    absolute path, or copied next to ``dest`` for a remote machine. Returns the files written."""
    written = [dest]
    names: dict[Path, str] = {}

    def copy(path: Path, out_path: Path):
        with open(path, "r", errors="replace") as fin, open(out_path, "w") as out:
            in_step = has_restart = False
            for line in fin:
                up = line.upper()
                if up.startswith("*INCLUDE"):
                    m = re.search(r"INPUT\s*=\s*([^,\n]+)", line, re.I)
                    inc = Path(m.group(1).strip().strip("'\"")) if m else None
                    if inc is not None:
                        inc = inc if inc.is_absolute() else (path.parent / inc)
                        if copy_includes:
                            name = names.setdefault(inc.resolve(), f"inc{len(names) + 1}_{inc.name}")
                            if not (dest.parent / name).exists():
                                written.append(dest.parent / name)
                                copy(inc, dest.parent / name)
                            line = f"*INCLUDE, INPUT={name}\n"
                        else:
                            line = f"*INCLUDE, INPUT={inc.resolve()}\n"
                elif up.startswith("*STEP"):
                    in_step, has_restart = True, False
                elif up.startswith("*RESTART"):
                    has_restart = True
                elif up.startswith("*END STEP") and in_step:
                    if restart and not has_restart:
                        out.write(RESTART_WRITE)
                    in_step = False
                out.write(line)

    copy(Path(src), dest)
    return written


def sta_rows(text: str) -> list[tuple[int, int, float, float]]:
    """(step, increment, total time, step time) for every converged increment in a .sta file."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 7 and p[0].isdigit() and p[1].isdigit() and p[2].isdigit():
            try:
                rows.append((int(p[0]), int(p[1]), float(p[4]), float(p[5])))
            except ValueError:
                continue
    return rows


def ccx_progress(sta: str, steps: list[dict], done_before: int = 0) -> tuple[float | None, str, int]:
    """Fraction done, a one line detail and the number of whole steps finished, from a .sta file. Steps that are
    a single increment (a linear static or a frequency step) report nothing until they finish."""
    periods = [max(float(s.get("period") or 1.0), 1e-30) for s in steps] or [1.0]
    total = sum(periods)
    rows = sta_rows(sta)
    if not rows:
        return (sum(periods[:done_before]) / total if done_before else None), "", done_before
    step, inc, _, st = rows[-1]
    k = min(max(step, 1), len(periods)) - 1
    finished = k + (1 if st >= periods[k] * (1 - 1e-6) else 0)
    frac = (sum(periods[:k]) + min(st, periods[k])) / total
    return min(frac, 1.0), f"step {step} of {len(periods)}, increment {inc}, step time {st:.4g} of {periods[k]:g}", \
        max(finished, done_before)


class CalculixDriver:
    kind = "calculix"

    def __init__(self, job: Job):
        self.job = job

    @property
    def name(self) -> str:
        return "job" if self.job.restarts == 0 else f"job_r{self.job.restarts}"

    def stage(self, dest: Path, copy_includes: bool = False) -> list[Path]:
        return stage_inp(Path(self.job.input), dest / "job.inp", self.job.checkpoint, copy_includes)

    def command(self, ccx: str) -> list[str]:
        return [ccx, "-i", self.name]

    def steps_done(self, read: Read) -> int:
        """Whole steps finished over every run of this job so far."""
        done = 0
        for r in range(self.job.restarts + 1):
            name = "job" if r == 0 else f"job_r{r}"
            done = max(done, ccx_progress(read(f"{name}.sta"), self.job.steps, done)[2])
        return done

    def progress(self, read: Read) -> tuple[float | None, str]:
        frac, detail, _ = ccx_progress(read(f"{self.name}.sta"), self.job.steps, self.steps_done(read))
        return frac, detail

    def can_resume(self, read: Read) -> tuple[bool, str]:
        if not self.job.checkpoint:
            return False, "the job was started without checkpoints, so it starts over"
        done = self.steps_done(read)
        if done == 0:
            return False, "no step had finished yet, so it starts over"
        if done >= len(self.job.steps):
            return False, "every step had already finished"
        return True, f"it continues after step {done} of {len(self.job.steps)}"

    def remaining_steps(self, done: int) -> list[str]:
        """The steps after the first ``done``, from the original deck, each with its restart write."""
        out = []
        for k, block in enumerate(iter_steps(Path(self.job.input))):
            if k >= done:
                if self.job.checkpoint and "*RESTART" not in block.upper():
                    block = re.sub(r"(?im)^\*END STEP", RESTART_WRITE + "*END STEP", block, count=1)
                out.append(block)
        return out


def iter_steps(path: Path):
    """Each *STEP ... *END STEP block of a deck, read line by line so the mesh above them is never held."""
    block = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            up = line.upper()
            if up.startswith("*STEP"):
                block = [line]
            elif block is not None:
                block.append(line)
                if up.startswith("*END STEP"):
                    yield "".join(block)
                    block = None


# ------------------------------------------------------------------------------------------------- wobbl
def wobbl_command(job: Job) -> list[str]:
    """The job's Wobbl command, whose inputs and result folder were made absolute when it started."""
    return [sys.executable, "-m", "wobbl", *job.args]


def result_folder(job: Job) -> Path:
    """Where a Wobbl job writes its result."""
    a = job.args
    for i, x in enumerate(a):
        if x in ("-o", "--out") and i + 1 < len(a):
            return Path(a[i + 1])
        if x.startswith("--out="):
            return Path(x[6:])
    return job.folder / "result"


def wobbl_progress(log: str, levels: int = 5) -> tuple[float | None, str]:
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    if not lines:
        return None, ""
    lv = [int(m.group(1)) for m in re.finditer(r"level (\d+)", log)]
    frac = min(0.95, (max(lv) + 0.5) / levels) if lv else None
    return frac, lines[-1][:160]


DRIVERS = {"calculix": CalculixDriver}
