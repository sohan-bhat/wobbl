"""The checks wobbl doctor runs, each in a child process with a time limit.

A solver that waits on a terminal, a library that hangs on import or a renderer stuck on a display would
otherwise hang doctor with nothing on screen. Here every check is a function named "module:function" that
returns (ok, detail), where ok is True, False, or None for a piece that installs on first use. run() starts it
with 'python -m wobbl.setup.probe', kills its whole process group when the time runs out, and says so.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

MARK = "WOBBL-PROBE "
DEFAULT_TIMEOUT = 60.0


@dataclass
class Probe:
    name: str
    call: str                       # "module:function", run in a child process
    args: tuple = ()
    timeout: float = DEFAULT_TIMEOUT
    pending: str = "on demand"      # what doctor calls an ok of None


@dataclass
class Result:
    name: str
    ok: bool | None
    detail: str
    seconds: float = 0.0
    timed_out: bool = False
    extra: dict = field(default_factory=dict)


def timeout_for(p: Probe) -> float:
    env = os.environ.get("WOBBL_DOCTOR_TIMEOUT", "").strip()
    try:
        return float(env) if env else p.timeout
    except ValueError:
        return p.timeout


def run(p: Probe) -> Result:
    """Run one check in its own process, and give up on it after its time limit."""
    limit = timeout_for(p)
    cmd = [sys.executable, "-m", "wobbl.setup.probe", p.call, *[str(a) for a in p.args]]
    t0 = time.time()
    posix = os.name == "posix"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            text=True, start_new_session=posix)
    try:
        out, errs = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _kill(proc, posix)
        return Result(p.name, False, f"gave no answer within {limit:.0f} s, so doctor stopped it and moved on",
                      time.time() - t0, timed_out=True)
    dt = time.time() - t0
    for ln in reversed(out.splitlines()):
        if ln.startswith(MARK):
            try:
                d = json.loads(ln[len(MARK):])
            except ValueError:
                break
            return Result(p.name, d.get("ok"), str(d.get("detail", "")), dt)
    tail = " ".join((errs or out or "").strip().splitlines()[-2:])[:160]
    return Result(p.name, False, f"the check stopped with exit {proc.returncode}" + (f" ({tail})" if tail else ""),
                  dt)


def _kill(proc: subprocess.Popen, posix: bool) -> None:
    """Stop the check and anything it started."""
    try:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for f in (proc.stdout, proc.stderr):
        try:
            f.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------- the checks themselves
def gmsh_version():
    import gmsh
    return True, gmsh.__version__


def rendering():
    """Draw a tetrahedron with the renderer the reports use."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from ..report.render import render_scalar
    X = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tris = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            render_scalar(Path(tmp) / "t.png", X, np.array([[0, 1, 2, 3]]), tris, X[:, 0], "x", "mm")
    except Exception as e:  # noqa: BLE001
        return False, f"heatmaps cannot be drawn ({str(e)[:60]})"
    return True, "heatmaps draw with numpy and matplotlib, no OpenGL needed"


def recipe(name: str):
    """One solver of the setup registry: run on its tiny problem when it is installed."""
    from .installer import RECIPES, installed
    r = RECIPES[name]
    if r.check is None or not (r.core or installed(r)):
        return (False, "not installed, run 'wobbl setup'") if r.core else \
            (None, f"installs the first time you run a {r.domains[0]} study")
    got = r.check(lambda m: None)
    if isinstance(got, str):
        return (False, got) if r.core else (None, f"installs the first time you run a {r.domains[0]} study")
    return bool(got["works"]), f"{got['version']} at {got['path']}" + (f" ({got['error']})" if got["error"] else "")


def motion():
    from ..motion.tools import footprint
    fp = footprint()
    return True, f"scipy {fp['scipy']}, built in, nothing to download ({fp['code_bytes'] / 1000:.0f} kB of code)"


def _resolve(call: str):
    mod, _, fn = call.partition(":")
    return getattr(importlib.import_module(mod), fn)


def main(argv: list[str]) -> None:
    ok, detail = False, ""
    try:
        got = _resolve(argv[0])(*argv[1:])
        ok, detail = got if isinstance(got, tuple) else (bool(got), "")
    except Exception as e:  # noqa: BLE001  a failed check is a result, not a crash
        ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
    sys.stdout.write("\n" + MARK + json.dumps({"ok": ok, "detail": detail}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
