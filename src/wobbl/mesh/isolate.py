"""Run fragile native work (OpenCASCADE, gmsh) in a child process.

Bad CAD can make a mesher hang forever or crash the interpreter. Running each
attempt in a separate Python process means Wobbl can time it out, kill it and
try the next strategy, while the CLI or MCP server keeps running. The child is
started with ``python -m wobbl.mesh.isolate`` rather than multiprocessing, so
scripts that import Wobbl never need an ``if __name__ == "__main__"`` guard.

Set WOBBL_NO_ISOLATE=1 to run everything in process (handy for debugging).
"""
from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
import traceback


class IsolatedTimeout(TimeoutError):
    pass


class IsolatedCrash(RuntimeError):
    pass


def run_isolated(fn, *args, timeout: float = 300.0, **kwargs):
    """Call fn(*args, **kwargs) in a fresh process. Re-raises its exception, or IsolatedTimeout/IsolatedCrash."""
    if os.environ.get("WOBBL_NO_ISOLATE"):
        return fn(*args, **kwargs)
    import wobbl

    name = getattr(fn, "__name__", "task")
    with tempfile.TemporaryDirectory(prefix="wobbl-") as tmp:
        inp, out = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.pkl")
        with open(inp, "wb") as f:
            pickle.dump((fn.__module__, fn.__qualname__, args, kwargs), f, protocol=pickle.HIGHEST_PROTOCOL)
        env = dict(os.environ)
        root = os.path.dirname(os.path.dirname(os.path.abspath(wobbl.__file__)))
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # the child never talks on stdout: an MCP server's stdout carries the protocol
        proc = subprocess.Popen([sys.executable, "-m", "wobbl.mesh.isolate", inp, out], env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise IsolatedTimeout(f"{name} took longer than {timeout:.0f} s") from None
        if not os.path.exists(out):
            tail = (err or b"").decode(errors="replace").strip().splitlines()[-3:]
            raise IsolatedCrash(f"{name} crashed (exit code {proc.returncode}){': ' + ' '.join(tail) if tail else ''}")
        with open(out, "rb") as f:
            status, payload = pickle.load(f)
    if status == "ok":
        return payload
    exc, tb = payload
    if isinstance(exc, BaseException):
        raise exc
    raise RuntimeError(str(exc))


def _main(inp: str, out: str) -> int:
    with open(inp, "rb") as f:
        mod, qual, args, kwargs = pickle.load(f)
    fn = importlib.import_module(mod)
    for part in qual.split("."):
        fn = getattr(fn, part)
    try:
        result = ("ok", fn(*args, **kwargs))
    except BaseException as e:  # noqa: BLE001
        result = ("err", (e, traceback.format_exc()))
    try:
        data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001  an exception type that does not pickle
        if result[0] != "err":
            raise
        e, tb = result[1]
        data = pickle.dumps(("err", (RuntimeError(f"{type(e).__name__}: {e}"), tb)))
    with open(out + ".part", "wb") as f:
        f.write(data)
    os.replace(out + ".part", out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1], sys.argv[2]))
