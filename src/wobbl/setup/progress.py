"""How an install says what it is doing: one stage per step, with bytes or counts when they are known.

The installer opens stages on the current Reporter, such as 'download micromamba' with its size in bytes or
'install calculix' with its package count, and moves them along. The plain Reporter here writes a line when a
stage starts and one when it ends, with the time it took, through the log function it was given. The command
line sets FACTORY to a reporter that draws a live progress display instead, and anything else (the MCP server,
a script) keeps the plain lines on stderr.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

Log = Callable[[str], None]
FACTORY: Callable[[Log | None], "Reporter"] | None = None
_current: list["Reporter"] = []


class Stage:
    """One step of an install. Use it as a context manager, or call done() or fail() yourself."""

    def __init__(self, reporter: "Reporter", title: str, total: float | None = None, unit: str = ""):
        self.reporter, self.title, self.total, self.unit = reporter, title, total, unit
        self.completed = 0.0
        self.detail = ""
        self.t0 = time.time()
        self.finished: bool | None = None       # None while running, then True or False
        self.handle = None                      # what a drawing reporter keeps for this stage

    def update(self, completed: float | None = None, total: float | None = None, detail: str | None = None,
               advance: float = 0.0) -> None:
        if total is not None:
            self.total = total
        if completed is not None:
            self.completed = completed
        self.completed += advance
        if detail is not None:
            self.detail = detail
        self.reporter._update(self)

    @property
    def seconds(self) -> float:
        return time.time() - self.t0

    def done(self, detail: str | None = None) -> None:
        self._finish(True, detail)

    def fail(self, detail: str | None = None) -> None:
        self._finish(False, detail)

    def _finish(self, ok: bool, detail: str | None) -> None:
        if self.finished is not None:
            return
        if detail is not None:
            self.detail = detail
        if ok and self.total:
            self.completed = self.total
        self.finished = ok
        self.reporter._finish(self)

    def __enter__(self) -> "Stage":
        return self

    def __exit__(self, kind, exc, tb) -> None:
        if kind is None:
            self.done()
        else:
            self.fail(self.detail or str(exc)[:120])


class Reporter:
    """Plain lines: each stage when it starts and when it ends, with the time it took."""

    def __init__(self, log: Log | None = None):
        import sys
        self.log = log or (lambda m: print(f"wobbl: {m}", file=sys.stderr, flush=True))
        self.t0 = time.time()

    def stage(self, title: str, total: float | None = None, unit: str = "") -> Stage:
        st = Stage(self, title, total, unit)
        self._start(st)
        return st

    def note(self, text: str) -> None:
        self.log(text)

    def _start(self, st: Stage) -> None:
        self.log(f"{st.title} ...")

    def _update(self, st: Stage) -> None:
        pass

    def _finish(self, st: Stage) -> None:
        what = "done" if st.finished else "failed"
        size = f", {nbytes(st.total)}" if st.unit == "bytes" and st.total else ""
        self.log(f"{st.title} {what} in {seconds(st.seconds)}{size}" + (f" ({st.detail})" if st.detail else ""))

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


def nbytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.0f} kB"


def seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    m, r = divmod(int(round(s)), 60)
    return f"{m} min {r} s"


def current() -> Reporter:
    """The reporter of the install under way, or a plain one when nothing set one up."""
    return _current[-1] if _current else Reporter()


@contextmanager
def reporting(log: Log | None = None):
    """Make a reporter current for the install inside, reusing the one already current when installs nest."""
    if _current:
        yield _current[-1]
        return
    rep = FACTORY(log) if FACTORY is not None else Reporter(log)
    _current.append(rep)
    rep.open()
    try:
        yield rep
    finally:
        _current.pop()
        rep.close()
