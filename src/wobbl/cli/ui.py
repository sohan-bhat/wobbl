"""Terminal presentation: verdict panels, check tables, live status and progress, built on rich.

Everything a result prints goes through the helpers here, so it keeps to a readable width (MAX_WIDTH, or the
terminal when that is narrower) and wraps with a hanging indent. When stdout is not a terminal, or NO_COLOR or
TERM=dumb is set, the same helpers print plain text with no boxes, colors or spinners, for agents and pipes.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Wobbl's own colors, the same as the site's (crash test yellow, and the green, amber and red of its verdict
# badges). The style names the commands already use resolve here first, so plain ANSI green and terminal dim
# never show, and a truecolor terminal gets the exact brand colors.
YELLOW, OK, WARN, FAIL, INK, MUTED, INFO = "#ffd400", "#46d17f", "#ffb020", "#ff5a4a", "#0e0f11", "#9ca099", "#3e9bfe"
THEME = Theme({
    "dim": MUTED, "green": OK, "yellow": YELLOW, "red": FAIL, "cyan": INFO, "blue": INFO,
    "bold green": f"bold {OK}", "bold yellow": f"bold {YELLOW}", "bold red": f"bold {FAIL}",
    "bold cyan": f"bold {INFO}", "bold blue": f"bold {INFO}",
    "bold black on green": f"bold {INK} on {OK}", "bold black on yellow": f"bold {INK} on {WARN}",
    "bold white on red": f"bold {INK} on {FAIL}",
})
console = Console(highlight=False, stderr=False, theme=THEME)
err = Console(stderr=True, highlight=False, theme=THEME)

MAX_WIDTH = 88

STATUS_STYLE = {"pass": "bold green", "warn": f"bold {WARN}", "fail": "bold red", "info": "bold cyan",
                "skip": "dim"}
VERDICT_STYLE = {"holds": "bold black on green", "marginal": "bold black on yellow", "fails": "bold white on red",
                 "ok": "bold black on green", "hot": "bold white on red", "warm": "bold black on yellow",
                 "pass": "bold black on green", "fail": "bold white on red"}
BORDER = {"holds": "green", "ok": "green", "pass": "green", "marginal": WARN, "warm": WARN,
          "fails": "red", "hot": "red", "fail": "red"}
CONF_STYLE = {"high": "bold green", "medium": f"bold {WARN}", "low": "bold red"}


def plain(c: Console | None = None) -> bool:
    """Whether to print plain text: not a terminal, or the user asked for no color."""
    c = c or console
    return (not c.is_terminal or bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb")


def width() -> int:
    """The width results keep to."""
    return max(40, min(console.width, MAX_WIDTH))


def out(renderable="", **kw) -> None:
    """Print a renderable at the result width, without the padding rich leaves at line ends in plain text."""
    if not plain():
        console.print(renderable, width=width(), **kw)
        return
    with console.capture() as cap:
        console.print(renderable, width=width(), **kw)
    console.file.write("\n".join(ln.rstrip() for ln in cap.get().split("\n")))
    console.file.flush()


def _text(s, style: str = "", markup: bool = False) -> Text:
    if isinstance(s, Text):
        return s
    return Text.from_markup(s, style=style) if markup else Text(str(s), style=style)


def say(s="", style: str = "", indent: int = 2, markup: bool = False) -> None:
    """A paragraph, wrapped at the result width with every line indented the same."""
    out(Padding(_text(s, style, markup), (0, 0, 0, indent), expand=False))


def line(s, style: str = "", markup: bool = False) -> None:
    """One line that never wraps, for paths and commands someone may copy."""
    console.print(_text(s, style, markup), soft_wrap=True)


def hanging(rows: list[tuple[str, object]], indent: int = 2, lead_style: str = "dim") -> None:
    """Rows of a short lead and wrapped text, the text lining up under itself."""
    if not rows:
        return
    g = Table.grid(padding=(0, 1))
    g.add_column(no_wrap=True, style=lead_style)
    g.add_column(overflow="fold")
    for lead, body in rows:
        g.add_row(lead, body if isinstance(body, Text) else _text(str(body)))
    out(Padding(g, (0, 0, 0, indent), expand=False))


@contextmanager
def _paused():
    """Stop the live status while something else draws, and start it again after."""
    t = _Ticker.active
    if t is None:
        yield
        return
    t.live.stop()
    try:
        yield
    finally:
        t.live.start()


class _Ticker:
    """The one line live status of a run: a spinner, the stage, the elapsed time and an estimate."""
    active: "_Ticker | None" = None

    def __init__(self, eta: float | None):
        self.t0 = time.time()
        self.eta = eta
        self.stage, self.detail = "starting", ""
        self.spinner = Spinner("dots", style="yellow")
        self.live = Live(self, console=console, refresh_per_second=8, transient=True)

    def __rich__(self):
        dt = time.time() - self.t0
        clock = f"{dt:4.0f}s"
        # an estimate under a few seconds says nothing useful, and a run a second or two slower than a quick
        # one is not "longer than usual", so the estimate only shows for runs that take a while
        if self.eta and self.eta >= 5:
            clock += f" of about {_dur(self.eta)}" if dt < self.eta * 1.5 + 5 else ", longer than usual"
        room = max(10, width() - len(self.stage) - len(clock) - 8)
        detail = self.detail if len(self.detail) <= room else self.detail[:room - 3] + "..."
        t = Text()
        t.append(f"{self.stage:<8} ", style="bold")
        t.append(detail)
        t.append(f"  {clock}", style="dim")
        self.spinner.text = t
        return self.spinner


def _dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


@contextmanager
def progress(enabled: bool = True, eta: float | None = None, start: tuple[str, str] = ("running", "")):
    """Yield a callback(stage, detail) for a run.

    On a terminal it drives one live status line (spinner, stage, detail, elapsed time and the estimate when
    there is one) and leaves a one line summary of the time each stage took. Elsewhere each stage is a plain
    line on stderr, so stdout keeps only the result.
    """
    t0 = time.time()
    stages: list[list] = []

    def mark(stage):
        now = time.time()
        if stages and stages[-1][0] == stage:
            return
        if stages:
            stages[-1][2] = now - stages[-1][1]
        stages.append([stage, now, 0.0])

    if not enabled:
        yield lambda stage, detail="": None
        return
    if plain() or _Ticker.active is not None:
        def plain_cb(stage, detail=""):
            err.print(f"{time.time() - t0:6.1f}s {stage}: {detail}", markup=False, highlight=False)
        yield plain_cb
        return
    tick = _Ticker(eta)
    tick.stage, tick.detail = start
    lock = threading.Lock()

    def cb(stage, detail=""):
        with lock:
            if stage != "assume":
                mark(stage)
            tick.stage, tick.detail = stage, " ".join(str(detail).split())

    _Ticker.active = tick
    tick.live.start()
    ok = False
    try:
        yield cb
        ok = True
    finally:
        tick.live.stop()
        _Ticker.active = None
        if stages:
            stages[-1][2] = time.time() - stages[-1][1]
        total = time.time() - t0
        parts = ", ".join(f"{s} {d:.1f} s" for s, _, d in _merge(stages) if d >= 0.05)
        head = f"done in {total:.1f} s" if ok else f"stopped during {tick.stage} after {total:.1f} s"
        console.print(Text(f"  {head}", style="green" if ok else "red")
                      + Text(f"  {parts}" if parts else "", style="dim"), width=width(), overflow="fold")


@contextmanager
def status(text: str):
    """A spinner with a few words while something short runs, and nothing at all off a terminal."""
    if plain() or _Ticker.active is not None:
        yield
        return
    with console.status(Text(text), spinner="dots", spinner_style="yellow"):
        yield


def _merge(stages):
    """Stage times added up by name, in the order each first ran."""
    order, total = [], {}
    for s, _, d in stages:
        if s not in total:
            order.append(s)
            total[s] = 0.0
        total[s] += d
    return [(s, 0, total[s]) for s in order]


def verdict_panel(word: str, style_key: str, headline: str, metrics: list[tuple[str, str]], confidence: str,
                  reasons: list[str]):
    """The answer: the verdict badge, the headline, the key numbers and the confidence with its reasons."""
    metrics = [(str(k), str(v)) for k, v in metrics]
    if plain():
        out(Text(f"{word}  ", style="bold") + Text(headline), overflow="fold")
        pad = max([len(k) for k, _ in metrics] + [len("confidence")])
        vpad = max([len(v) for _, v in metrics] + [0])
        for k, v in metrics:
            console.print(f"  {k:<{pad}}  {v:>{vpad}}", markup=False, soft_wrap=True)
        console.print(f"  {'confidence':<{pad}}  {confidence}", markup=False, soft_wrap=True)
        for r in reasons[:4]:
            out(Padding(Text(f"- {clip(r, 2 * width() - 20)}"), (0, 0, 0, 4), expand=False))
        return
    head = Table.grid(padding=(0, 2))
    head.add_column(no_wrap=True)
    head.add_column(overflow="fold")
    head.add_row(Text(f" {word} ", style=VERDICT_STYLE.get(style_key, "bold reverse")), Text(headline, style="bold"))
    nums = Table.grid(padding=(0, 3))
    nums.add_column(style="dim", no_wrap=True)
    nums.add_column(justify="right", style="bold", no_wrap=True)
    for k, v in metrics:
        nums.add_row(k, v)
    conf = Text("confidence ", style="dim") + Text(confidence, style=CONF_STYLE.get(confidence, "bold"))
    why = Table.grid(padding=(0, 1))
    why.add_column(no_wrap=True, style="dim")
    why.add_column(overflow="fold", style="dim")
    for r in reasons[:4]:
        why.add_row("  -", clip(r, 2 * width() - 20))
    body = Group(head, Text(""), *([nums, Text("")] if metrics else []), conf, why)
    out(Panel(body, border_style=BORDER.get(style_key, "yellow"), padding=(1, 2), width=width()))


def clip(s: str, n: int) -> str:
    """Text cut at a word near n characters, for a summary that the full section below repeats."""
    s = " ".join(str(s).split())
    if len(s) <= n:
        return s
    cut = s.rfind(" ", 0, n - 3)
    return s[:cut if cut > n // 2 else n - 3].rstrip(" ,;:") + "..."


def checks_table(checks):
    tb = Table(show_header=True, header_style="dim", box=None, padding=(0, 1), pad_edge=False)
    tb.add_column("", width=4, no_wrap=True)
    tb.add_column("check", style="bold", no_wrap=True, max_width=28, overflow="ellipsis")
    tb.add_column("result", overflow="fold", ratio=1)
    for c in checks:
        st = c.status if hasattr(c, "status") else c["status"]
        name = c.name if hasattr(c, "name") else c["name"]
        summ = c.summary if hasattr(c, "summary") else c["summary"]
        tb.add_row(Text(st, style=STATUS_STYLE.get(st, "")), name, summ)
    out(Padding(tb, (0, 0, 0, 2), expand=False))


def section(title: str):
    console.print()
    console.print(Text(title.upper(), style="bold yellow"))


def bullets(items, style="dim"):
    hanging([("-", Text(str(i), style=style)) for i in items], lead_style=style)


def fixes(items) -> None:
    """Each fix as a bold title over its wrapped summary."""
    for f in items:
        say(Text(f["title"], style="bold") + (Text("  (auto)", style="dim") if f.get("auto") else Text("")))
        say(Text(f["summary"], style="dim"), indent=4)


def files(rows: list[tuple[str, object]]) -> None:
    """Labelled paths, the paths lined up and never wrapped."""
    rows = [(k, str(v)) for k, v in rows if v]
    if not rows:
        return
    pad = max(len(k) for k, _ in rows)
    for k, v in rows:
        console.print(Text(f"  {k:<{pad}}  ", style="dim") + Text(v), soft_wrap=True)


def outcome(o, word: str | None = None, style: str | None = None, metrics=None, extra=None,
            files_rows=None, fixes_title: str = "how to fix it", max_assumptions: int | None = None) -> None:
    """The whole result of one analysis: the answer, its checks, fixes, assumptions and files."""
    verdict = getattr(o, "verdict", "")
    console.print()
    verdict_panel(word or verdict.upper(), style or verdict, getattr(o, "headline", ""),
                  list(metrics if metrics is not None else getattr(o, "metrics", [])),
                  getattr(o, "confidence", ""), list(getattr(o, "confidence_reasons", [])))
    if getattr(o, "checks", None):
        section("checks")
        checks_table(o.checks)
    if extra:
        extra()
    if getattr(o, "fixes", None):
        section(fixes_title)
        fixes(o.fixes)
    if getattr(o, "assumptions", None):
        section("assumptions")
        bullets(o.assumptions[:max_assumptions] if max_assumptions else o.assumptions)
    rows = files_rows if files_rows is not None else [*getattr(o, "images", {}).items(),
                                                       ("report", getattr(o, "report", ""))]
    if rows:
        section("files")
        files(rows)


def _mark(ok, pending: str, timed_out: bool = False) -> Text:
    if ok:
        return Text("ok", style="bold green")
    if ok is None:
        return Text(pending, style="dim")
    return Text("timed out" if timed_out else "missing", style="bold red")


def checklist(items, run, workers: int = 4, show: bool = True, pad: int = 0) -> list:
    """Run checks a few at a time and show each one as it runs, then whether it passed.

    items have .name and .pending (the word for an ok of None), and run(item) returns an object with .ok,
    .detail and .timed_out. On a terminal every row has a spinner and its running time until it finishes; off
    a terminal each row prints, in order, once it is done. The results come back in the order of items.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    started: dict[int, float] = {}
    done: dict[int, object] = {}

    def task(i):
        started[i] = time.time()
        try:
            done[i] = run(items[i])
        except Exception as e:  # noqa: BLE001  a broken check is a failed row, never a crashed doctor
            from types import SimpleNamespace
            done[i] = SimpleNamespace(ok=False, detail=f"{type(e).__name__}: {e}", timed_out=False)
        return done[i]

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = [pool.submit(task, i) for i in range(len(items))]
    pad = pad or max([len(it.name) for it in items] + [4])
    try:
        if not show:
            for f in futures:
                f.result()
        elif plain():
            for i, f in enumerate(futures):
                r = f.result()
                mark = _mark(r.ok, items[i].pending, getattr(r, "timed_out", False)).plain
                console.print(f"  {mark:<10} {items[i].name:<{pad}}  {r.detail}", markup=False, soft_wrap=True)
        else:
            spin = Spinner("dots", style="yellow")

            class Rows:
                def __rich__(self):
                    g = Table.grid(padding=(0, 2))
                    g.add_column(no_wrap=True, width=10)
                    g.add_column(no_wrap=True, width=pad)
                    g.add_column(overflow="fold")
                    for i, it in enumerate(items):
                        r = done.get(i)
                        if r is not None:
                            g.add_row(_mark(r.ok, it.pending, getattr(r, "timed_out", False)), it.name,
                                      Text(r.detail, style="dim"))
                        elif i in started:
                            g.add_row(spin, it.name, Text(f"checking  {time.time() - started[i]:.0f}s", style="dim"))
                        else:
                            g.add_row(Text("...", style="dim"), it.name, Text("waiting", style="dim"))
                    return Padding(g, (0, 0, 0, 2))

            with Live(Rows(), console=console, refresh_per_second=10, transient=False):
                for f in futures:
                    while not f.done():
                        time.sleep(0.05)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return [done[i] for i in range(len(items))]


class InstallProgress:
    """The live display of an install on stderr: a line per stage with a spinner, then ok or failed, a bar with
    bytes or a package count and the percent when the size is known, and the time each stage took."""

    def __init__(self, log=None):
        from rich.progress import (BarColumn, DownloadColumn, Progress, ProgressColumn, TaskProgressColumn,
                                   TimeElapsedColumn)

        class State(ProgressColumn):
            def __init__(self):
                super().__init__()
                self.spin = Spinner("dots", style="yellow")

            def render(self, task):
                ok = task.fields.get("ok")
                if ok is None:
                    return self.spin.render(task.get_time())
                return Text("ok", style="bold green") if ok else Text("failed", style="bold red")

        class Bar(BarColumn):
            def render(self, task):
                return super().render(task) if task.total else Text("")

        class Amount(DownloadColumn):
            def render(self, task):
                if not task.total:
                    return Text(task.fields.get("detail", ""), style="dim")
                if task.fields.get("unit") == "bytes":
                    return super().render(task)
                return Text(f"{task.completed:.0f}/{task.total:.0f} {task.fields.get('unit', '')}", style="dim")

        class Percent(TaskProgressColumn):
            def render(self, task):
                return super().render(task) if task.total else Text("")

        self.progress = Progress(State(), "{task.description}", Bar(bar_width=24), Percent(), Amount(),
                                 TimeElapsedColumn(), console=err, refresh_per_second=8,
                                 expand=False)
        self.t0 = time.time()
        self._pause = None

    def open(self):
        self._pause = _paused()
        self._pause.__enter__()
        self.progress.start()

    def close(self):
        self.progress.stop()
        if self._pause is not None:
            self._pause.__exit__(None, None, None)

    def stage(self, title, total=None, unit=""):
        from ..setup.progress import Stage
        st = Stage(self, title, total, unit)
        st.handle = self.progress.add_task(title, total=total, ok=None, unit=unit, detail="")
        return st

    def note(self, text):
        self.progress.console.print(Text(f"  {text}", style="dim"))

    def _update(self, st):
        self.progress.update(st.handle, completed=st.completed, total=st.total, detail=st.detail)

    def _finish(self, st):
        self.progress.update(st.handle, completed=st.completed if not st.total else (st.total if st.finished
                                                                                        else st.completed),
                             total=st.total, ok=bool(st.finished), detail=st.detail)
        self.progress.stop_task(st.handle)


def install_reporter(log=None):
    """What an install reports through: the live display on a terminal, plain lines elsewhere."""
    from ..setup.progress import Reporter
    if plain(err):
        return Reporter(log or (lambda m: err.print(f"wobbl: {m}", markup=False, highlight=False)))
    return InstallProgress(log)


def fail(msg: str, hint: str = "") -> None:
    err.print(Text("error ", style="bold red") + Text(str(msg)))
    if hint:
        err.print(Text(hint, style="dim"))
    sys.exit(2)


def _use_install_progress() -> None:
    from ..setup import progress as setup_progress
    setup_progress.FACTORY = install_reporter


_use_install_progress()
