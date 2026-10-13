"""Terminal presentation: verdict panels, check tables and progress, built on rich."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False, stderr=False)
err = Console(stderr=True, highlight=False)

STATUS_STYLE = {"pass": "bold green", "warn": "bold yellow", "fail": "bold red", "info": "bold cyan",
                "skip": "dim"}
VERDICT_STYLE = {"holds": "bold black on green", "marginal": "bold black on yellow", "fails": "bold white on red",
                 "ok": "bold black on green", "hot": "bold white on red", "warm": "bold black on yellow"}
CONF_STYLE = {"high": "bold green", "medium": "bold yellow", "low": "bold red"}


@contextmanager
def progress(enabled: bool = True):
    """Yield a callback(stage, detail) that shows a live status line with elapsed time."""
    if not enabled or not console.is_terminal:
        t0 = time.time()

        def plain(stage, detail=""):
            if enabled:
                err.print(f"[dim]{time.time() - t0:6.1f}s[/dim] {stage}: {detail}")
        yield plain
        return
    t0 = time.time()
    with console.status("starting", spinner="dots") as st:
        def cb(stage, detail=""):
            st.update(f"[bold]{stage}[/bold] {detail} [dim]({time.time() - t0:.0f}s)[/dim]")
            console.print(f"[dim]{time.time() - t0:6.1f}s[/dim]  {stage:<8} {detail}")
        yield cb


def verdict_panel(word: str, style_key: str, headline: str, metrics: list[tuple[str, str]], confidence: str,
                  reasons: list[str]):
    t = Text()
    t.append(f" {word} ", style=VERDICT_STYLE.get(style_key, "bold"))
    t.append("  ")
    t.append(headline)
    grid = Table.grid(padding=(0, 3))
    for _ in metrics:
        grid.add_column()
    grid.add_row(*[Text(k, style="dim") for k, _ in metrics])
    grid.add_row(*[Text(v, style="bold") for _, v in metrics])
    conf = Text("confidence ", style="dim")
    conf.append(confidence, style=CONF_STYLE.get(confidence, "bold"))
    body = Table.grid()
    body.add_row(t)
    body.add_row("")
    body.add_row(grid)
    body.add_row("")
    body.add_row(conf)
    for r in reasons[:4]:
        body.add_row(Text(f"  {r}", style="dim"))
    console.print(Panel(body, border_style="yellow", padding=(1, 2)))


def checks_table(checks):
    tb = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
    tb.add_column("", width=5)
    tb.add_column("check", style="bold", no_wrap=True)
    tb.add_column("result")
    for c in checks:
        st = c.status if hasattr(c, "status") else c["status"]
        name = c.name if hasattr(c, "name") else c["name"]
        summ = c.summary if hasattr(c, "summary") else c["summary"]
        tb.add_row(Text(st, style=STATUS_STYLE.get(st, "")), name, summ)
    console.print(tb)


def section(title: str):
    console.print()
    console.print(Text(title.upper(), style="bold yellow"))


def bullets(items, style="dim"):
    for i in items:
        console.print(Text(f"  - {i}", style=style))


def fail(msg: str, hint: str = "") -> None:
    err.print(Text("error ", style="bold red") + Text(str(msg)))
    if hint:
        err.print(Text(hint, style="dim"))
    sys.exit(2)
