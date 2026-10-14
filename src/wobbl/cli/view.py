"""wobbl view: open a finished run's interactive 3D simulation in the browser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import ui


@click.command("view")
@click.argument("result", type=click.Path())
@click.option("--no-open", is_flag=True, help="Print the page's path without opening it.")
@click.option("--json", "as_json", is_flag=True, help="Print the page's path as JSON.")
def view(result, no_open, as_json):
    """Open a finished run's interactive 3D simulation in the browser.

    RESULT is the run's output folder (or any file in it). The view is built from the run's own files when it
    does not have one yet, and it plays offline from report.html.
    """
    from ..viz import attach, open_view, simulation_view
    from ..viz.embed import link

    d = Path(result)
    if not d.exists():
        raise click.BadParameter(f"{d} does not exist", param_hint="RESULT")
    d = d.parent if d.is_file() else d
    page = simulation_view(d / "simulation.json") if (d / "simulation.json").exists() else None
    page = page or attach(d)
    if page is None:
        msg = (f"{d} has no 3D view: it holds no scene.wobbl, and nothing in it (outcome.json, playback.json or "
               "trajectory.csv) describes one. Run the study again with this version of Wobbl to write it.")
        if as_json:
            click.echo(json.dumps({"error": msg}))
        else:
            ui.err.print(f"[red]{msg}[/red]")
        sys.exit(2)
    if as_json:
        click.echo(json.dumps({"view": str(page), "url": link(page)}))
    else:
        ui.console.print(f"  view     {page}")
    if not no_open and not as_json:
        if not open_view(page):
            ui.console.print(f"  open it in a browser: {link(page)}")
