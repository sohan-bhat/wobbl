"""wobbl setup, doctor and uninstall."""
from __future__ import annotations

import platform
import sys

import click

from .. import __version__
from . import ui

# what setup suggests last: check the install, then the no file question the first run guide starts with
NEXT_STEP = ("Next, run [bold]wobbl doctor[/bold] to check every piece, then ask a first question that needs no "
             "files with [bold]wobbl \"will a blue LED on 5V with a 47 ohm resistor fry?\"[/bold], or point it at "
             "a part of your own with [bold]wobbl part.step \"will it hold 200 N?\"[/bold]")


@click.command()
@click.option("--agents", default="detected",
              help="Comma list of claude,codex,cursor,gemini, or 'all', 'detected' (default) or 'none'.")
@click.option("--no-agents", is_flag=True, help="Only install solvers.")
@click.option("--no-solvers", is_flag=True, help="Only register agents.")
@click.option("--only", default=None, help="Install only these solvers, e.g. calculix.")
@click.option("--project", is_flag=True, help="Also write a Cursor rule into the current project.")
@click.option("--force", is_flag=True, help="Reinstall from scratch.")
@click.option("--dry-run", is_flag=True, help="Show what would change without changing anything.")
def setup(agents, no_agents, no_solvers, only, project, force, dry_run):
    """Install solvers into ~/.wobbl and connect Wobbl to your coding agents."""
    from pathlib import Path

    from ..setup import agents as ag
    from ..setup import installer
    from ..solvers.locate import wobbl_home

    log = lambda m: ui.console.print(f"  {m}")  # noqa: E731
    if not no_solvers:
        ui.section("solvers")
        p = installer.plan()
        ui.console.print(f"  platform {p['platform']}, installing into {wobbl_home()}")
        for k, why in p["unavailable"].items():
            ui.console.print(f"  [yellow]{k}: {why}[/yellow]")
        if dry_run:
            ui.console.print(f"  would install {', '.join(p['env'] + p['env_x86'])}")
        else:
            try:
                st = installer.install(log, force=force, only=only.split(",") if only else None)
            except installer.SetupError as e:
                ui.fail(str(e), "Rerun 'wobbl setup'; details are in ~/.wobbl/setup.log")
            for k, why in st.get("unavailable", {}).items():
                ui.console.print(f"  [yellow]{k} unavailable: {why}[/yellow]")
            ui.console.print(f"  disk used {installer.disk_usage() / 1e6:.0f} MB")
    if not no_agents and agents != "none":
        ui.section("agents")
        found = ag.detect()
        if agents == "detected":
            chosen = [a for a, ok in found.items() if ok]
            if not chosen:
                ui.console.print("  no coding agents found; run 'wobbl setup --agents all' to register anyway")
        elif agents == "all":
            chosen = list(ag.AGENTS)
        else:
            chosen = [a.strip() for a in agents.split(",") if a.strip()]
            bad = [a for a in chosen if a not in ag.AGENTS]
            if bad:
                ui.fail(f"unknown agent {bad}; choose from {', '.join(ag.AGENTS)}")
        if chosen:
            r = ag.register(chosen, log, dry_run)
            if project:
                r.project_rules(Path.cwd())
            ui.console.print("  [dim]backups of changed files are in ~/.wobbl/backups; undo with "
                             "'wobbl uninstall'[/dim]")
    ui.console.print(f"\n  {NEXT_STEP}\n")


@click.command()
@click.option("--all", "everything", is_flag=True, help="Also delete ~/.wobbl (solvers, backups, cache).")
@click.option("--agents", default="all", help="Which agent registrations to remove.")
@click.confirmation_option(prompt="Remove Wobbl's agent registrations?")
def uninstall(everything, agents):
    """Remove the MCP registrations and skills Wobbl installed, and optionally ~/.wobbl."""
    from ..setup import agents as ag
    from ..setup import installer

    chosen = list(ag.AGENTS) if agents == "all" else [a.strip() for a in agents.split(",")]
    ag.Registrar(lambda m: ui.console.print(f"  {m}")).uninstall(chosen)
    if everything:
        installer.remove_all(lambda m: ui.console.print(f"  {m}"))
    ui.console.print("  done. The wobbl command itself is removed with 'pipx uninstall wobbl' or "
                     "'uv tool uninstall wobbl'.")


@click.command()
@click.option("--json", "as_json", is_flag=True)
def doctor(as_json):
    """Check that every piece Wobbl needs is installed and working."""
    import json

    rows, data = [], {}

    def add(name, ok, detail, key=None):
        rows.append((name, ok, detail))
        data[key or name] = {"ok": ok, "detail": detail}

    add("wobbl", True, f"{__version__} on Python {platform.python_version()} ({sys.executable})")
    try:
        import gmsh
        add("gmsh", True, gmsh.__version__)
    except Exception as e:  # noqa: BLE001
        add("gmsh", False, f"not importable: {e}")
    try:
        import pyvista
        import vtk
        add("pyvista", True, f"{pyvista.__version__} (VTK {vtk.vtkVersion.GetVTKVersion()})")
    except Exception as e:  # noqa: BLE001
        add("pyvista", False, f"not importable: {e}")
    add("rendering", *_render_test())
    from ..setup import installer
    from ..solvers import base
    mm = installer.micromamba_path()
    add("micromamba", mm.exists(), str(mm) if mm.exists() else "not installed (run wobbl setup)")
    st = installer.verify(lambda m: None)
    for name, recipe in installer.RECIPES.items():
        info = st["solvers"].get(name)
        if info:
            add(name, info["works"], f"{info['version']} at {info['path']}" + (f" ({info['error']})" if
                                                                             info["error"] else ""))
        elif recipe.core:
            add(name, False, st["unavailable"].get(name, "not installed; run 'wobbl setup'"))
        else:
            add(name, None, f"installs the first time you run a {recipe.domains[0]} study")
    for a in base.adapters():
        if a.name in ("OpenFOAM", "MuJoCo", "Elmer"):
            add(a.name, None, a.install_hint)
    from ..setup import agents as ag
    for name, s in ag.status().items():
        ok = s["mcp"] and s["skill"]
        add(f"agent {name}", ok if (s["mcp"] or s["skill"]) else None,
            ("MCP and skill installed" if ok else f"MCP {'yes' if s['mcp'] else 'no'}, skill "
                                                  f"{'yes' if s['skill'] else 'no'}"))
    if as_json:
        click.echo(json.dumps(data, indent=1))
        return
    from rich.table import Table
    from rich.text import Text

    tb = Table(box=None, padding=(0, 2), show_header=False)
    tb.add_column()
    tb.add_column()
    tb.add_column()
    for name, ok, detail in rows:
        if ok:
            mark = Text("ok", style="bold green")
        elif ok is None:
            mark = Text("off" if name.startswith("agent") else "on demand" if name in installer.RECIPES else
                        "planned", style="dim")
        else:
            mark = Text("missing", style="bold red")
        tb.add_row(mark, name, Text(detail, style="dim"))
    ui.console.print()
    ui.console.print(tb)
    ui.console.print()
    if not all(ok for _, ok, _ in rows if ok is not None and not _.startswith("agent")):
        ui.console.print("  run [bold]wobbl setup[/bold] to install what is missing\n")


def _render_test():
    try:
        import tempfile
        from pathlib import Path

        import pyvista as pv
        pv.OFF_SCREEN = True
        with tempfile.TemporaryDirectory() as tmp:
            p = pv.Plotter(off_screen=True, window_size=(64, 64))
            p.add_mesh(pv.Sphere())
            p.screenshot(str(Path(tmp) / "t.png"))
            p.close()
        return True, "off screen VTK rendering works"
    except Exception as e:  # noqa: BLE001
        return True, f"VTK cannot render here ({str(e)[:60]}); Wobbl will use its matplotlib fallback"
