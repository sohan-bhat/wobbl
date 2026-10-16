"""wobbl setup, doctor and uninstall."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import click

from .. import __version__
from . import ui

# what setup suggests last: check the install, then the no file question the first run guide starts with
NEXT_STEPS = (("wobbl doctor", "checks every piece"),
              ('wobbl simulate "will a blue LED on 5V with a 47 ohm resistor fry?"', "a first question, no files"),
              ('wobbl simulate part.step "will it hold 200 N?"', "a part of your own"))


def next_steps() -> None:
    ui.section("next")
    for cmd, why in NEXT_STEPS:
        ui.line(f"  {cmd}")
        ui.say(why, style="dim", indent=4)
    ui.console.print()


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

    log = lambda m: ui.say(m)  # noqa: E731
    if not no_solvers:
        ui.section("solvers")
        p = installer.plan()
        ui.files([("platform", p["platform"]), ("installing into", wobbl_home())])
        for k, why in p["unavailable"].items():
            ui.say(f"{k}: {why}", style="yellow")
        if dry_run:
            ui.say(f"would install {', '.join(p['env'] + p['env_x86'])}")
        else:
            try:
                st = installer.install(log, force=force, only=only.split(",") if only else None)
            except installer.SetupError as e:
                if getattr(e, "tail", ""):
                    ui.section("the end of the install log")
                    for ln in e.tail.splitlines():
                        ui.line(f"  {ln}", style="dim")
                ui.fail(str(e), "Run wobbl setup again once the problem above is fixed. Every install log is in "
                                f"{wobbl_home() / 'logs'}")
            ui.console.print()
            asked = st.get("chosen")
            for k, why in st.get("unavailable", {}).items():
                if asked is None or k in asked:
                    ui.say(f"{k} unavailable: {why}", style="yellow")
            size = st.get("size", {})
            envs = sum(v for k, v in size.items() if k.startswith("env"))
            took = f" Setup took {st['seconds']:.0f} s." if st.get("seconds") else ""
            ui.say(f"Solvers take {envs / 1e6:.0f} MB, and all of Wobbl's home folder takes "
                   f"{size.get('total', 0) / 1e6:.0f} MB.{took}")
            if st.get("cache_freed"):
                ui.say(f"Cleared {st['cache_freed'] / 1e6:.0f} MB of downloads left over from the install.",
                       style="dim")
        from ..chem.tools import footprint as chem_footprint
        from ..life.tools import footprint as life_footprint
        from ..mechanism.install import footprint as mechanism_footprint
        from ..motion.tools import footprint
        ui.bullets([f()["line"] for f in (footprint, life_footprint, chem_footprint, mechanism_footprint)])
    if not no_agents and agents != "none":
        ui.section("agents")
        found = ag.detect()
        if agents == "detected":
            chosen = [a for a, ok in found.items() if ok]
            if not chosen:
                ui.say("No coding agents found. Run wobbl setup --agents all to register them anyway.")
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
            ui.say("Backups of changed files are in ~/.wobbl/backups, and wobbl uninstall undoes it all.",
                   style="dim")
    next_steps()


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


def doctor_probes() -> list:
    """Every check doctor runs in a child process, in the order it lists them."""
    from ..setup import installer
    from ..setup.probe import Probe
    P = "wobbl.setup.probe"
    probes = [Probe("gmsh", f"{P}:gmsh_version", timeout=45), Probe("rendering", f"{P}:rendering", timeout=90)]
    probes += [Probe(name, f"{P}:recipe", (name,), timeout=90) for name in installer.RECIPES]
    probes.append(Probe("motion", f"{P}:motion", timeout=45))
    return probes


@dataclass
class _Row:
    name: str
    ok: bool | None
    detail: str
    pending: str = "on demand"
    timed_out: bool = False


@click.command()
@click.option("--json", "as_json", is_flag=True)
def doctor(as_json):
    """Check that every piece Wobbl needs is installed and working."""
    import json

    from ..setup import agents as ag
    from ..setup import installer
    from ..setup import probe
    from ..solvers import base

    if not as_json:
        ui.console.print()
    rows = [_Row("wobbl", True, f"{__version__} on Python {platform.python_version()} ({sys.executable})")]
    mm = installer.micromamba_path()
    rows.append(_Row("micromamba", mm.exists(), str(mm) if mm.exists() else "not installed (run wobbl setup)"))
    probes = doctor_probes()
    pad = max(len(n) for n in ["agent gemini", *(p.name for p in probes), *(r.name for r in rows)])
    if not as_json:
        ui.checklist(rows, lambda r: r, pad=pad)
    results = ui.checklist(probes, probe.run, show=not as_json, pad=pad)
    rows += [_Row(p.name, r.ok, r.detail, p.pending, r.timed_out) for p, r in zip(probes, results)]
    tail = [_Row(a.name, None, a.install_hint, "planned") for a in base.adapters() if a.name in ("Elmer",)]
    for name, st in ag.status().items():
        ok = st["mcp"] and st["skill"]
        tail.append(_Row(f"agent {name}", ok if (st["mcp"] or st["skill"]) else None,
                         "MCP and skill installed" if ok else
                         f"MCP {'yes' if st['mcp'] else 'no'}, skill {'yes' if st['skill'] else 'no'}", "off"))
    rows += tail
    if as_json:
        click.echo(json.dumps({r.name: {"ok": r.ok, "detail": r.detail} for r in rows}, indent=1))
        return
    ui.checklist(tail, lambda r: r, pad=pad)
    ui.console.print()
    slow = [r.name for r in rows if r.timed_out]
    if slow:
        ui.say(f"{', '.join(slow)} gave no answer in time. Run wobbl doctor again, since a first run can be slow "
               "while Python builds its caches, and if it stays stuck run wobbl setup --force.")
    if not all(r.ok for r in rows if r.ok is not None and not r.name.startswith("agent")):
        ui.say("Run wobbl setup to install what is missing.", style="bold")
    ui.console.print()
