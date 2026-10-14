"""wobbl estimate, wobbl jobs and wobbl remote: any run, known before it starts and run where it fits.

All three take any Wobbl command as it would be typed, of any domain, as well as solver decks.

    wobbl estimate run part.step -m pla ...         time, memory and disk before it runs, with options
    wobbl estimate deck.inp
    wobbl estimate --calibrate                      measure this machine
    wobbl jobs start thermal pack.py ...            run in the background, here or with --on a remote
    wobbl jobs status | logs | cancel | resume | pull | list
    wobbl remote add workstation me@box             a machine you own, reached over SSH
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from . import ui


def _show_estimate(d: dict):
    from ..estimate import duration

    size = d["size"]
    what = ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in size.items()
                     if isinstance(v, int) and k in ("nodes", "elements", "cells", "cycles", "solves", "dof"))
    if size.get("analyses"):
        name, kind = d["target"][:70] + ("..." if len(d["target"]) > 70 else ""), " and ".join(size["analyses"])
    elif d["solver"].startswith("wobbl "):
        name, kind = "wobbl " + (d["target"][:70] + ("..." if len(d["target"]) > 70 else "")), d["procedure"]
    else:
        name = Path(d["target"]).name if d["target"] else "a run given by size"
        kind = f"{d['solver']} {d['procedure']}"
    ui.console.print(f"\n[bold]{name}[/bold]  {kind}" + (f", {what}" if what else ""))
    style = {"fits": "bold green", "long": "bold yellow", "too long": "bold red", "does not fit": "bold red",
             "unknown": "bold yellow"}
    m = d["machine"]
    where = f"{m['threads']} cores of {m['name']}" if m["threads"] > 1 else f"one core of {m['name']}"
    if d["verdict"] == "unknown":
        ui.console.print(f"  [{style['unknown']}]UNKNOWN[/]  no measure of this run yet")
    else:
        ui.console.print(f"  [{style.get(d['verdict'], 'bold')}]{d['verdict'].upper()}[/]  about "
                         f"{duration(d['time_s'])} on {where}, {d['peak_memory_gb']:.3g} GB of memory at the peak, "
                         f"{d['disk_gb']:.3g} GB on disk")
    for p in d["problems"]:
        ui.console.print(f"  [red]{p}[/red]")
    if d["options"]:
        ui.section("options")
        ui.bullets([o["summary"] for o in d["options"]], style="")
    ui.section("how this was predicted")
    ui.bullets(d["basis"])
    ui.console.print()


class PassThrough(click.Command):
    """A command that wraps another Wobbl command. Options before the wrapped command belong to this one, and
    everything from the wrapped command on is passed through untouched, its own options included."""

    def parse_args(self, ctx, args):
        from ..jobs.workload import is_command

        takes = {o: not p.is_flag for p in self.params if isinstance(p, click.Option) for o in p.opts}
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                break
            if a.startswith("-"):
                i += 2 if takes.get(a.split("=", 1)[0]) and "=" not in a else 1
                continue
            if is_command(a):
                return super().parse_args(ctx, args[:i] + ["--"] + args[i:])
            break
        return super().parse_args(ctx, args)


def _machine(on: str, threads: int, where: str = "."):
    from .. import estimate as E

    if on:
        from ..jobs import remote as R
        try:
            R.get(on)
        except R.RemoteError as e:
            ui.fail(str(e))
        m = R.machines()[on]
        if threads:
            m.threads = threads
        return m
    return E.this_machine(where, threads) if threads else None


@click.command(cls=PassThrough)
@click.argument("target", required=False, default="")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option("--solver", default="", help="calculix, openradioss, openfoam or ngspice, for a run given by size.")
@click.option("--nodes", type=float, default=0, help="Nodes of a model that is not on this machine.")
@click.option("--elements", type=float, default=0, help="Elements (cells for OpenFOAM).")
@click.option("--end-time", type=float, default=0.0, help="Explicit end time in seconds, e.g. 0.2.")
@click.option("--time-step", type=float, default=0.0, help="Explicit time step in seconds (default 1e-6).")
@click.option("--iterations", type=int, default=0, help="CFD iterations or circuit time points.")
@click.option("--procedure", default="static", help="CalculiX procedure for a run given by size: static or heat.")
@click.option("--on", "on", default="", help="Predict it on a remote added with wobbl remote add.")
@click.option("--threads", type=int, default=0, help="Cores the run gets (default: as a job would get).")
@click.option("--max-hours", type=float, default=None, help="Longest acceptable run (default 24).")
@click.option("--calibrate", is_flag=True, help="Measure this machine with three small CalculiX solves first.")
@click.option("--json", "as_json", is_flag=True)
def estimate(target, args, solver, nodes, elements, end_time, time_step, iterations, procedure, on, threads,
             max_hours, calibrate, as_json):
    """Predict run time, peak memory and disk before anything runs, and what to do when it will not fit.

    Give any Wobbl command as you would type it, a solver deck or case folder, or a size.

    \b
    wobbl estimate run part.step -m pla --fix "x=min" --load "50 N down @ top"
    wobbl estimate thermal pack.py -m li-ion-cell --power "6 W"
    wobbl estimate deck.inp
    wobbl estimate --solver openradioss --elements 1.5e6 --end-time 0.2
    wobbl estimate --calibrate
    """
    from .. import estimate as E
    from ..jobs.workload import is_command

    if calibrate:
        ui.console.print("  calibrating CalculiX on this machine")
        try:
            r = E.calibrate(lambda m: ui.console.print("  " + m), threads or None)
        except Exception as e:  # noqa: BLE001
            ui.fail(str(e), getattr(e, "hint", ""))
        ui.console.print(f"  this machine runs at {r['time_factor']:.2f} times the modeled time and "
                         f"{r['memory_factor']:.2f} times the modeled memory, from {r['calibrated_from']} runs\n")
        if not target and not solver:
            return
    command = bool(target) and is_command(target)
    if args and not command:
        ui.fail(f"unexpected {' '.join(args)}", "Put options before the deck, or give a Wobbl command")
    machine = _machine(on, threads, str(Path(target).resolve().parent) if target and not command else ".")
    try:
        from ..geometry.cad import SUPPORTED
        if command:
            est = E.estimate_command([target, *args], machine, max_hours)
        elif target and (Path(target).suffix.lower() in SUPPORTED or Path(target).suffix.lower() == ".py"):
            est = E.estimate_command(["run", target], machine, max_hours)
        elif target:
            est = E.estimate_deck(target, machine, max_hours)
        elif solver:
            est = E.estimate_size(solver, int(nodes), int(elements), end_time, time_step, iterations, procedure,
                                  machine=machine, max_hours=max_hours)
        else:
            ui.fail("give a Wobbl command, a solver deck or case folder, or a size with --solver",
                    "Example: wobbl estimate run part.step -m pla")
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    d = est.to_dict()
    if as_json:
        click.echo(json.dumps(d, indent=1))
        return
    _show_estimate(d)


# ------------------------------------------------------------------------------------------------------ jobs
@click.group()
def jobs():
    """Long runs in the background, here or on a machine you own, that survive a closed terminal and resume."""


def _print_job(d: dict):
    frac = d.get("progress")
    pct = f"{100 * frac:.0f}%" if frac is not None else ""
    ui.console.print(f"\n[bold]{d['job_id']}[/bold]  {d['state']} {pct}  on {d['runner']}, {d['threads']} "
                     f"thread{'s' if d['threads'] != 1 else ''}")
    if d.get("detail"):
        ui.console.print(f"  {d['detail']}")
    line = f"  elapsed {d['elapsed']}"
    if d.get("eta"):
        line += f", about {d['eta']} left"
    if d.get("predicted"):
        line += f", predicted {d['predicted']}"
    ui.console.print(line)
    for p in d.get("peaks", []):
        where = ", ".join(f"{v:.4g}" for v in p["xyz"]) if p.get("xyz") else ""
        ui.console.print(f"  largest {p['quantity']} {p['max']:.5g} at time {p['time']:g}" +
                         (f" at ({where})" if where else ""))
    if d.get("cloud"):
        c = d["cloud"]
        ui.console.print(f"  cloud {c['instance_id']} in {c['region']}" +
                         (", terminated" if c.get("terminated") else ", running"))
    ui.console.print(f"  report {d['report']}")
    if d.get("resumed_from"):
        ui.console.print(f"  resumed from {d['resumed_from']}")
    ui.console.print()


def _job_error(e):
    from ..jobs.manage import JobError

    if isinstance(e, JobError):
        ui.fail(str(e), e.hint)
    ui.fail(str(e), getattr(e, "hint", ""))


@jobs.command(cls=PassThrough)
@click.argument("target")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option("--on", "on", default="", help="Run on a remote added with wobbl remote add.")
@click.option("--threads", type=int, default=0, help="Cores to use (default: up to 4 here, all on a remote).")
@click.option("--no-checkpoint", is_flag=True, help="Do not write restart data after each step.")
@click.option("--force", is_flag=True, help="Start even when the estimate says it will not fit.")
@click.option("--max-hours", type=float, default=None, help="Longest acceptable run (default 24).")
@click.option("--cloud", type=click.Choice(["aws"]), default=None, help="Rent a machine with your own AWS CLI.")
@click.option("--instance", default="", help="Cloud instance type, e.g. c7i.8xlarge.")
@click.option("--region", default="us-east-1", show_default=True)
@click.option("--key-name", default="", help="Your EC2 key pair, for SSH to the rented machine.")
@click.option("--security-group", default="", help="A security group of yours that lets SSH in.")
@click.option("--price-per-hour", type=float, default=None, help="Your own price for the instance, USD.")
@click.option("--confirm", default="", help="The token of a cloud plan you agreed to.")
@click.option("--json", "as_json", is_flag=True)
def start(target, args, on, threads, no_checkpoint, force, max_hours, cloud, instance, region, key_name,
          security_group, price_per_hour, confirm, as_json):
    """Start a CalculiX deck, or a whole Wobbl study, in the background.

    \b
    wobbl jobs start deck.inp
    wobbl jobs start deck.inp --on workstation
    wobbl jobs start run part.step -m pla --fix "x=min" --load "50 N down @ top"
    wobbl jobs start deck.inp --cloud aws --instance c7i.8xlarge --key-name me --security-group sg-1
    """
    from ..jobs import manage
    from .main import cli

    argv = [target, *args] if target in cli.commands else []
    try:
        if cloud:
            r = _cloud_start(target, threads, max_hours, instance, region, key_name, security_group,
                             price_per_hour, confirm, not no_checkpoint)
            if r is None:
                return
        else:
            r = manage.start(target if not argv else "", argv, on, threads, not no_checkpoint, force, max_hours)
    except Exception as e:  # noqa: BLE001
        _job_error(e)
    if as_json:
        click.echo(json.dumps(r, indent=1))
        return
    _print_job(r)
    ui.console.print(f"  follow it with wobbl jobs status {r['job_id']}\n")


def _cloud_start(target, threads, max_hours, instance, region, key_name, security_group, price, confirm,
                 checkpoint):
    from .. import estimate as E
    from ..jobs import cloud as C
    from ..jobs import manage
    from ..jobs.store import new_id

    if not instance:
        raise C.CloudError("pick a machine with --instance, for example the one wobbl estimate suggests")
    from ..decks import read_inp

    cost = E.deck_cost(read_inp(target), checkpoint)
    job = new_id(target)
    p = C.plan(cost, instance, region, key_name, security_group, job, max_hours, price)
    ui.section("cloud plan")
    ui.console.print(f"  {p.instance} in {p.region}, {p.cores} cores and {p.ram_gb:g} GB at ${p.price_per_hour:.4g} "
                     f"an hour ({C.PRICES})")
    ui.console.print(f"  predicted run {E.duration(p.time_s)}, about {p.hours:.2f} h with setup, so about "
                     f"${p.cost_usd:.2f}")
    ui.console.print(f"  the machine shuts itself down after {p.max_hours:g} h whatever happens, so at most "
                     f"${p.max_cost_usd:.2f}")
    ui.console.print("  it runs these commands with your AWS CLI and its credentials")
    ui.bullets(p.commands)
    if confirm != p.token:
        if confirm or not sys.stdin.isatty():
            ui.console.print(f"\n  nothing was started. To rent this machine, run the same command with "
                             f"--confirm {p.token}\n")
            return None
        if click.prompt("\n  type yes to rent this machine", default="no").strip().lower() != "yes":
            ui.console.print("  nothing was started\n")
            return None
        confirm = p.token
    launched = C.launch(p, confirm)
    ui.console.print(f"  rented {launched['instance_id']}, waiting for it to accept SSH")
    return manage.start(target, threads=threads, checkpoint=checkpoint, max_hours=max_hours,
                        cloud={**launched, "job": job, "cores": p.cores, "ram": p.ram_gb * 1e9})


@jobs.command(name="run-plan", hidden=True)
@click.argument("thing")
@click.option("--question", default="")
@click.option("--domain", default="")
@click.option("--kind", default="")
@click.option("--physics", multiple=True)
@click.option("--opt", "opts", nargs=2, multiple=True)
@click.option("--quick", is_flag=True)
@click.option("--out", "-o", default="result")
def run_plan_cmd(thing, question, domain, kind, physics, opts, quick, out):
    """Run a wobbl simulate plan exactly as planned. Background and remote jobs run plans through it."""
    from ..simulate import route, run_plan
    from ..simulate.runner import overall

    try:
        plan = route("" if thing == "-" else thing, question, list(physics) or None, domain or None, kind or None)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    options: dict[str, list[str]] = {}
    for k, v in opts:
        options.setdefault(k, []).append(v)
    try:
        steps = run_plan(plan, options, Path(out), lambda stage, msg: click.echo(f"{stage}: {msg}"), quick)
    except Exception as e:  # noqa: BLE001  a plan of one analysis raises its error, said the way the CLI says it
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    for s in steps:
        click.echo(f"{s.analysis.key}: {s.error or s.get('headline', '')}")
    if all(s.error for s in steps):
        raise SystemExit(1)
    click.echo(f"overall {overall(steps)}")


@jobs.command(name="list")
@click.option("--json", "as_json", is_flag=True)
def list_cmd(as_json):
    """Every job, newest last."""
    from ..jobs import manage

    rows = manage.list_jobs()
    if as_json:
        click.echo(json.dumps(rows, indent=1))
        return
    if not rows:
        ui.console.print("  no jobs yet. Start one with wobbl jobs start deck.inp")
        return
    from rich.table import Table

    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("job", "state", "progress", "on", "input", "started"):
        tb.add_column(c)
    for r in rows:
        p = r.get("progress")
        tb.add_row(r["job_id"], r["state"] or "", f"{100 * p:.0f}%" if p is not None else "", r["runner"],
                   r["input"], time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created"])))
    ui.console.print(tb)


@jobs.command()
@click.argument("job_id", default="")
@click.option("--watch", is_flag=True, help="Keep showing it until the job ends.")
@click.option("--json", "as_json", is_flag=True)
def status(job_id, watch, as_json):
    """Progress, ETA and the peaks so far (the newest job by default)."""
    from ..jobs import manage

    while True:
        try:
            d = manage.status(job_id)
        except Exception as e:  # noqa: BLE001
            _job_error(e)
        if as_json:
            click.echo(json.dumps(d, indent=1))
        else:
            _print_job(d)
        if not watch or d["state"] not in ("starting", "running"):
            return
        time.sleep(10)


@jobs.command()
@click.argument("job_id", default="")
@click.option("--lines", "-n", type=int, default=40, show_default=True)
def logs(job_id, lines):
    """The newest lines the solver printed."""
    from ..jobs import manage

    try:
        click.echo(manage.logs(job_id, lines))
    except Exception as e:  # noqa: BLE001
        _job_error(e)


@jobs.command()
@click.argument("job_id", default="")
def cancel(job_id):
    """Stop a job. Its checkpoint stays, so wobbl jobs resume carries on from it."""
    from ..jobs import manage

    try:
        _print_job(manage.cancel(job_id))
    except Exception as e:  # noqa: BLE001
        _job_error(e)


@jobs.command()
@click.argument("job_id", default="")
def resume(job_id):
    """Carry on a cancelled or interrupted job from its last checkpoint."""
    from ..jobs import manage

    try:
        _print_job(manage.resume(job_id))
    except Exception as e:  # noqa: BLE001
        _job_error(e)


@jobs.command()
@click.argument("job_id", default="")
@click.option("--full", is_flag=True, help="Also fetch the full .frd results, which can be large.")
def pull(job_id, full):
    """Bring a remote job's results back."""
    from ..jobs import manage

    try:
        r = manage.pull(job_id, full)
    except Exception as e:  # noqa: BLE001
        _job_error(e)
    ui.console.print(f"  {len(r.get('files', []))} files in {r['results']}")


# ---------------------------------------------------------------------------------------------------- remote
@click.group()
def remote():
    """Machines you own, reached over SSH with your own keys. Wobbl stores no credentials."""


@remote.command(name="add")
@click.argument("name")
@click.argument("host")
@click.option("--root", default="wobbl-jobs", show_default=True, help="Folder for jobs, relative to the home.")
@click.option("--port", type=int, default=None)
@click.option("--ccx", default="", help="CalculiX on the remote, when it is not on its PATH.")
def remote_add(name, host, root, port, ccx):
    """Add a machine, e.g. wobbl remote add workstation me@192.168.1.20."""
    from ..jobs import remote as R

    try:
        e = R.add(name, host, root, port, ccx)
    except (R.RemoteError, OSError) as err:
        ui.fail(str(err), "Check that ssh reaches it without a password prompt, for example ssh " + host + " true")
    ui.console.print(f"  added {name} with {e.get('cores')} cores, {float(e.get('ram') or 0) / 1e9:.0f} GB, "
                     f"CalculiX {'at ' + e['ccx'] if e.get('ccx') else 'not found, so pass --ccx'}")


@remote.command(name="list")
def remote_list():
    """The machines added so far."""
    from ..jobs import remote as R

    cfg = R.load()
    if not cfg:
        ui.console.print("  none yet. Add one with wobbl remote add NAME user@host")
    for name, e in cfg.items():
        ui.console.print(f"  {name}  {e['host']}  {e.get('cores', '?')} cores, "
                         f"{float(e.get('ram') or 0) / 1e9:.0f} GB, CalculiX {e.get('ccx') or 'not found'}")


@remote.command(name="remove")
@click.argument("name")
def remote_remove(name):
    """Forget a machine. Nothing on it is touched."""
    from ..jobs import remote as R

    ui.console.print(f"  removed {name}" if R.remove(name) else f"  no remote named {name}")
