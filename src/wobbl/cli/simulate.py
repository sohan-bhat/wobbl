"""wobbl simulate: give it the thing and the question, and it works out the physics."""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import click

from . import ui

EPILOG = """\b
Examples
  wobbl simulate bracket.step "will it hold 200 N and stay under 60 C at 12 W?"
  wobbl simulate bracket.step --physics stress,vibration -m 6061-t6 --fix "bolt holes d=4.5" --rpm 6000
  wobbl simulate board.cir --probe "v(out)"
  wobbl simulate rocket.toml "what is the apogee?"
  wobbl simulate "a 9 cm rock dropped from 300 m"
  wobbl simulate "CH4 + 2 O2 -> CO2 + 2 H2O" "how hot is the flame?"
  wobbl simulate --list

\b
Any option an analysis reads works as a flag, like --power "12 W", --rpm 6000
or --supply 5V, and --set key=value does the same. Every guess Wobbl makes is
printed with the exact option that overrides it, and --plan shows the route
and the guesses without running anything."""


def split_args(args: tuple[str, ...]) -> tuple[str, str, dict[str, list[str]]]:
    """The thing, the question and the --name value pairs from what click left unparsed."""
    pos, extra, i = [], {}, 0
    toks = list(args)
    while i < len(toks):
        t = toks[i]
        if t.startswith("--") and len(t) > 2:
            name, eq, val = t[2:].partition("=")
            if not eq:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and not nxt.startswith("--"):
                    val, i = nxt, i + 1
                else:
                    val = "true"
            extra.setdefault(name, []).append(val)
        elif extra:
            raise click.UsageError(f"'{t}' comes after the options; put the thing and the question first")
        else:
            pos.append(t)
        i += 1
    thing = pos[0] if pos else ""
    return thing, " ".join(pos[1:]), extra


def options_from(material, loads, fixes, sets, extra) -> dict[str, list[str]]:
    opts: dict[str, list[str]] = {}
    for s in sets:
        if "=" not in s:
            raise click.UsageError(f"--set needs key=value, got '{s}'")
        k, v = s.split("=", 1)
        opts.setdefault(k.strip(), []).append(v.strip())
    for k, v in extra.items():
        opts.setdefault(k, []).extend(v)
    if material:
        opts["material"] = [material]
    if loads:
        opts.setdefault("load", []).extend(loads)
    if fixes:
        opts.setdefault("fix", []).extend(fixes)
    return opts


def check_options(plan, opts: dict) -> None:
    from ..simulate.runner import settle_options
    takes = {o.name for a in plan.analyses for o in a.all_options()}
    bad = settle_options(plan, opts)
    if bad:
        raise click.UsageError(f"no analysis in this plan reads --{bad[0]}. The plan ({', '.join(plan.names)}) reads "
                               f"{', '.join('--' + t for t in sorted(takes))}")


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    skip = {"a", "an", "the", "will", "is", "it", "does", "do", "how", "what", "with", "of", "on", "at", "to", "can"}
    return "-".join([w for w in words if w not in skip][:4]) or "simulation"


def out_name(plan) -> str:
    p = Path(plan.thing) if plan.thing else None
    if p is not None and p.exists():
        return p.stem if p.is_file() else f"{p.name}-{plan.names[0]}"
    return _slug(plan.thing or plan.question)


def explicit(plan, steps) -> str:
    """The command that runs the same study with nothing left to guess."""
    used: dict[str, list[str]] = {}
    for s in steps:
        for k, v in s.job.used.items():
            used.setdefault(k, v)
    parts = ["wobbl", "simulate", plan.thing or "-"]
    if plan.question:
        parts.append(plan.question)
    parts += ["--physics", ",".join(plan.names)]
    for k, vs in used.items():
        for v in vs:
            parts += [f"--{k}"] if v == "true" else [f"--{k}", v]
    return " ".join(shlex.quote(p) for p in parts)


def show_plan(plan) -> None:
    ui.section("plan")
    for decided, override in plan.reasons:
        ui.console.print(f"  {decided}")
        if override:
            ui.console.print(f"  [dim]override with {override}[/dim]")


def show_step(step) -> None:
    o = step.outcome
    if step.error:
        ui.console.print(f"  [red]{step.analysis.name} did not run: {step.error}[/red]")
        if step.hint:
            ui.console.print(f"  [dim]{step.hint}[/dim]")
        return
    if getattr(o, "domain", "") == "strength":
        from .main import show_strength
        return show_strength(o)
    if getattr(o, "domain", "") == "mechanism":
        from .mechanism import show_mechanism
        return show_mechanism(o)
    if getattr(o, "domain", "") == "plastic":
        from .nonlinear import show
        return show(o)
    if all(hasattr(o, k) for k in ("metrics", "checks", "assumptions", "images", "report")):
        from .thermal import show_generic
        return show_generic(o)
    ui.console.print(f"  {step.get('verdict', '').upper()}  {step.get('headline', '')}")


def show_many(plan, steps) -> None:
    from rich.table import Table

    from ..simulate import overall
    word = overall(steps)
    style = {"pass": "holds", "marginal": "marginal", "fail": "fails"}.get(word, "marginal")
    ui.console.print()
    ui.verdict_panel(word.upper(), style, f"{len(steps)} physics on {Path(plan.thing).name or plan.thing}",
                     [(s.analysis.name, (s.get("verdict") or "error")) for s in steps], _lowest(steps),
                     [f"{s.analysis.name}: {s.error or s.get('headline', '')}"[:110] for s in steps])
    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("physics", "verdict", "confidence", "result"):
        tb.add_column(c)
    for s in steps:
        tb.add_row(s.analysis.name, s.get("verdict") or "error", s.get("confidence") or "", str(s.job.out))
    ui.console.print(tb)


def _lowest(steps) -> str:
    order = ["low", "medium", "high"]
    confs = [s.get("confidence") for s in steps if s.get("confidence") in order]
    return min(confs, key=order.index) if confs else "low"


def show_readings(steps) -> None:
    rows = [(s.analysis.name, k, v) for s in steps for k, v in s.job.readings() if k != "kind"]
    if not rows:
        return
    ui.section("what wobbl read from the question")
    many = len(steps) > 1
    for name, k, vs in rows:
        ui.console.print(f"  {name + '  ' if many else ''}{k} {', '.join(vs)}")
        ui.console.print(f"  [dim]override with {' '.join(f'--{k} {shlex.quote(v)}' for v in vs)}[/dim]")


def show_guesses(steps, title: str = "what wobbl assumed") -> None:
    rows = [(s.analysis.name, t, o) for s in steps for t, o in s.job.assumptions]
    if not rows:
        return
    ui.section(title)
    many = len(steps) > 1
    for name, t, o in rows:
        ui.console.print(f"  {name + '  ' if many else ''}{t}")
        if o:
            ui.console.print(f"  [dim]override with {o}[/dim]")


def show_catalog() -> None:
    from ..simulate import analyses, domains, errors
    for d in domains():
        mine = analyses(d.name)
        ui.console.print(f"\n[bold yellow]{d.name}[/bold yellow]  [dim]{d.summary}[/dim]")
        if not mine:
            ui.console.print("  [dim]no analyses registered yet[/dim]")
        for a in mine:
            opts = " ".join(f"--{o.name}" for o in a.options if o.name != "*")
            ui.console.print(f"  [bold]{a.name:<11}[/bold] {a.summary}")
            if opts:
                ui.console.print(f"  [dim]{'':<11} {opts}[/dim]")
        if d.options:
            ui.console.print(f"  [dim]every {d.name} analysis reads {' '.join('--' + o.name for o in d.options)}"
                             f"[/dim]")
    for mod, why in errors.items():
        ui.console.print(f"[red]  {mod} did not load: {why}[/red]")


def _plain(x) -> dict:
    return x if isinstance(x, dict) else x.to_dict() if hasattr(x, "to_dict") else dict(vars(x))


def show_estimate(plan, opts, as_json: bool) -> None:
    """What the plan will cost to run, from the estimator, before anything runs."""
    try:
        from ..estimate import estimate_plan
    except ImportError:
        ui.fail("this version of Wobbl cannot estimate a simulate plan yet")
    try:
        d = _plain(estimate_plan(plan, opts))
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if as_json:
        return click.echo(json.dumps(d, indent=1, default=str))
    try:
        from .jobs import _show_estimate
        _show_estimate(d)
    except (KeyError, TypeError):
        ui.console.print(f"  {d.get('verdict', '')}  {d.get('headline', '')}")


def start_job(plan, opts, remote: str, as_json: bool, out: str | None = None) -> None:
    """Start the plan as a background job, here or on a remote, and say how to follow it."""
    try:
        from ..jobs.manage import start_plan
    except ImportError:
        ui.fail("this version of Wobbl cannot run a simulate plan as a job yet")
    try:
        d = _plain(start_plan(plan, opts, on=remote, out=out))
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if as_json:
        return click.echo(json.dumps(d, indent=1, default=str))
    ui.console.print(f"\n  started job {d.get('job_id', '')} ({d.get('state', '')})"
                     + (f" on {remote}" if remote else ""))
    ui.console.print(f"  follow it with wobbl jobs status {d.get('job_id', '')}\n")


def open_view(path: Path) -> None:
    import webbrowser
    webbrowser.open(Path(path).resolve().as_uri())


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, epilog=EPILOG)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option("--physics", multiple=True, help="The physics to run, e.g. stress,heat, read from the question "
                                                 "when left out. See --list.")
@click.option("--domain", default=None, help="What the thing is when Wobbl reads it wrongly, one of part, "
                                              "circuit, mechanism, flight, fluid, field or chemistry.")
@click.option("--kind", default=None, help="The kind of thing within the domain, e.g. buck, solenoid or rocket.")
@click.option("--material", "-m", default=None, help="Material name, or custom:E=..,nu=..,yield=..")
@click.option("--load", "-l", "loads", multiple=True, help="Load, e.g. '200 N down @ top' (repeatable).")
@click.option("--fix", "-f", "fixes", multiple=True, help="Support, e.g. 'bolt holes d=4.5' (repeatable).")
@click.option("--set", "sets", multiple=True, help="Any option as key=value, e.g. --set power=12W (repeatable).")
@click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<thing>).")
@click.option("--plan", "plan_only", is_flag=True, help="Show the route and the guesses without running.")
@click.option("--quick", is_flag=True, help="Skip images, reports and fixes for a fast look.")
@click.option("--list", "show_list", is_flag=True, help="List every domain, its physics and their options.")
@click.option("--no-open", is_flag=True, help="Do not open the 3D view in the browser at the end.")
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@click.option("--quiet", "-q", is_flag=True, help="No progress lines.")
@click.option("--estimate", "estimate_only", is_flag=True, help="Predict run time, memory and disk, then stop.")
@click.option("--background", is_flag=True, help="Run it as a background job on this machine, followed with wobbl jobs.")
@click.option("--remote", default="", help="Run it as a job on a machine added with wobbl remote add.")
@click.option("--no-images", is_flag=True, hidden=True)
def simulate(args, physics, domain, kind, material, loads, fixes, sets, out, plan_only, quick, show_list, no_open,
             as_json, quiet, estimate_only, background, remote, no_images):
    """Simulate anything, from a CAD part or a circuit to a mechanism, a flight, a flow, a field or a chemical.

    Pass the thing (a file, a name, a formula or a description) and the
    question in plain words. Wobbl works out what the thing is and which
    physics the question needs, runs each one with its checks, and says what
    it assumed and how to override it.
    """
    from ..simulate import RouteError, route
    from ..simulate.runner import run_plan, view

    if show_list:
        return show_catalog()
    thing, question, extra = split_args(args)
    if not thing and not question:
        raise click.UsageError("pass the thing to simulate, e.g. wobbl simulate bracket.step \"will it hold 200 N?\"")
    try:
        plan = route(thing, question, list(physics), domain, kind)
    except RouteError as e:
        ui.fail(str(e), e.hint)
    opts = options_from(material, loads, fixes, sets, extra)
    check_options(plan, opts)
    if not as_json:
        show_plan(plan)
    if plan_only:
        from ..simulate.runner import preview
        steps = preview(plan, opts)
        if as_json:
            click.echo(json.dumps(plan.to_dict() | {
                "assumptions": [{"physics": s.analysis.name, "assumed": t, "override": o}
                                for s in steps for t, o in s.job.assumptions],
                "read": [{"physics": s.analysis.name, "option": k, "values": v} for s in steps
                         for k, v in s.job.readings()],
                "pending": {s.analysis.name: s.error or s.hint for s in steps if s.error or s.hint},
                "explicit": explicit(plan, steps)}, indent=1))
            return
        show_readings(steps)
        show_guesses(steps, "what wobbl would assume")
        for s in steps:
            if s.error or s.hint:
                ui.console.print(f"  [dim]{s.analysis.name}: {s.error or s.hint}[/dim]")
        ui.section("explicit form")
        ui.console.print(f"  {explicit(plan, steps)}", soft_wrap=True)
        return
    if estimate_only:
        return show_estimate(plan, opts, as_json)
    if background or remote:
        return start_job(plan, opts, remote, as_json, out)
    from .main import _out_dir
    out_dir = _out_dir(out_name(plan), out)
    try:
        with ui.progress(not quiet and not as_json) as cb:
            steps = run_plan(plan, opts, out_dir, cb, quick or no_images)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    page = view(out_dir)
    cmd = explicit(plan, steps)
    if as_json:
        from ..simulate import overall
        click.echo(json.dumps({"plan": plan.to_dict(), "verdict": overall(steps), "view": str(page or ""),
                               "explicit": cmd, "result_dir": str(out_dir),
                               "steps": [s.summary() for s in steps]}, indent=1, default=str))
        return
    if len(steps) > 1:
        show_many(plan, steps)
        for s in steps:
            ui.section(s.analysis.name)
            show_step(s)
    else:
        show_step(steps[0])
    show_readings(steps)
    show_guesses(steps)
    ui.section("explicit form")
    ui.console.print(f"  {cmd}", soft_wrap=True)
    if page:
        ui.console.print(f"\n  view     {page}")
        if not no_open and sys.stdout.isatty():
            open_view(page)
    ui.console.print(f"  results  {out_dir}\n")
