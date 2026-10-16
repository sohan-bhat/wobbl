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


def explicit(plan, steps, wants=None) -> str:
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
    if wants is not None and wants.format:
        target = wants.target or (wants.preview(plan) if wants.where else None)
        parts += ["--export", wants.format] + (["--out", str(target)] if target else [])
    return " ".join(shlex.quote(p) for p in parts)


class Wants:
    """The file the user asked for and where it goes, from --export and --out or from the words, as 'a 3d gif'
    and 'in downloads'. --out with a file suffix names the file itself; the results then go to their usual
    folder. Words name a folder, and the file in it gets a clear name from the run."""

    def __init__(self, fmt: str | None, out: str | None, said: list[str], words_out: str = ""):
        self.format, self.said, self.where = fmt or "", said, ""
        self.results = out
        suffix = Path(out).suffix.lower() if out else ""
        if suffix in (".gif", ".mp4", ".png", ".html"):
            self.where, self.results = out, None
            self.format = self.format or {".gif": "gif", ".mp4": "mp4", ".png": "png", ".html": "report"}[suffix]
        elif words_out:
            self.where = words_out
        self.target: Path | None = None

    def name(self, plan) -> str:
        return f"wobbl-{out_name(plan)}"

    def preview(self, plan) -> str:
        """Where the file will go, said before the run, without making any folder."""
        from ..export.save import FOLDERS, SUFFIX, known_folder
        if not self.where:
            return f"the results folder, as {self.name(plan)}{SUFFIX[self.format]}"
        if self.where in FOLDERS:
            return str(known_folder(self.where) / f"{self.name(plan)}{SUFFIX[self.format]}")
        p = Path(self.where).expanduser()
        return str(p if p.suffix else p / f"{self.name(plan)}{SUFFIX[self.format]}")


def read_wants(thing: str, question: str, export: str | None, out: str | None):
    """The thing and the question with the words about the result taken out, and what the user wants back."""
    from ..simulate.intent import read_output
    from ..simulate.router import _as_path
    t = read_output(thing) if _as_path(thing) is None and thing != "-" else None
    q = read_output(question)
    fmt = export or (t.format if t else "") or q.format
    words_out = (t.out if t else "") or q.out
    said = (t.said if t else []) + q.said
    wants = Wants(fmt, out, said, "" if out and Path(out).suffix else words_out)
    if words_out and out and not Path(out).suffix:   # words name a folder, --out the results
        wants.where = words_out
    return (t.text if t else thing), q.text, wants


def show_wants(plan, wants) -> None:
    if not wants.format:
        return
    kind = {"gif": "a GIF", "mp4": "an MP4 video", "png": "a PNG picture", "report": "the report page"}[wants.format]
    words = " and ".join(f'"{w}"' for w in wants.said)
    why = f"the words {words} ask for {kind}" if wants.said else f"{kind}, as asked"
    ui.say(f"{why}, saved to {wants.preview(plan)}")
    ui.say(f"override with --export {wants.format} --out PATH", style="dim", indent=4)


def do_export(out_dir: Path, plan, wants, quiet: bool):
    """Write the file the user asked for from the finished run, and say where it went. A failed export never
    loses the run."""
    from ..export import ExportError, export
    if wants.format == "report" and not wants.where:
        return None
    try:
        with ui.progress(not quiet, start=("export", wants.format)) as cb:
            done = export(out_dir, wants.format, wants.where or None, wants.name(plan), cb)
    except ExportError as e:
        ui.say(f"the {wants.format} was not written: {e}", style="red")
        if e.hint:
            ui.say(e.hint, style="dim")
        return None
    wants.target = done.path
    return done


def _overridable(rows) -> None:
    """What Wobbl decided, each with the option that changes it on a dim line below."""
    for decided, override in rows:
        ui.say(decided)
        if override:
            ui.say(f"override with {override}", style="dim", indent=4)


def show_plan(plan) -> None:
    ui.section("plan")
    _overridable(plan.reasons)


def show_step(step) -> None:
    o = step.outcome
    if step.error:
        ui.say(f"{step.analysis.name} did not run: {step.error}", style="red")
        if step.hint:
            ui.say(step.hint, style="dim")
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
    ui.say(f"{step.get('verdict', '').upper()}  {step.get('headline', '')}")


def show_many(plan, steps) -> None:
    from ..simulate import overall
    word = overall(steps)
    style = {"pass": "holds", "marginal": "marginal", "fail": "fails"}.get(word, "marginal")
    ui.console.print()
    ui.verdict_panel(word.upper(), style, f"{len(steps)} physics on {Path(plan.thing).name or plan.thing}",
                     [(s.analysis.name, (s.get("verdict") or "error")) for s in steps], _lowest(steps),
                     [f"{s.analysis.name}: {s.error or s.get('headline', '')}" for s in steps])
    ui.section("results")
    ui.files([(s.analysis.name, s.job.out) for s in steps])


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
    _overridable([(f"{name + '  ' if many else ''}{k} {', '.join(vs)}",
                   " ".join(f"--{k} {shlex.quote(v)}" for v in vs)) for name, k, vs in rows])


def show_guesses(steps, title: str = "what wobbl assumed") -> None:
    rows = [(s.analysis.name, t, o) for s in steps for t, o in s.job.assumptions]
    if not rows:
        return
    ui.section(title)
    many = len(steps) > 1
    _overridable([(f"{name + '  ' if many else ''}{t}", o) for name, t, o in rows])


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


GREETING = re.compile(r"^\W*(?:hi|hello|hey|hiya|howdy|yo|sup|help|thanks|thank you|ok|okay|test)\b|"
                      r"\b(?:what|how)\s+(?:do|does|can|could|would)\s+(?:you|u|yo?i?u|ya|wobbl|this|it)\b|"
                      r"\bwho\s+are\s+(?:you|u)\b|\bwhat\s+(?:is|are)\s+(?:wobbl|you|this)\b", re.I)


def part_question(text: str):
    """The file only domain (a CAD part, say) whose physics a question with no file asks for, or None."""
    from ..simulate.registry import analyses, domains
    for d in domains():
        if not d.extensions or d.words or d.recognize:
            continue      # only domains that need their file; a circuit or a rocket can be described in words
        for a in analyses(d.name):
            if any(re.search(rx, text, re.I) for rx in a.keywords):
                return d
    return None


def unrouted(thing: str, question: str, e, as_json: bool) -> None:
    """A question Wobbl could not route: say which file a part question needs, and otherwise say what Wobbl does
    instead of an error. Returns only when the error should stand, as for a file that does not exist."""
    from ..simulate.router import _as_path
    text = " ".join(x for x in (thing, question) if x).strip()
    if _as_path(thing) is not None or "file not found" in str(e):
        return
    d = part_question(text)
    if d is not None:
        kinds = ", ".join(d.extensions)
        ui.fail(f"this question is about a {d.name}, so Wobbl needs its file",
                f'Pass the file first, as in wobbl simulate bracket.step "{text}". Wobbl reads {kinds} files.')
    introduce(text, as_json)
    sys.exit(0 if not text or GREETING.search(text) else 2)


def introduce(text: str, as_json: bool) -> None:
    """A short answer about what Wobbl does, with a few commands to try."""
    examples = ['wobbl simulate "will a blue LED on 5V with a 47 ohm resistor fry?"',
                'wobbl simulate "a 9 cm rock dropped from 300 m, how fast does it land?"',
                'wobbl simulate bracket.step "will it hold 200 N?"']
    unknown = bool(text) and not GREETING.search(text)
    lead = f'Wobbl found nothing to simulate in "{text}".' if unknown else "Hi, this is Wobbl."
    about = ("It simulates things before you build them. Give it a CAD part, a circuit, a mechanism, a rocket "
             "or a few words, and ask your question in plain words. It picks the physics, runs a real solver, "
             "checks its own answer and says what it assumed.")
    if as_json:
        click.echo(json.dumps({"simulated": False, "message": f"{lead} {about}", "examples": examples}, indent=1))
        return
    ui.console.print()
    ui.say(lead, style="bold")
    ui.say(about)
    ui.section("try one of these")
    for ex in examples:
        ui.line(f"  {ex}")
    ui.console.print()
    ui.say("wobbl simulate --list shows everything Wobbl can simulate.", style="dim")
    ui.console.print()


def usual_time(plan, opts) -> float | None:
    """How long this plan usually takes here, from the runs the estimator has measured, without meshing or
    solving anything. None when any analysis has no measured or bundled time yet."""
    try:
        from ..jobs.workload import PRIORS, analysis_workload, history
        total = 0.0
        for a in plan.analyses:
            w = analysis_workload(a, plan.thing, opts)
            runs, _ = history(w)
            times = sorted(r["measured_time"] for r in runs[-12:] if r.get("measured_time"))
            if times:
                total += times[len(times) // 2]
            elif w.command.split()[0] in PRIORS:
                total += PRIORS[w.command.split()[0]]
            else:
                return None
        return total or None
    except Exception:  # noqa: BLE001  an estimate is a nicety, never a reason to fail a run
        return None


def open_view(path: Path) -> None:
    import webbrowser
    webbrowser.open(Path(path).resolve().as_uri())


class DomainOption(click.Option):
    """--domain, whose help names the domains the registry holds. The names are read when help is shown, so the
    list follows the registry and costs nothing at startup."""

    @property
    def help(self):
        from ..simulate.registry import domains
        names = [d.name for d in domains()]
        listed = ", ".join(names[:-1]) + (" or " if len(names) > 1 else "") + names[-1] if names else "none"
        return f"What the thing is when Wobbl reads it wrongly, one of {listed}."

    @help.setter
    def help(self, value):
        pass   # click sets the static help at decoration; the property above is the one shown


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, epilog=EPILOG)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option("--physics", multiple=True, help="The physics to run, e.g. stress,heat, read from the question "
                                                 "when left out. See --list.")
@click.option("--domain", default=None, cls=DomainOption,
              help="What the thing is when Wobbl reads it wrongly, one of the registered domains.")
@click.option("--kind", default=None, help="The kind of thing within the domain, e.g. buck, solenoid or rocket.")
@click.option("--material", "-m", default=None, help="Material name, or custom:E=..,nu=..,yield=..")
@click.option("--load", "-l", "loads", multiple=True, help="Load, e.g. '200 N down @ top' (repeatable).")
@click.option("--fix", "-f", "fixes", multiple=True, help="Support, e.g. 'bolt holes d=4.5' (repeatable).")
@click.option("--set", "sets", multiple=True, help="Any option as key=value, e.g. --set power=12W (repeatable).")
@click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<thing>), or with --export, or a "
                                                "name ending in .gif, .mp4 or .png, the file to write.")
@click.option("--export", "export_as", type=click.Choice(["gif", "mp4", "png", "report"]), default=None,
              help="Also write the run as a GIF, an MP4 (needs ffmpeg, else a GIF), a PNG or the report page. Words "
                   "like 'a 3d gif' or 'a png on my desktop' do the same.")
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
def simulate(args, physics, domain, kind, material, loads, fixes, sets, out, export_as, plan_only, quick, show_list,
             no_open, as_json, quiet, estimate_only, background, remote, no_images):
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
    thing, question, wants = read_wants(thing, question, export_as, out)
    out = wants.results if wants.format else out
    if not thing and question:
        thing, question = question, ""
    if not thing and not question:
        if wants.format:
            ui.fail(f"what should Wobbl simulate for the {wants.format}?",
                    'Say the thing too, as in wobbl simulate "a rocket thrust test, give me a gif", or pass a '
                    "result folder first.")
        return introduce("", as_json)
    try:
        plan = route(thing, question, list(physics), domain, kind)
    except RouteError as e:
        if not (physics or domain or kind):
            unrouted(thing, question, e, as_json)
        ui.fail(str(e), e.hint)
    opts = options_from(material, loads, fixes, sets, extra)
    check_options(plan, opts)
    if not as_json:
        show_plan(plan)
        show_wants(plan, wants)
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
                **({"export": {"format": wants.format, "to": wants.preview(plan), "said": wants.said}}
                   if wants.format else {}),
                "explicit": explicit(plan, steps, wants)}, indent=1))
            return
        show_readings(steps)
        show_guesses(steps, "what wobbl would assume")
        for s in steps:
            if s.error or s.hint:
                ui.say(f"{s.analysis.name}: {s.error or s.hint}", style="dim")
        ui.section("explicit form")
        ui.line(f"  {explicit(plan, steps, wants)}")
        return
    if estimate_only:
        return show_estimate(plan, opts, as_json)
    if background or remote:
        return start_job(plan, opts, remote, as_json, out)
    from .main import _out_dir
    out_dir = _out_dir(out_name(plan), out)
    try:
        with ui.progress(not quiet and not as_json, eta=usual_time(plan, opts),
                         start=("running", ", ".join(plan.names))) as cb:
            steps = run_plan(plan, opts, out_dir, cb, quick or no_images)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    page = view(out_dir)
    done = do_export(out_dir, plan, wants, quiet or as_json) if wants.format else None
    cmd = explicit(plan, steps, wants)
    if as_json:
        from ..simulate import overall
        click.echo(json.dumps({"plan": plan.to_dict(), "verdict": overall(steps), "view": str(page or ""),
                               "explicit": cmd, "result_dir": str(out_dir),
                               **({"export": done.to_dict()} if done else {}),
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
    ui.line(f"  {cmd}")
    ui.console.print()
    ui.files([("view", page), ("results", out_dir)] + ([(done.format, done.path)] if done else []))
    if done and done.note:
        ui.say(done.note, style="dim")
    if page and not no_open and sys.stdout.isatty():
        open_view(page)
    ui.console.print()
