"""The wobbl command line.

    wobbl simulate part.step "will it hold 200 N and stay under 60 C?"   any thing, any question
    wobbl part.step "will it crack under 200 N?"                       the same, without the word simulate
    wobbl simulate "a 9 cm rock dropped from 300 m"                    words, names, formulas and specs too
    wobbl simulate --list                                              every domain, its physics and options
    wobbl inspect part.step                                            faces, holes, fillets, thickness
    wobbl check | report | fix | uncertainty | credibility | test-plan wobbl-out/part
    wobbl estimate | jobs | materials | validate | setup | doctor | mcp

Every domain and scenario command (run, thermal, circuit, motion, mechanism, flow and the rest) still works
as a hidden alias, so scripts keep running, but the help shows only simulate and the general verbs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from .. import __version__
from . import ui


# what 'wobbl --help' shows, in this order; every other command is a hidden alias that still works
VISIBLE = ("simulate", "view", "inspect", "check", "report", "fix", "uncertainty", "credibility", "test-plan",
           "estimate", "jobs", "materials", "validate", "setup", "doctor", "mcp")


def _routable(args) -> bool:
    """Whether simulate can take a bare 'wobbl thing "question"', which needs an analysis registered for it."""
    from ..simulate import RouteError, route
    thing, rest = args[0], [a for a in args[1:] if not a.startswith("-")]
    try:
        route(thing, rest[0] if rest else "")
    except (RouteError, KeyError, OSError):
        return False
    return True


class AskGroup(click.Group):
    """Lets 'wobbl part.step "question"' and 'wobbl "question"' work next to the subcommands, through simulate."""

    def resolve_command(self, ctx, args):
        bare = args and args[0] not in self.commands and not args[0].startswith("-")
        if bare and (" " in args[0].strip() or os.path.exists(args[0]) or "." in Path(args[0]).name):
            if _routable(args):
                return "simulate", self.commands["simulate"], list(args)
            if " " in args[0].strip():
                return "ask", self.commands["ask"], ["-", *args]  # a question with no file
            return "ask", self.commands["ask"], args
        return super().resolve_command(ctx, args)

    def format_commands(self, ctx, formatter):
        rows = [(n, self.commands[n].get_short_help_str(limit=formatter.width - 22)) for n in VISIBLE
                if n in self.commands]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


def _out_dir(part: str, out: str | None) -> Path:
    if out:
        return Path(out).resolve()
    base = Path("wobbl-out").resolve() / Path(part).stem
    p, k = base, 2
    while p.exists():
        p = Path(f"{base}-{k}")
        k += 1
    return p


def _strength_options(f):
    opts = [
        click.option("--material", "-m", default=None, help="Material name or custom:E=..,nu=..,yield=.."),
        click.option("--load", "-l", "loads", multiple=True,
                     help="Load, e.g. '200 N down @ top', or '300 N -x bearing @ hole d=6.6' for a bolt or pin "
                          "pressing on one side of a hole (repeatable)."),
        click.option("--fix", "-f", "fixes", multiple=True,
                     help="Support, e.g. 'holes d=4.5', 'bolt holes', 'wall x=min', 'elastic 50 N/mm3 @ z=min' "
                          "(repeatable)."),
        click.option("--target-sf", type=float, default=None, help="Required safety factor."),
        click.option("--quality", type=click.Choice(["fast", "normal", "fine"]), default="normal"),
        click.option("--budget", type=float, default=300.0, show_default=True, help="Seconds for the study."),
        click.option("--mesh-size", type=float, default=None, help="Global element size in mm."),
        click.option("--print-direction", default=None, help="Axis the print layers stack along, e.g. z."),
        click.option("--scale", type=float, default=1.0, help="Scale the geometry (1000 for files in meters)."),
        click.option("--body", type=int, default=None, help="Pick one body from a multi body file."),
        click.option("--param", "params", multiple=True, help="Parameter of a .py part, e.g. thickness=6."),
        click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<part>)."),
        click.option("--json", "as_json", is_flag=True, help="Print the result as JSON."),
        click.option("--no-images", is_flag=True, help="Skip heatmap rendering."),
        click.option("--shell", is_flag=True, help="Solve thin walls as shells on their midsurface, checked against "
                                                    "a coarse solid. Faster on enclosures, covers and tubes."),
        click.option("--layup", default=None, help="Composite ply stack on the walls, inner face out, e.g. "
                                                    "'[0/±45/90]s'. Gives first ply failure by Tsai-Wu and Hashin."),
        click.option("--ply", default="t300-5208", show_default=True,
                     help="Ply material for --layup: t300-5208, im7-8552, as4-3501-6 or e-glass-epoxy."),
        click.option("--ply-thickness", type=float, default=None, help="Cured ply thickness in mm for --layup."),
        click.option("--reference", default="", help="Direction the 0 degree plies run along, e.g. x (default the "
                                                     "part's long axis)."),
        click.option("--quiet", "-q", is_flag=True, help="No progress lines."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


@click.group(cls=AskGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="wobbl")
def cli():
    """Wobbl: simulate anything before you build it.

    Give simulate the thing and the question in plain words, as in
    wobbl simulate part.step "will it hold 200 N and stay under 60 C?", and it
    works out the physics.
    """


@cli.command(hidden=True)
@click.argument("part")
@click.argument("question", required=False, default="")
@_strength_options
@click.pass_context
def ask(ctx, part, question, **kw):
    """Answer a plain English question about a part."""
    from ..ask import detect_domain

    dom = detect_domain(question, part)
    if part == "-" and dom not in ("circuit", "mechanism"):
        ui.fail("pass the part file first: wobbl part.step \"your question\"")
    if dom == "circuit":
        from .circuits import circuit_ask
        return circuit_ask(part, question, kw)
    if dom == "mechanism":
        from .mechanism import mechanism_ask
        return mechanism_ask(part, question, kw)
    if dom == "thermal":
        from .thermal import thermal_ask
        return thermal_ask(part, question, kw)
    _strength(part, question, kw, interpret=True)


@cli.command()
@click.argument("part")
@_strength_options
@click.option("--question", default="", help="The question this run answers (shown in the report).")
def run(part, question, **kw):
    """Static strength study with explicit loads and supports."""
    if not kw["loads"] or not kw["fixes"]:
        ui.fail("wobbl run needs at least one --load and one --fix",
                "Example: wobbl run bracket.step -m pla --fix 'holes d=4.5' --load '60 N down @ top'")
    _strength(part, question, kw, interpret=False)


def _study_from(part, question, kw):
    from ..structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(part=str(Path(part).resolve()), material=kw.get("material") or "auto",
                         target_sf=kw.get("target_sf"), quality=kw.get("quality", "normal"),
                         budget=kw.get("budget", 300.0), question=question, scale=kw.get("scale", 1.0),
                         body=kw.get("body"), mesh_size=kw.get("mesh_size"),
                         print_direction=kw.get("print_direction"))
    st.params = dict(p.split("=", 1) for p in kw.get("params", ()) if "=" in p)
    st.loads = [parse_load(x) for x in kw.get("loads", ())]
    st.fixtures = [parse_fixture(x) for x in kw.get("fixes", ())]
    return st


def _strength(part, question, kw, interpret: bool):
    from ..ask import read_question
    from ..geometry.cad import GeometryError
    from ..geometry.select import SelectionError
    from ..materials import resolve_material
    from ..mesh.mesher import MeshError
    from ..solvers.base import SolverError
    from ..structural.pipeline import analyze_strength
    from ..structural.run import describe_part
    from ..structural.spec import SpecError

    if not Path(part).exists():
        ui.fail(f"file not found: {part}")
    out = _out_dir(part, kw.get("out"))
    try:
        st = _study_from(part, question, kw)
        if st.material != "auto":
            resolve_material(st.material)
        with ui.progress(not kw.get("quiet") and not kw.get("as_json")) as cb:
            cb("geometry", f"reading {Path(part).name}")
            pre = describe_part(st.part, out / "work", st.scale, st.body, st.params)
            guesses = []
            if interpret or st.material == "auto" or not st.loads or not st.fixtures:
                reading = read_question(question, pre[1], st)
                guesses = reading.guesses
                for g in guesses:
                    cb("assume", g)
            if kw.get("layup"):
                o = _laminate(st, kw, out, cb, pre)
            elif kw.get("shell"):
                from ..structural.shell_path import analyze_shell
                o = analyze_shell(st, out, cb, render=not kw.get("no_images"), pre=pre)
            else:
                o = analyze_strength(st, out, cb, render=not kw.get("no_images"), pre=pre)
    except (SpecError, SelectionError, GeometryError, MeshError, SolverError, KeyError, ValueError) as e:
        hint = getattr(e, "hint", "")
        ui.fail(str(e).strip("'\""), hint if hint and hint not in str(e) else "")
    if kw.get("layup"):
        from .thermal import show_generic
        if kw.get("as_json"):
            click.echo(json.dumps(o.to_dict(), indent=1))
        else:
            show_generic(o)
        return
    if guesses:
        o.assumptions = [f"Wobbl guessed: {g}." for g in guesses] + o.assumptions
        from ..structural.report import write_outputs
        write_outputs(o, out)
    if kw.get("as_json"):
        click.echo(json.dumps(o.to_dict(), indent=1))
        return
    show_strength(o)


def _laminate(st, kw, out, cb, pre):
    """The strength study of a composite layup on the part's walls (wobbl run --layup)."""
    from ..composites.study import run_laminate
    from ..composites.tools import study_from

    loads = list(kw.get("loads") or []) or [ld.describe() for ld in st.loads]
    fixes = list(kw.get("fixes") or []) or [fx.describe() for fx in st.fixtures]
    lam = study_from(st.part, kw["layup"], kw.get("ply") or "t300-5208", loads, fixes, kw.get("ply_thickness"),
                     st.target_sf,
                     kw.get("reference") or "", "fast" if st.quality == "fast" else "normal", st.question,
                     st.params)
    lam.scale = st.scale
    return run_laminate(lam, out, cb, pre=pre, render=not kw.get("no_images"))


def show_strength(o):
    from ..units import fmt_length, fmt_stress

    imp = o.imperial
    word = {"holds": "HOLDS", "marginal": "MARGINAL", "fails": "FAILS"}[o.verdict]
    ui.console.print()
    ui.verdict_panel(word, o.verdict, o.headline,
                     [("peak stress", fmt_stress(o.governing_MPa, imp)), ("safety factor", f"{o.safety_factor:.2f}"),
                      ("target", f"{o.target_sf:g}"), ("max deflection", fmt_length(o.max_disp_mm, imp))],
                     o.confidence, o.confidence_reasons)
    ui.section("checks")
    ui.checks_table(o.checks)
    if o.fixes:
        ui.section("how to fix it" if o.verdict != "holds" else "worth knowing")
        for f in o.fixes:
            ui.console.print(f"  [bold]{f['title']}[/bold]" + ("  [dim](auto)[/dim]" if f.get("auto") else ""))
            ui.console.print(f"  [dim]{f['summary']}[/dim]")
    ui.section("assumptions")
    ui.bullets(o.assumptions[:8])
    ui.section("files")
    if o.images.get("stress"):
        ui.console.print(f"  heatmap  {o.images['stress']}")
    ui.console.print(f"  report   {o.report}")
    ui.console.print(f"  summary  {Path(o.workdir) / 'summary.txt'}")
    ui.console.print()


@cli.command()
@click.argument("part")
@click.option("--json", "as_json", is_flag=True)
@click.option("--scale", type=float, default=1.0)
@click.option("--body", type=int, default=None)
def inspect(part, as_json, scale, body):
    """List faces, holes, fillets, sharp corners and wall thickness, with ids for --load and --fix."""
    import tempfile

    from ..geometry.cad import GeometryError
    from ..structural.run import describe_part

    try:
        with tempfile.TemporaryDirectory() as tmp:
            prep, feat = describe_part(part, tmp, scale, body)
    except GeometryError as e:
        ui.fail(str(e))
    d = feat.to_dict()
    d["notes"] = prep.notes
    d["file"] = str(Path(part).resolve())
    if as_json:
        click.echo(json.dumps(d, indent=1))
        return
    from rich.table import Table

    lo, hi = feat.bbox
    size = [h - l for l, h in zip(lo, hi)]
    ui.console.print(f"\n[bold]{Path(part).name}[/bold]  {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} mm, "
                     f"volume {feat.volume / 1000:.4g} cm3, walls about {feat.thickness_typical:.3g} mm "
                     f"(thinnest {feat.thickness_min:.3g} mm)")
    for n in prep.notes:
        ui.console.print(f"[dim]  {n}[/dim]")
    if feat.holes():
        ui.section("holes")
        for h in feat.holes():
            ui.console.print(f"  d {2 * h.radius:.4g} mm at ({', '.join(f'{v:.4g}' for v in h.axis_point)})  "
                             f"[dim]select with 'hole d={2 * h.radius:.3g}' or 'face {h.id}'[/dim]")
    if feat.fillets():
        ui.section("inside fillets")
        rs = []
        for r in sorted(f.radius for f in feat.fillets()):
            if not rs or r > rs[-1] * 1.03:
                rs.append(r)
        ui.console.print("  radii " + ", ".join(f"{r:.3g} mm" for r in rs))
    if feat.corners:
        ui.section("sharp inside corners (stress raisers)")
        for c in feat.corners[:8]:
            ui.console.print(f"  {c.length:.3g} mm long between faces {c.faces[0]} and {c.faces[1]} near "
                             f"({', '.join(f'{v:.3g}' for v in c.midpoint)})")
    ui.section("faces")
    tb = Table(box=None, header_style="dim", padding=(0, 1))
    for col in ("id", "kind", "area mm2", "center", "normal or size"):
        tb.add_column(col)
    for f in sorted(feat.faces.values(), key=lambda f: -f.area)[:40]:
        extra = (", ".join(f"{v:+.2f}" for v in f.normal) if f.normal else
                 (f"r {f.radius:.3g} ({f.role})" if f.radius else f.role))
        tb.add_row(str(f.id), f.kind, f"{f.area:.4g}", ", ".join(f"{v:.3g}" for v in f.centroid), extra)
    ui.console.print(tb)
    if len(feat.faces) > 40:
        ui.console.print(f"[dim]  ... {len(feat.faces) - 40} more (use --json)[/dim]")
    ui.console.print()


@cli.command()
@click.argument("result_dir", default="")
@click.option("--json", "as_json", is_flag=True)
def check(result_dir, as_json):
    """Show the verification checks and confidence of a finished run."""
    o = _load_outcome(result_dir)
    if as_json:
        click.echo(json.dumps({"confidence": o.confidence, "reasons": o.confidence_reasons,
                               "checks": [c.to_dict() for c in o.checks]}, indent=1))
        return
    ui.console.print(f"\n[bold]{Path(o.part).name}[/bold]  confidence "
                     f"[{ui.CONF_STYLE.get(o.confidence, 'bold')}]{o.confidence}[/]")
    ui.checks_table(o.checks)
    ui.section("why")
    ui.bullets(o.confidence_reasons)
    ui.console.print()


@cli.command()
@click.argument("result_dir", default="")
@click.option("--open", "open_it", is_flag=True, help="Open the report in a browser.")
def report(result_dir, open_it):
    """Rebuild the self contained HTML report of a finished run."""
    o = _load_outcome(result_dir)
    if o.domain == "strength":
        from ..structural.report import write_report
        path = write_report(o, Path(o.workdir))
    else:
        path = o.report
    ui.console.print(f"report {path}")
    if open_it:
        import webbrowser
        webbrowser.open(Path(path).resolve().as_uri())


def _latest_dir() -> Path | None:
    root = Path("wobbl-out")
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if (p / "outcome.json").exists()]
    return max(cands, key=lambda p: (p / "outcome.json").stat().st_mtime) if cands else None


def _load_outcome(result_dir: str):
    d = Path(result_dir) if result_dir else _latest_dir()
    if d is None or not (d / "outcome.json").exists():
        ui.fail(f"no Wobbl results in '{result_dir or 'wobbl-out'}'", "Pass the folder a run printed.")
    data = json.loads((d / "outcome.json").read_text())
    if data.get("domain") == "strength":
        from ..structural.outcome import Outcome
        return Outcome.load(d / "outcome.json")
    if data.get("domain") == "motion":
        from ..motion.study import MotionOutcome
        return MotionOutcome.load(d / "outcome.json")
    if data.get("domain") == "mechanism":
        from ..mechanism.study import MechanismOutcome
        return MechanismOutcome.load(d / "outcome.json")
    from ..outcome_generic import GenericOutcome
    return GenericOutcome.load(d / "outcome.json")


@cli.command()
@click.argument("result_dir", default="")
@click.option("--apply", "do_apply", is_flag=True,
              help="Build each automatic fix, solve it again and compare it with the original.")
@click.option("--only", default=None,
              help="Comma list of fix kinds to apply, from fillet, thicken, gusset and material. Thicken works on "
                   "parametric .py parts.")
def fix(result_dir, do_apply, only):
    """Show the fixes for a failing part, and with --apply build them and compare before and after.

    \b
    --apply adds fillets and gussets to any part, changes the thickness and
    fillet parameters of a parametric .py part and leaves its other parameters
    alone, and swaps the material. A material swap reuses the first solve when
    only the stiffness changes, and is solved again when Poisson's ratio moves
    by more than about 0.05, the load depends on the material, or the new
    material is judged by a different stress. The output says which it did.
    """
    from ..fix.apply import check_kinds

    try:
        kinds = check_kinds(only.split(",")) if only else None
    except ValueError as e:
        ui.fail(str(e))
    o = _load_outcome(result_dir)
    if not o.fixes:
        ui.console.print("  no fixes needed: the part meets its target")
        return
    if not do_apply:
        for f in o.fixes:
            ui.console.print(f"  [bold]{f['title']}[/bold]" + ("  [dim](auto)[/dim]" if f.get("auto") else ""))
            ui.console.print(f"  [dim]{f['summary']}[/dim]")
        ui.console.print("\n  run with --apply to re-simulate the automatic ones")
        return
    from ..fix.apply import apply_fixes

    with ui.progress() as cb:
        res = apply_fixes(o.workdir, kinds, cb)
    from rich.table import Table

    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("", "peak MPa", "safety factor", "verdict"):
        tb.add_column(c)
    b = res["before"]
    tb.add_row("before", f"{b['governing_MPa']:.3g}", f"{b['safety_factor']:.2f}", b["verdict"])
    for r in res["after"]:
        if "error" in r:
            tb.add_row(r["fix"], "", "", f"[red]{r['error'][:60]}[/red]")
        else:
            tb.add_row(r["fix"], f"{r['governing_MPa']:.3g}", f"{r['safety_factor']:.2f}", r["verdict"])
    ui.console.print()
    ui.console.print(tb)
    for r in res["after"]:
        if r.get("method"):
            ui.console.print(f"  [dim]{r['fix']} was {r['method']}.[/dim]")
    ui.console.print(f"\n  report {res['report']}\n")


@cli.command()
def materials():
    """List the built in materials and their properties."""
    from rich.table import Table

    from ..materials import all_materials

    tb = Table(box=None, header_style="dim", padding=(0, 1))
    for c in ("key", "name", "made", "E MPa", "limit MPa", "criterion", "density", "k W/mK", "max C"):
        tb.add_column(c)
    for m in all_materials():
        tb.add_row(m.key, m.name, m.process, f"{m.E:g}", f"{m.allowable:g}", "yield" if m.ductile else "tensile",
                   f"{m.density:g}", f"{m.conductivity:g}", f"{m.max_service_temp:g}")
    ui.console.print(tb)


@cli.command()
@click.option("--only", default=None, help="Comma list of case ids.")
@click.option("--out", "-o", default="wobbl-out/validation", show_default=True)
@click.option("--markdown", default=None, help="Also write the results table to this Markdown file.")
def validate(only, out, markdown):
    """Run the validation suite: textbook and NAFEMS cases with known answers."""
    from ..validation.cases import CASES, markdown_table, run_cases

    ids = only.split(",") if only else None
    ui.console.print(f"  running {len([c for c in CASES if not ids or c.id in ids])} cases into {out}")
    res = run_cases(out, ids, log=lambda m: ui.console.print("  " + m))
    table = markdown_table(res)
    (Path(out) / "results.md").write_text(table)
    if markdown:
        Path(markdown).write_text(table)
    passed = sum(r["pass"] for r in res)
    ui.console.print(f"\n  {passed} of {len(res)} cases within tolerance\n")
    if passed != len(res):
        sys.exit(1)


@cli.command()
def mcp():
    """Run the MCP server on stdio (agents start this; 'wobbl setup' registers it)."""
    from ..mcp.server import main as serve

    serve()


def _register():
    from .admin import doctor, setup, uninstall
    from .chem import chem, flame, motor, propellant, runaway
    from .circuits import circuit
    from .crack import crack
    from .credibility import credibility, test_plan, uncertainty, validate_against
    from .flow import flow
    from .mechanism import mechanism
    from .motion import motion
    from .nonlinear import plastic
    from .thermal import thermal

    for c in (setup, doctor, uninstall, thermal, circuit, motion, uncertainty, flow, validate_against, credibility,
              test_plan, crack):
        cli.add_command(c)
    for c in (chem, motor, flame, propellant, runaway, mechanism):
        cli.add_command(c)
    from .life import buckling, fatigue, vibration
    for c in (buckling, vibration, fatigue):
        cli.add_command(c)
    for c in (plastic,):
        cli.add_command(c)
    from .fields import antenna, board, electric, magnetic, sound
    for c in (magnetic, electric, board, antenna, sound):
        cli.add_command(c)
    from .view import view
    cli.add_command(view)
    from .jobs import estimate, jobs, remote
    from .submodel import submodel
    for c in (estimate, jobs, remote, submodel):
        cli.add_command(c)
    from .process import cast
    for c in (cast,):
        c.hidden = True       # the domain router reaches these; the names stay as aliases
        cli.add_command(c)


def _hide_aliases():
    """Keep every domain and scenario command working while the help lists only simulate and the general verbs."""
    from .jobs import jobs, remote
    from .simulate import simulate
    cli.add_command(simulate)
    if "remote" not in jobs.commands:
        jobs.add_command(remote)       # wobbl jobs remote add, next to the jobs it serves
    for name, c in cli.commands.items():
        if name not in VISIBLE:
            c.hidden = True


_register()
_hide_aliases()


def main():
    try:
        cli(prog_name="wobbl")
    except KeyboardInterrupt:
        ui.err.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
