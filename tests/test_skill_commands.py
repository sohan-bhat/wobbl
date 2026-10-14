"""Every command the agent skill teaches has to work against the real CLI, so an agent that follows the skill
never hits a usage error. Nothing is solved here. Each command goes through click's parser the way it would
before running, and its loads, supports, materials and template values go through Wobbl's own readers."""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from importlib import resources

import click
import pytest

from wobbl.cli.main import cli

SKILL = resources.files("wobbl").joinpath("skill/SKILL.md").read_text(encoding="utf-8")


def _spans(text: str) -> list[str]:
    """Inline code spans outside fenced blocks, with line breaks inside a span folded to spaces."""
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    return [" ".join(s.split()) for s in re.findall(r"`([^`]+)`", prose)]


def skill_commands(text: str = SKILL) -> list[str]:
    """Every `wobbl ...` command in the skill, from inline code and from fenced blocks."""
    cmds = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", " ", block)
        cmds += [ln.strip() for ln in joined.splitlines() if ln.strip().startswith("wobbl ")]
    cmds += [s for s in _spans(text) if s.startswith("wobbl ")]
    return cmds


def dry_parse(args: list[str]) -> tuple[str, click.Context]:
    """Parse a command line exactly as click does before it runs a command, without running it."""
    root = click.Context(cli, info_name="wobbl")
    name, cmd, rest = cli.resolve_command(root, list(args))
    ctx = cmd.make_context(name, list(rest), parent=root)
    while isinstance(cmd, click.Group):   # a group such as 'wobbl flow' parses its subcommand too
        held = ctx._protected_args if hasattr(ctx, "_protected_args") else ctx.protected_args   # click 8.2 or older
        sub, cmd, rest = cmd.resolve_command(ctx, [*held, *ctx.args])
        name = f"{name} {sub}"
        ctx = cmd.make_context(sub, list(rest), parent=ctx)
    return name, ctx


ROOT = Path(__file__).resolve().parents[1]
STAND_IN = {".step": ROOT / "src" / "wobbl" / "validation" / "parts" / "shelf_bracket_fillet.step",
            ".py": ROOT / "examples" / "shelf-bracket" / "bracket.py",
            ".toml": ROOT / "examples" / "rocket-launch" / "rocket.toml",
            ".kicad_sch": ROOT / "tests" / "data" / "circuits" / "ldo_mcu_board.kicad_sch",
            ".kicad_pcb": ROOT / "examples" / "pcb-buck-ringing" / "buck_tight.kicad_pcb"}


def stand_in(thing: str, tmp: Path) -> str:
    """A real file of the same kind for a placeholder the skill names, such as bracket.step or board.cir."""
    p = Path(thing)
    if not thing or p.exists():
        return thing
    if thing.startswith("wobbl-out/"):
        return str(ROOT / "examples" / "shelf-bracket" / "output")
    if p.suffix == ".toml" and "motor" in p.stem:
        return str(ROOT / "examples" / "knsb-motor" / "motor.toml")
    if p.suffix in STAND_IN:
        return str(STAND_IN[p.suffix])
    texts = {".cir": "* divider\nV1 in 0 12\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n",
             ".urdf": '<?xml version="1.0"?>\n<robot name="arm"><link name="base"/></robot>\n',
             ".xml": "<mujoco model='robot'><worldbody/></mujoco>\n"}
    if p.suffix in texts:
        (tmp / p.name).write_text(texts[p.suffix])
        return str(tmp / p.name)
    return thing


def check_simulate(p: dict):
    """Route a simulate command the way the command would, with a real file standing in for each placeholder,
    and check that every option it passes is one an analysis of the plan reads."""
    import tempfile

    from wobbl.cli.simulate import check_options, options_from, split_args
    from wobbl.simulate import route

    thing, question, extra = split_args(p["args"])
    if p["show_list"] or not (thing or question):
        return
    with tempfile.TemporaryDirectory() as tmp:
        plan = route(stand_in(thing, Path(tmp)), question, list(p["physics"]), p["domain"], p["kind"])
    check_options(plan, options_from(p["material"], p["loads"], p["fixes"], p["sets"], extra))
    from wobbl.structural.spec import parse_fixture, parse_load
    for x in p["loads"]:
        parse_load(x)
    for x in p["fixes"]:
        parse_fixture(x)
    if p["material"]:
        from wobbl.materials import resolve_material
        resolve_material(p["material"])


def check_values(name: str, p: dict):
    """Read the option values the way the command would, so a load or a template value that cannot be read
    fails here instead of in front of the agent."""
    from wobbl.materials import resolve_material

    if name == "simulate":
        check_simulate(p)
    elif name in ("run", "ask"):
        from wobbl.cli.main import _study_from

        _study_from(p["part"], "", p)
        if p["material"]:
            resolve_material(p["material"])
    elif name == "thermal":
        from wobbl.thermal.tools import study_from

        resolve_material(p["material"])
        study_from(p["part"], p["material"], p["power"], p["heat_faces"], list(p["films"]), p["ambient"],
                   list(p["fixed"]), p["duration"], p["limit"], p["radiation"])
    elif name == "motion":
        from wobbl.motion.tools import study_from

        if p["spec"]:
            assert p["spec"].endswith(".toml"), p["spec"]
        else:
            study_from(p["name"], p["mass"], p["density"], p["diameter"], p["cd"], p["height"], p["speed"],
                       p["angle"], p["wind"])
    elif name == "mechanism":
        from wobbl.mechanism.templates import is_template, template

        if is_template(p["target"]):
            assert all("=" in s for s in p["sets"]), p["sets"]
            template(p["target"], dict(s.split("=", 1) for s in p["sets"]))
        else:
            assert p["target"].endswith((".toml", ".json", ".xml", ".urdf")), p["target"]
    elif name.startswith("flow "):
        from wobbl.flow.tools import build

        build(name.split()[1], p)
    elif name == "circuit":
        from wobbl.circuits.describe import build
        from wobbl.circuits.templates import TEMPLATES

        assert all("=" in s for s in p["sets"]), p["sets"]
        if p["target"] in TEMPLATES:
            TEMPLATES[p["target"]](dict(s.split("=", 1) for s in p["sets"]))
        elif not p["target"].lower().endswith((".cir", ".sp", ".spice", ".net", ".kicad_sch")):
            build(p["target"], dict(s.split("=", 1) for s in p["sets"]))    # words must build a circuit
    elif name in ("magnetic", "electric", "board", "antenna", "sound"):
        from wobbl.fields.tools import check_params

        assert all("=" in s for s in p["sets"]), p["sets"]
        check_params(name, p["template"], dict(s.split("=", 1) for s in p["sets"]))


def test_the_skill_teaches_simulate_for_every_kind_of_thing():
    """The skill teaches one command for all physics, plus the general verbs, and no hidden alias."""
    import tempfile

    from wobbl.cli.main import VISIBLE
    from wobbl.cli.simulate import split_args
    from wobbl.simulate import route

    parsed = [dry_parse(shlex.split(c)[1:]) for c in skill_commands()]
    names = {n.split()[0] for n, _ in parsed}
    assert {"simulate", "inspect", "fix", "uncertainty", "credibility", "test-plan"} <= names, names
    assert names <= set(VISIBLE), names - set(VISIBLE)
    domains = set()
    with tempfile.TemporaryDirectory() as tmp:
        for n, ctx in parsed:
            if n == "simulate" and ctx.params["args"]:
                thing, question, _ = split_args(ctx.params["args"])
                domains.add(route(stand_in(thing, Path(tmp)), question, list(ctx.params["physics"]),
                                  ctx.params["domain"], ctx.params["kind"]).domain)
    assert domains >= {"part", "circuit", "mechanism", "flight", "fluid", "field", "chemistry"}, domains


@pytest.mark.parametrize("command", skill_commands())
def test_skill_command_parses(command):
    args = shlex.split(command)
    assert args[0] == "wobbl"
    try:
        name, ctx = dry_parse(args[1:])
    except click.UsageError as e:
        pytest.fail(f"'{command}' does not parse: {e.format_message()}")
    check_values(name, ctx.params)


def test_flags_named_alone_exist():
    """Flags the skill names on their own, like `--quality fine`, belong to a command and take the value shown."""
    bases = {"run": ["run", "part.step"], "thermal": ["thermal", "part.step", "-m", "6061-t6"],
             "circuit": ["circuit", "led"], "motion": ["motion", "--diameter", "9 cm", "--mass", "1 kg"],
             "crack": ["crack", "part.step", "-m", "7075-t6", "-l", "1 kN +x @ x=max", "-f", "x=min", "-c",
                       "through 1 mm"]}
    from wobbl.simulate import analyses

    spans = [s for s in _spans(SKILL) if s.startswith("--")]
    assert spans
    declared = {f"--{o.name}" for a in analyses() for o in a.all_options()}
    simulate_flags = {o for p in cli.commands["simulate"].params for o in p.opts}
    for s in spans:
        toks = shlex.split(s)
        if toks[0] in declared or toks[0] in simulate_flags:
            continue
        owner = next((n for n in bases if any(toks[0] in p.opts for p in cli.commands[n].params)), None)
        assert owner, f"no command or physics has {toks[0]}"
        if len(toks) > 1:
            dry_parse([*bases[owner], *toks])


def test_skill_loads_and_supports_read():
    from wobbl.structural.spec import parse_fixture, parse_load

    loads = re.search(r"- loads:(.*?)- supports:", SKILL, re.S).group(1)
    supports = re.search(r"- supports:(.*?)- selectors:", SKILL, re.S).group(1)
    load_texts = re.findall(r'`"([^"`]+)"`', loads)
    support_texts = re.findall(r'`"([^"`]+)"`', supports)
    assert len(load_texts) >= 5 and len(support_texts) >= 5
    kinds = {t: parse_load(t).kind for t in load_texts}
    assert kinds["300 N -x bearing @ hole d=6.6"] == "bolt"
    for t in support_texts:
        parse_fixture(t)


@pytest.mark.parametrize("broken", [
    'wobbl circuit led --supply 5V --resistor 150 --led red',
    'wobbl thermal part.step --power "12 W" --film "5 W/m2K @ all" --ambient 25',
])
def test_the_old_skill_commands_are_caught(broken):
    with pytest.raises(click.UsageError):
        dry_parse(shlex.split(broken)[1:])
