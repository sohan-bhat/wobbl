"""Every command the agent skill teaches has to work against the real CLI, so an agent that follows the skill
never hits a usage error. Nothing is solved here. Each command goes through click's parser the way it would
before running, and its loads, supports, materials and template values go through Wobbl's own readers."""
from __future__ import annotations

import re
import shlex
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


def check_values(name: str, p: dict):
    """Read the option values the way the command would, so a load or a template value that cannot be read
    fails here instead of in front of the agent."""
    from wobbl.materials import resolve_material

    if name in ("run", "ask"):
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
    elif name == "crack" and p["part"].startswith("wobbl-out/"):
        assert not p["loads"] and not p["fixes"], "a result folder brings its own loads and supports"
    elif name == "crack":
        from wobbl.cli.crack import study_from_options

        study_from_options(p["part"], p["material"], p["loads"], p["fixes"], p["crack"], r_ratio=p["r_ratio"],
                           target_life=p["target_life"])
    elif name.startswith("flow "):
        from wobbl.flow.tools import build

        build(name.split()[1], p)
    elif name == "circuit":
        from wobbl.circuits.templates import TEMPLATES

        if p["target"].lower().endswith((".cir", ".sp", ".spice", ".net")):
            assert p["probes"], "a netlist run needs at least one --probe"
        else:
            assert all("=" in s for s in p["sets"]), p["sets"]
            TEMPLATES[p["target"]](dict(s.split("=", 1) for s in p["sets"]))
    elif name in ("magnetic", "electric", "board", "antenna", "sound"):
        from wobbl.fields.tools import check_params

        assert all("=" in s for s in p["sets"]), p["sets"]
        check_params(name, p["template"], dict(s.split("=", 1) for s in p["sets"]))


def test_the_skill_names_every_kind_of_command():
    names = {dry_parse(shlex.split(c)[1:])[0] for c in skill_commands()}
    assert {"inspect", "run", "materials", "fix", "thermal", "circuit", "motion", "motor", "flame", "runaway",
            "propellant", "mechanism", "crack"} <= names, names


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
    spans = [s for s in _spans(SKILL) if s.startswith("--")]
    assert spans
    for s in spans:
        toks = shlex.split(s)
        owner = next((n for n in bases if any(toks[0] in p.opts for p in cli.commands[n].params)), None)
        assert owner, f"no command has {toks[0]}"
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
