"""The buckling, vibration and fatigue commands and MCP tools, without a solver."""
import asyncio
import json

from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.mcp import tools


def test_commands_are_listed():
    r = CliRunner().invoke(cli, ["simulate", "--list"])
    assert r.exit_code == 0
    for cmd in ("buckling", "vibration", "fatigue"):
        assert cmd in r.output
        assert CliRunner().invoke(cli, [cmd, "--help"]).exit_code == 0   # the old command stays as a hidden alias


def test_mcp_tools_are_listed():
    from wobbl.mcp.server import INSTRUCTIONS, build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert {"solve_buckling", "solve_vibration", "fatigue_life"} <= names   # kept as deprecated aliases
    assert "simulate(thing, question)" in INSTRUCTIONS


def test_fatigue_from_the_command_line(tmp_path):
    r = CliRunner().invoke(cli, ["fatigue", "--material", "6061-t6", "--amplitude", "120 MPa", "--mean", "20 MPa",
                                 "--cycles", "1e6", "--json", "-o", str(tmp_path / "f")])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output[r.output.index("{"):])
    assert d["domain"] == "fatigue" and d["verdict"] in ("marginal", "fails") and d["fixes"]
    assert d["values"]["blocks"][0]["amplitude_MPa"] == 120.0
    # suggest_fix sends an agent back to the fatigue tool with a new input
    fx = tools.suggest_fix(str(tmp_path / "f"))
    assert fx["domain"] == "fatigue" and "fatigue_life" in fx["note"]


def test_fatigue_tool_and_errors(tmp_path, monkeypatch):
    from wobbl.life.tools import fatigue_life_tool, footprint, solve_buckling_tool

    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    d = fatigue_life_tool(material="4140", amplitude="200 MPa", mean="100 MPa", kt=2.0, radius="1 mm",
                          cycles="1e6")
    assert "error" not in d and d["kf"] < 2.0 and d["safety_factor"] > 0 and d["report"].endswith("report.html")
    assert "error" in fatigue_life_tool(material="copper", amplitude="10 MPa")
    assert "error" in solve_buckling_tool(str(tmp_path / "missing.step"), "6061-t6", ["1 N -z @ top"], ["bottom"])
    fp = footprint()
    assert fp["download_bytes"] == 0 and "download nothing" in fp["line"]


def test_the_skills_life_commands_read_their_values(tmp_path):
    """Every buckling, vibration and fatigue run the skill teaches routes to that physics, and its loads, supports,
    masses, speeds, sweeps, PSDs and blocks all read, so an agent that copies one never hits an input error."""
    import shlex

    from test_skill_commands import dry_parse, skill_commands, stand_in
    from wobbl.cli.simulate import options_from, split_args
    from wobbl.life import modal
    from wobbl.life.fatigue import parse_block
    from wobbl.life.vibspec import parse_mass, parse_rpm, parse_sine
    from wobbl.materials import resolve_material
    from wobbl.simulate import route
    from wobbl.structural.spec import parse_fixture, parse_load

    seen = set()
    for cmd in skill_commands():
        name, ctx = dry_parse(shlex.split(cmd)[1:])
        p = ctx.params
        if name != "simulate" or not p["args"]:
            continue
        thing, question, extra = split_args(p["args"])
        plan = route(stand_in(thing, tmp_path), question, list(p["physics"]), p["domain"], p["kind"])
        life = {"buckling", "vibration", "fatigue"} & set(plan.names)
        if not life:
            continue
        seen |= life
        o = options_from(p["material"], p["loads"], p["fixes"], p["sets"], extra)
        if o.get("material"):
            resolve_material(o["material"][-1])
        for f in o.get("fix", ()):
            parse_fixture(f)
        for ld in o.get("load", ()):
            parse_load(ld)
        for m in o.get("mass", ()):
            parse_mass(m)
        for r in o.get("rpm", ()):
            parse_rpm(r, int(o.get("blades", ["2"])[-1]))
        for x in o.get("sine", ()):
            parse_sine(x)
        for x in o.get("psd", ()):
            modal.parse_psd(x)
        for b in o.get("block", ()):
            parse_block(b, 1.0, 1.0)
    assert seen == {"buckling", "vibration", "fatigue"}


def test_commands_refuse_missing_inputs(tmp_path):
    part = tmp_path / "p.step"
    part.write_text("x")
    r = CliRunner().invoke(cli, ["buckling", str(part), "-m", "6061-t6", "--fix", "z=min"])
    assert r.exit_code == 2 and "--load" in r.output
    r = CliRunner().invoke(cli, ["vibration", str(part), "-m", "6061-t6"])
    assert r.exit_code == 2 and "--fix" in r.output
    r = CliRunner().invoke(cli, ["vibration", str(part), "-m", "6061-t6", "--fix", "x=min", "--rpm", "fast"])
    assert r.exit_code == 2 and "speed" in r.output
