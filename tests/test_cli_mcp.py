import json
import math
import re
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.mcp import tools

ROOT = Path(__file__).resolve().parents[1]


def test_help_and_version():
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    listed = re.findall(r"^  (\S+)\s", r.output.split("Commands:")[1], re.M)
    assert listed[0] == "simulate"
    assert set(listed) == {"simulate", "view", "inspect", "check", "report", "fix", "uncertainty", "credibility",
                           "test-plan", "estimate", "jobs", "materials", "validate", "setup", "doctor", "mcp"}
    for alias in ("run", "thermal", "circuit", "motion", "mechanism", "flow", "buckling", "plastic", "magnetic"):
        assert alias not in listed and cli.commands[alias].hidden
        assert CliRunner().invoke(cli, [alias, "--help"]).exit_code == 0, alias
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert CliRunner().invoke(cli, ["--version"]).output.strip().endswith(version)


def test_materials_command():
    r = CliRunner().invoke(cli, ["materials"])
    assert r.exit_code == 0 and "6061-t6" in r.output and "pla" in r.output


def test_inspect_json(parts):
    r = CliRunner().invoke(cli, ["inspect", parts["mount"], "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert len(d["holes"]) == 4
    assert d["holes"][0]["diameter"] == 4.5


def test_mcp_tools_inspect_and_study(parts, tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    d = tools.inspect_geometry(parts["plate_hole"])
    assert d["units"] == "mm" and len(d["holes"]) == 1
    s = tools.new_study(parts["plate_hole"], "pla")
    sid = s["study_id"]
    assert tools.apply_load(sid, "100 N +x @ x=max")["loads"] == ["100 N +x @ x=max"]
    assert tools.add_fixture(sid, "x=min")["fixtures"] == ["x=min"]
    assert "error" in tools.apply_load(sid, "100 @ top")
    assert "6061" in tools.set_material(sid, "6061-t6")["material"]


def test_mcp_server_lists_tools():
    import asyncio

    from wobbl.mcp.server import build_server

    srv = build_server()
    names = {t.name for t in asyncio.run(srv.list_tools())}
    for n in ("inspect_geometry", "solve_structural", "verify", "render", "suggest_fix", "solve_thermal",
              "solve_circuit", "solve_motion", "apply_load", "set_material", "mesh"):
        assert n in names


def test_check_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["check"])
    assert r.exit_code != 0 and "no Wobbl results" in r.output


def _result_copy(folder: str, tmp_path: Path) -> Path:
    d = tmp_path / Path(folder).name
    d.mkdir()
    shutil.copy(ROOT / folder / "outcome.json", d / "outcome.json")
    return d


@pytest.mark.parametrize("folder,domain,tool", [("examples/battery-pack/output", "thermal", "solve_thermal"),
                                                ("examples/circuits/led", "circuit", "solve_circuit")])
def test_suggest_fix_handles_heat_and_circuit_results(folder, domain, tool, tmp_path):
    d = _result_copy(folder, tmp_path)
    r = tools.suggest_fix(str(d))
    assert "error" not in r, r
    assert r["domain"] == domain and r["verdict"] and r["fixes"] and tool in r["note"]
    assert all(not f["auto"] and f["title"] for f in r["fixes"])
    r = tools.suggest_fix(str(d), apply=True)
    assert "error" not in r, r
    assert r["applied"] is False and "only for strength results" in r["note"] and tool in r["note"]


def test_suggest_fix_strength_and_edge_cases(tmp_path):
    d = _result_copy("examples/shelf-bracket/output", tmp_path)
    r = tools.suggest_fix(str(d))
    assert r["domain"] == "strength" and r["verdict"] == "fails" and r["safety_factor"] < 1 and r["fixes"]
    # applying needs the study and geometry files next to the outcome; a missing one is an error, not a crash
    assert "error" in tools.suggest_fix(str(d), apply=True)
    assert "error" in tools.suggest_fix(str(tmp_path / "nothing"))
    calm = tmp_path / "calm"
    calm.mkdir()
    (calm / "outcome.json").write_text(json.dumps({"domain": "circuit", "verdict": "ok", "fixes": []}))
    r = tools.suggest_fix(str(calm), apply=True)
    assert r["fixes"] == [] and "no fixes" in r["note"] and r["applied"] is False


def test_solve_structural_time_matches_the_real_runs():
    """The tool tells agents how long to wait, so its upper bound has to cover the slowest strength study in the
    published validation results, rounded up to the minute."""
    import asyncio

    from wobbl.mcp.server import build_server

    desc = next(t.description for t in asyncio.run(build_server().list_tools()) if t.name == "solve_structural")
    m = re.search(r"one to (\w+) minutes", desc)
    assert m, desc
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    upper = words.index(m.group(1)) + 1
    rows = json.loads((ROOT / "docs" / "validation-results.json").read_text())
    slowest = max(r["wall_s"] for r in rows if r["domain"] == "strength")
    assert upper == math.ceil(slowest / 60), (upper, slowest)


def test_solve_motion_takes_a_spec_as_text_and_its_fixes_point_back_to_it(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    spec = (ROOT / "examples" / "rocket-launch" / "rocket.toml").read_text().replace('cg = "1.36 m"', 'cg = "1.50 m"')
    r = tools_motion(spec=spec)
    assert "error" not in r, r
    assert r["verdict"] == "fails" and r["values"]["margin_cal"] < 1 and r["events"][0]["name"] == "liftoff"
    assert Path(r["trajectory"]).exists() and {c["name"] for c in r["checks"]} >= {"delta v budget", "stability margin"}
    fix = tools.suggest_fix(r["result_dir"])
    assert fix["domain"] == "motion" and "solve_motion" in fix["note"]
    assert {f["kind"] for f in fix["fixes"]} >= {"nose weight", "fin area"}
    rock = tools_motion(name="rock", diameter="9 cm", density="2700 kg/m3", cd=0.6, height="300 m")
    assert rock["verdict"] == "ok" and rock["confidence"] == "high"
    assert "error" in tools_motion(diameter="9 cm", height="300 m")


def tools_motion(**kw):
    from wobbl.motion.tools import solve_motion_tool

    return solve_motion_tool(**kw)
