import json

from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.mcp import tools


def test_help_and_version():
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    for cmd in ("run", "inspect", "check", "report", "fix", "setup", "doctor", "thermal", "circuit", "mcp"):
        assert cmd in r.output
    assert CliRunner().invoke(cli, ["--version"]).output.strip().endswith("0.1.0")


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
              "solve_circuit", "apply_load", "set_material", "mesh"):
        assert n in names


def test_check_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["check"])
    assert r.exit_code != 0 and "no Wobbl results" in r.output
