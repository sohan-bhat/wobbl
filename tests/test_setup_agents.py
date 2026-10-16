import json
import shlex

import click
import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.setup import agents, installer


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    h = tmp_path / "home"
    w = tmp_path / "wobbl"
    h.mkdir()
    monkeypatch.setenv("WOBBL_AGENT_HOME", str(h))
    monkeypatch.setenv("WOBBL_HOME", str(w))
    return h, w


def test_register_and_uninstall_all_agents(homes):
    h, w = homes
    (h / ".codex").mkdir()
    (h / ".codex" / "config.toml").write_text('model = "o4"\n')
    (h / ".cursor").mkdir()
    (h / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    agents.register(list(agents.AGENTS), log=lambda m: None)
    st = agents.status()
    assert all(s["mcp"] and s["skill"] for s in st.values()), st
    cfg = (h / ".codex" / "config.toml").read_text()
    assert 'model = "o4"' in cfg and "[mcp_servers.wobbl]" in cfg
    cur = json.loads((h / ".cursor" / "mcp.json").read_text())
    assert "other" in cur["mcpServers"] and "wobbl" in cur["mcpServers"]
    ext = json.loads((h / ".gemini" / "extensions" / "wobbl" / "gemini-extension.json").read_text())
    assert ext["mcpServers"]["wobbl"]["args"][-1] == "mcp"
    assert "Never skip" in (h / ".claude" / "skills" / "wobbl" / "SKILL.md").read_text()
    # backups of the files that existed before
    backups = list((w / "backups").rglob("config.toml")) + list((w / "backups").rglob("mcp.json"))
    assert len(backups) >= 2
    agents.Registrar(lambda m: None).uninstall(list(agents.AGENTS))
    st = agents.status()
    assert not any(s["mcp"] or s["skill"] for s in st.values()), st
    assert (h / ".codex" / "config.toml").read_text().strip() == 'model = "o4"'
    assert "other" in json.loads((h / ".cursor" / "mcp.json").read_text())["mcpServers"]


def test_register_twice_is_idempotent(homes):
    h, _ = homes
    agents.register(["codex"], log=lambda m: None)
    agents.register(["codex"], log=lambda m: None)
    assert (h / ".codex" / "config.toml").read_text().count("[mcp_servers.wobbl]") == 1


def test_install_plan_covers_every_platform():
    for plat in ("osx-64", "osx-arm64", "linux-64", "linux-aarch64", "win-64"):
        p = installer.plan(plat)
        assert any(s.startswith("calculix") for s in p["env"])
        has_spice = any(s.startswith("ngspice") for s in p["env"] + p["env_x86"])
        assert has_spice or "ngspice" in p["unavailable"]
    assert installer.plan("osx-arm64")["env_x86"]
    assert "ngspice" in installer.plan("linux-aarch64")["unavailable"]


def test_setup_ends_with_real_next_steps(homes):
    """The last line of setup names commands that exist. It used to suggest 'wobbl examples', which does not."""
    r = CliRunner().invoke(cli, ["setup", "--no-solvers", "--agents", "none"])
    assert r.exit_code == 0, r.output
    text = " ".join(r.output.split())
    assert "wobbl examples" not in text
    assert text.startswith("NEXT wobbl doctor checks every piece")
    from wobbl.cli.admin import NEXT_STEPS
    suggested = [cmd for cmd, _ in NEXT_STEPS]
    assert len(suggested) == 3, suggested
    for cmd in suggested:
        assert cmd in r.output                          # each on one line, never wrapped
        args = shlex.split(cmd)[1:]
        root = click.Context(cli, info_name="wobbl")
        name, sub, rest = cli.resolve_command(root, args)
        sub.make_context(name, list(rest), parent=root)   # parses without running
    assert any("LED" in s for s in suggested)           # the no file question from the first run guide
