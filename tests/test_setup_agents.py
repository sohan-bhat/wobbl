import json

import pytest

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
