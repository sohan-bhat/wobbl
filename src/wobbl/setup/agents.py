"""Register the Wobbl MCP server and install the agent skill for Claude Code, Codex, Cursor and Gemini CLI.

Every file Wobbl changes is backed up first under ~/.wobbl/backups/<time>/, and
what was installed is recorded in ~/.wobbl/agents.json so 'wobbl uninstall'
removes exactly that and nothing else.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

from .. import __version__
from ..solvers.locate import wobbl_home

AGENTS = ("claude", "codex", "cursor", "gemini")
BEGIN, END = "# >>> wobbl >>>", "# <<< wobbl <<<"


def home() -> Path:
    return Path(os.environ.get("WOBBL_AGENT_HOME") or Path.home())


def skill_text() -> str:
    return resources.files("wobbl").joinpath("skill/SKILL.md").read_text(encoding="utf-8")


def server_command() -> tuple[str, list[str]]:
    """How an agent should start the MCP server: the wobbl executable if it is on disk, else python -m."""
    exe = shutil.which("wobbl")
    argv0 = Path(sys.argv[0])
    if exe:
        return str(Path(exe).resolve()), ["mcp"]
    if argv0.name.startswith("wobbl") and argv0.exists():
        return str(argv0.resolve()), ["mcp"]
    return sys.executable, ["-m", "wobbl", "mcp"]


def detect() -> dict[str, bool]:
    h = home()
    return {
        "claude": bool(shutil.which("claude")) or (h / ".claude").exists(),
        "codex": bool(shutil.which("codex")) or (h / ".codex").exists(),
        "cursor": bool(shutil.which("cursor")) or (h / ".cursor").exists(),
        "gemini": bool(shutil.which("gemini")) or (h / ".gemini").exists(),
    }


class Registrar:
    def __init__(self, log=print, dry_run: bool = False):
        self.log = log
        self.dry = dry_run
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.manifest_path = wobbl_home() / "agents.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}

    # -- helpers ------------------------------------------------------------
    def backup(self, path: Path):
        if path.exists():
            dest = wobbl_home() / "backups" / self.stamp / path.relative_to(home()) if \
                str(path).startswith(str(home())) else wobbl_home() / "backups" / self.stamp / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    def write(self, path: Path, text: str):
        if self.dry:
            self.log(f"would write {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backup(path)
        path.write_text(text, encoding="utf-8")

    def record(self, agent: str, **items):
        self.manifest.setdefault(agent, {}).update(items)

    def save(self):
        if not self.dry:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(json.dumps(self.manifest, indent=1))

    def _json_server(self, path: Path, key: str = "mcpServers", entry: dict | None = None):
        cmd, args = server_command()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                raise RuntimeError(f"{path} is not valid JSON; fix it or remove it, then run 'wobbl setup' again")
        data.setdefault(key, {})["wobbl"] = entry or {"command": cmd, "args": args}
        self.write(path, json.dumps(data, indent=2) + "\n")

    def _json_remove(self, path: Path, key: str = "mcpServers"):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError:
            return
        if data.get(key, {}).pop("wobbl", None) is not None:
            self.write(path, json.dumps(data, indent=2) + "\n")

    # -- agents -------------------------------------------------------------
    def claude(self):
        h = home()
        skill = h / ".claude" / "skills" / "wobbl" / "SKILL.md"
        self.write(skill, skill_text())
        cmd, args = server_command()
        claude = shutil.which("claude")
        how = "file"
        if claude and not os.environ.get("WOBBL_AGENT_HOME") and not self.dry:
            subprocess.run([claude, "mcp", "remove", "--scope", "user", "wobbl"], capture_output=True, text=True)
            r = subprocess.run([claude, "mcp", "add", "--scope", "user", "wobbl", "--", cmd, *args],
                               capture_output=True, text=True)
            if r.returncode == 0:
                how = "claude cli"
        if how == "file":
            self._json_server(h / ".claude.json", entry={"type": "stdio", "command": cmd, "args": args, "env": {}})
        self.record("claude", skill=str(skill), mcp=how)
        self.log(f"Claude Code: MCP server registered ({how}), skill at {skill}")

    def codex(self):
        h = home()
        cfg = h / ".codex" / "config.toml"
        cmd, args = server_command()
        text = cfg.read_text() if cfg.exists() else ""
        text = _strip_block(text)
        block = (f"\n{BEGIN}\n[mcp_servers.wobbl]\ncommand = {json.dumps(cmd)}\nargs = {json.dumps(args)}\n"
                 f"startup_timeout_sec = 30\ntool_timeout_sec = 900\n{END}\n")
        self.write(cfg, text.rstrip() + ("\n" if text.strip() else "") + block)
        skill = h / ".codex" / "skills" / "wobbl" / "SKILL.md"
        self.write(skill, skill_text())
        self.record("codex", config=str(cfg), skill=str(skill))
        self.log(f"Codex: MCP server in {cfg}, skill at {skill}")

    def cursor(self):
        h = home()
        cfg = h / ".cursor" / "mcp.json"
        self._json_server(cfg)
        skill = h / ".cursor" / "skills" / "wobbl" / "SKILL.md"
        self.write(skill, skill_text())
        self.record("cursor", config=str(cfg), skill=str(skill))
        self.log(f"Cursor: MCP server in {cfg}, skill at {skill}")

    def gemini(self):
        h = home()
        ext = h / ".gemini" / "extensions" / "wobbl"
        cmd, args = server_command()
        manifest = {"name": "wobbl", "version": __version__, "description": "Verified engineering simulation",
                    "mcpServers": {"wobbl": {"command": cmd, "args": args, "timeout": 900000}},
                    "contextFileName": "WOBBL.md"}
        self.write(ext / "gemini-extension.json", json.dumps(manifest, indent=2) + "\n")
        body = re.sub(r"^---.*?---\s*", "", skill_text(), flags=re.S)
        self.write(ext / "WOBBL.md", body)
        self.record("gemini", extension=str(ext))
        self.log(f"Gemini CLI: extension with MCP server and context at {ext}")

    def project_rules(self, project: Path):
        """Cursor project rule, for teams that keep agent rules in the repo."""
        rule = project / ".cursor" / "rules" / "wobbl.mdc"
        body = re.sub(r"^---.*?---\s*", "", skill_text(), flags=re.S)
        self.write(rule, "---\ndescription: Use Wobbl for engineering simulation\nalwaysApply: false\n---\n" + body)
        self.log(f"Cursor project rule at {rule}")

    # -- removal ------------------------------------------------------------
    def uninstall(self, agents=AGENTS):
        h = home()
        for a in agents:
            info = self.manifest.get(a, {})
            if a == "claude":
                claude = shutil.which("claude")
                if info.get("mcp") == "claude cli" and claude:
                    subprocess.run([claude, "mcp", "remove", "--scope", "user", "wobbl"], capture_output=True)
                self._json_remove(h / ".claude.json")
                _rmtree(h / ".claude" / "skills" / "wobbl")
            elif a == "codex":
                cfg = h / ".codex" / "config.toml"
                if cfg.exists():
                    t = cfg.read_text()
                    if BEGIN in t:
                        self.write(cfg, _strip_block(t))
                _rmtree(h / ".codex" / "skills" / "wobbl")
            elif a == "cursor":
                self._json_remove(h / ".cursor" / "mcp.json")
                _rmtree(h / ".cursor" / "skills" / "wobbl")
            elif a == "gemini":
                _rmtree(h / ".gemini" / "extensions" / "wobbl")
            self.manifest.pop(a, None)
            self.log(f"removed Wobbl from {a}")
        self.save()


def _strip_block(text: str) -> str:
    return re.sub(rf"\n?{re.escape(BEGIN)}.*?{re.escape(END)}\n?", "\n", text, flags=re.S).rstrip() + \
        ("\n" if text.strip() else "")


def _rmtree(p: Path):
    if p.exists():
        shutil.rmtree(p)


def register(agents, log=print, dry_run=False) -> Registrar:
    r = Registrar(log, dry_run)
    for a in agents:
        getattr(r, a)()
    r.save()
    return r


def status() -> dict[str, dict]:
    h = home()
    out = {}
    cl = h / ".claude.json"
    claude_mcp = False
    if cl.exists():
        try:
            claude_mcp = "wobbl" in json.loads(cl.read_text()).get("mcpServers", {})
        except Exception:  # noqa: BLE001
            pass
    out["claude"] = {"skill": (h / ".claude/skills/wobbl/SKILL.md").exists(), "mcp": claude_mcp}
    cx = h / ".codex/config.toml"
    out["codex"] = {"skill": (h / ".codex/skills/wobbl/SKILL.md").exists(),
                    "mcp": cx.exists() and "[mcp_servers.wobbl]" in cx.read_text()}
    cu = h / ".cursor/mcp.json"
    cu_mcp = False
    if cu.exists():
        try:
            cu_mcp = "wobbl" in json.loads(cu.read_text()).get("mcpServers", {})
        except Exception:  # noqa: BLE001
            pass
    out["cursor"] = {"skill": (h / ".cursor/skills/wobbl/SKILL.md").exists(), "mcp": cu_mcp}
    ge = h / ".gemini/extensions/wobbl/gemini-extension.json"
    out["gemini"] = {"skill": (h / ".gemini/extensions/wobbl/WOBBL.md").exists(), "mcp": ge.exists()}
    return out
