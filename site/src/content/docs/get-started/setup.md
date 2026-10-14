---
title: "Setup"
description: "What wobbl setup installs, where it puts it and how it connects your coding agents."
lead: "One command installs CalculiX and ngspice into your home folder and connects every coding agent it finds."
order: 2
---

```sh
wobbl setup
wobbl doctor
```

Setup needs no admin rights and touches nothing outside your home folder. It does two jobs,
and you can run either one alone.

## The solvers

Wobbl downloads a pinned micromamba (version 2.9.0-0) into `~/.wobbl/bin`, checks it against
its published SHA-256 when that file is available, and uses it to create a private
conda-forge environment in `~/.wobbl/env` with CalculiX 2.21 or newer and ngspice 37 or
newer. On Apple Silicon ngspice goes into a second environment, `~/.wobbl/env-x86`, because
conda-forge only builds it for Intel Macs and it runs under Rosetta 2. On Linux arm64 there
is no ngspice build at all, so setup skips it and tells you to install it with your package
manager, as [Platforms](/get-started/platforms/) explains.

When the install finishes, Wobbl runs each solver on a tiny problem, a single tetrahedron in
CalculiX and a 1 kilohm resistor in ngspice, so a broken install shows up now rather than in
your first study. The full log goes to `~/.wobbl/setup.log` and the result to
`~/.wobbl/config.json`.

Setup then moves aside what no solver run needs, such as headers, static libraries, documentation and
every program no solver calls, runs each solver on its tiny problem again, and puts the files back
if one stopped working. It also deletes the packages it downloaded. The whole thing takes about 70 MB,
and setup prints what the solvers take. Set `WOBBL_KEEP_CACHE=1` to keep the downloads, for a CI
cache for example.

## The agents

Setup looks for Claude Code, Codex, Cursor and Gemini CLI, counting an agent as present when
its command is on your path or its folder (`~/.claude`, `~/.codex`, `~/.cursor` or
`~/.gemini`) exists. For each one it registers the MCP server and installs the skill.

| Agent | MCP server | Skill or context |
|---|---|---|
| Claude Code | `claude mcp add --scope user wobbl`, or `~/.claude.json` when the claude command is missing | `~/.claude/skills/wobbl/SKILL.md` |
| Codex | `[mcp_servers.wobbl]` in `~/.codex/config.toml` | `~/.codex/skills/wobbl/SKILL.md` |
| Cursor | `mcpServers.wobbl` in `~/.cursor/mcp.json` | `~/.cursor/skills/wobbl/SKILL.md` |
| Gemini CLI | an extension in `~/.gemini/extensions/wobbl/` | `WOBBL.md` in the same extension |

The server command is the full path of your `wobbl` executable, so an agent finds it even
when its own PATH is different from your shell's. Every file setup changes is copied to
`~/.wobbl/backups/<time>/` first, and what it added is recorded in `~/.wobbl/agents.json` so
that `wobbl uninstall` removes exactly that and nothing else. Restart an agent after setup so
it picks up the new server. The [agent guides](/guides/agents/) show what each one gets.

## Options

| Option | What it does |
|---|---|
| `--agents claude,cursor` | register only these agents, or pass `all`, `detected` (the default) or `none` |
| `--no-agents` | install the solvers and leave the agents alone |
| `--no-solvers` | register the agents and skip the solvers |
| `--only calculix` | install only the solvers you name |
| `--project` | also write a Cursor rule to `.cursor/rules/wobbl.mdc` in the current folder |
| `--force` | download micromamba again and rebuild the solver environments from scratch |
| `--dry-run` | print what would change without changing anything |

An agent you install later is one command away.

```sh
wobbl setup --no-solvers --agents codex
```

## Your own solvers or another folder

Wobbl looks for a solver in three places, in this order. First an environment variable,
`WOBBL_CCX` for CalculiX or `WOBBL_NGSPICE` for ngspice, pointing at the binary. Then its own
environments under `~/.wobbl`. Then your PATH. So a CalculiX you built yourself works as soon
as you point `WOBBL_CCX` at it. To keep everything somewhere other than `~/.wobbl`, set
`WOBBL_HOME` before running setup and keep it set when you run Wobbl.

## Check it

`wobbl doctor` lists every piece with `ok`, `missing`, `off` or `planned` next to it, which
[wobbl doctor](/guides/cli/doctor/) explains line by line. If a solver shows as missing,
running `wobbl setup` again usually fixes it, and `wobbl setup --force` starts over. Setup
itself ends by pointing you at `wobbl doctor` and then at the LED question that
[First run](/get-started/first-run/) begins with, which needs no files.
