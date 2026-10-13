# Get started

Install Wobbl, add the solvers, connect your coding agent and get a first answer.

## Install

Wobbl needs Python 3.10 or newer on macOS, Linux or Windows. Until it is on
PyPI, install it from GitHub with pipx or uv:

```sh pipx
pipx install "git+https://github.com/sohan-bhat/wobbl"
```

```sh uv
uv tool install "git+https://github.com/sohan-bhat/wobbl"
```

## Add the solvers and connect your agents

```sh
wobbl setup
```

Setup downloads a pinned micromamba, installs CalculiX and ngspice from
conda-forge into `~/.wobbl` (about 340 MB, no admin rights), then registers the
MCP server and the skill with every coding agent it finds.

| Platform | CalculiX | ngspice |
|---|---|---|
| macOS Apple Silicon | native | Intel build under Rosetta 2 |
| macOS Intel | native | native |
| Linux x86_64 | native | native |
| Linux arm64 | native | from your package manager; Wobbl finds it |
| Windows x64 | native | native |

`--no-agents` installs the solvers only, `--agents claude,cursor` picks agents,
and `--dry-run` shows what would change without changing anything.

## Check every piece

```sh
wobbl doctor
```

Each solver, the renderer and every agent's registration comes back as `ok`,
`missing` or `off`.

## Your first answer

This one needs no files:

```sh
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

```text
FAILS   The LED fries: 36.9 mA through it, above its 30 mA rating (1.2x).

LED current   rating   resistor power   LED power
36.9 mA       30 mA    63.9 mW          120 mW

confidence high
  passed hand calculation, power balance

HOW TO FIX IT
  Use a 91 ohm resistor
  Gives about 19.8 mA; it dissipates 35.6 mW, so a 0.25 W part is enough.
```

## Your own part

```sh
wobbl inspect part.step
wobbl part.step "will this 6061 bracket yield under 50 lbf at the tip?"
```

`inspect` lists the holes, fillets, sharp corners and face ids. A plain question
works, and Wobbl prints every guess it made. For an answer you rely on, give the
material, supports and loads yourself: [Check a part's strength](strength.md).

## Connect your coding agent

`wobbl setup` already did this for every agent it found. To add one later,
register it by name and restart the agent:

```sh
wobbl setup --no-solvers --agents codex
```

| Agent | MCP server | Skill or context |
|---|---|---|
| Claude Code | `claude mcp add --scope user wobbl`, or `~/.claude.json` when the CLI is absent | `~/.claude/skills/wobbl/SKILL.md` |
| Codex | `[mcp_servers.wobbl]` in `~/.codex/config.toml`, 900 s tool timeout | `~/.codex/skills/wobbl/SKILL.md` |
| Cursor | `mcpServers.wobbl` in `~/.cursor/mcp.json` | `~/.cursor/skills/wobbl/SKILL.md`; `wobbl setup --project` also writes `.cursor/rules/wobbl.mdc` |
| Gemini CLI | extension at `~/.gemini/extensions/wobbl/`, 900000 ms timeout | `WOBBL.md` in the same extension |

Every file Wobbl changes is copied to `~/.wobbl/backups/<time>/` first, and
`wobbl uninstall` removes exactly what it added. The server command is the full
path of the `wobbl` executable, so agents find it even when their PATH does not.

## CLI basics

Every run writes a folder, `wobbl-out/<part>` unless you pass `-o`:

| File | What it holds |
|---|---|
| `summary.txt` | verdict, checks, assumptions and fixes in plain text; what an agent reads |
| `report.html` | a self contained report with the heatmaps |
| `outcome.json` | every number, for scripts |
| `images/` | stress, zoom, safety, deformed shape and convergence plots |
| `result.vtu` | the full field, for ParaView |

```sh
wobbl check          # checks and confidence of the latest run
wobbl report --open  # open the latest report
wobbl materials      # the material table
```

Add `--json` to any run to print the result as JSON. Every command and option
is in the [command reference](cli.md).

## Troubleshooting

**`wobbl doctor` says a solver is missing.** Run `wobbl setup` again; its log is
`~/.wobbl/setup.log`, and `wobbl setup --force` reinstalls from scratch. To use
solvers you built yourself, set `WOBBL_CCX` or `WOBBL_NGSPICE` to the binary.

**ngspice on Apple Silicon.** It is the Intel build and needs Rosetta 2:
`softwareupdate --install-rosetta --agree-to-license`.

**ngspice on Linux arm64.** conda-forge has no build for it. Install ngspice
with your package manager and Wobbl finds it on PATH.

**The agent does not see Wobbl.** Restart the agent, then run `wobbl doctor`.
If setup found no agents, `wobbl setup --agents all` registers every one.

**Setup stops on a config file.** It will not overwrite a file that is not valid
JSON. Fix or remove that file and run `wobbl setup` again.

**The agent gives up on a long study.** Strength studies take one to seven
minutes on a 2018 laptop. For Claude Code, raise `MCP_TOOL_TIMEOUT`
(milliseconds) in its environment, or ask for `quality="fast"` for a first look.

**The part is the wrong size.** Files are read in millimeters. Pass
`--scale 25.4` for a file drawn in inches or `--scale 1000` for meters.

**"free to spin about z".** Wobbl will not solve a part that is not held. Add a
support that stops the motion it names.

## Next

- [Check a part's strength](strength.md)
- [Fix a failing part](fix.md)
- [Use Wobbl from your agent](agents.md)
- [Browse the examples](../examples)
