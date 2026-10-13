# Using Wobbl from coding agents

`wobbl setup` registers Wobbl's MCP server and installs its skill for every
coding agent it finds. Pick agents explicitly with `--agents claude,codex` or
`--agents all`.

| Agent | MCP server | Skill or context |
|---|---|---|
| Claude Code | `claude mcp add --scope user wobbl` (or `~/.claude.json` when the CLI is absent) | `~/.claude/skills/wobbl/SKILL.md` |
| Codex | `[mcp_servers.wobbl]` block in `~/.codex/config.toml` | `~/.codex/skills/wobbl/SKILL.md` |
| Cursor | `mcpServers.wobbl` in `~/.cursor/mcp.json` | `~/.cursor/skills/wobbl/SKILL.md`; `wobbl setup --project` also writes `.cursor/rules/wobbl.mdc` |
| Gemini CLI | extension at `~/.gemini/extensions/wobbl/` | `WOBBL.md` context in the same extension |

Every file Wobbl changes is copied to `~/.wobbl/backups/<time>/` first. What was
installed is recorded in `~/.wobbl/agents.json`; `wobbl uninstall` removes exactly
that and leaves the rest of each config alone. `wobbl doctor` shows the state of
each agent.

The server command is the absolute path of the `wobbl` executable, so agents
find it even when their PATH does not include pipx's bin folder.

## Tools

| Tool | Does |
|---|---|
| `inspect_geometry(path)` | bounding box, holes, fillets, sharp corners, thickness, largest faces, selector examples |
| `mesh(path, size)` | first second order mesh and its size |
| `new_study`, `set_material`, `apply_load`, `add_fixture` | build a study step by step (saved under `~/.wobbl/studies`) |
| `solve_structural(...)` | the verified strength study; returns verdict, governing stress, location, safety factor, deflection, confidence with reasons, checks, assumptions, fixes, image paths, report path |
| `verify(result_dir)` | checks, confidence, excluded artifact peaks |
| `render(result_dir)` | image and report paths |
| `suggest_fix(result_dir, apply)` | fixes with numbers; `apply=true` re-runs them and returns before and after |
| `solve_thermal(...)` | steady or transient heat |
| `solve_circuit(template or netlist, params)` | LED, RC, buck, MOSFET, or any netlist |
| `list_materials()` | the materials table |

Strength studies take one to seven minutes on a 2018 laptop. Wobbl registers Codex with a 900 s
tool timeout and Gemini CLI with 900000 ms; if your Claude Code setup cuts long
tool calls short, raise `MCP_TOOL_TIMEOUT` (milliseconds) in its environment, or
pass `quality="fast"` for a first look.

Results are compact JSON: numbers, statuses, one line summaries and file
paths. Agents open the PNG heatmaps or the HTML report when they need to see
the part.

## The skill

The skill (`src/wobbl/skill/SKILL.md`) tells the agent the workflow (inspect,
choose loads and supports explicitly, solve, read the checks before the number,
look at the heatmap, fix, report) and what never to skip: never quote a low
confidence number as the answer, never quote an excluded peak, check units, make
the supports match the real build, and do not use `--quality fast` for a final
answer.

## Example session

This is the shelf bracket from `examples/shelf-bracket`, driven through the tools:

```
> Will this printed PLA shelf bracket hold its share of an 8 kg shelf?

agent: inspect_geometry("bracket.py")
agent: solve_structural(path="bracket.py", material="pla",
                        fixtures=["bolt holes d=4.5", "wall x=min"],
                        loads=["40 N down @ top and within 12 of 65,12.5,6"])
       -> verdict fails, 52.3 MPa on the 0.5 mm inside fillet, safety factor 0.96, confidence high
agent: suggest_fix(result_dir, apply=true)
       -> 3 mm fillet: 24.1 MPa (2.08); legs at 10 mm: 21 MPa (2.38); gusset: 9.82 MPa (5.09)
agent: "It cracks at the inside corner: 52.3 MPa against PLA's 50 MPa. A 3 mm fillet halves the
        stress; a gusset brings it to 9.82 MPa, safety factor 5.1. The gusset toe is a sharp corner
        Wobbl excluded as a singularity, so round it too. Heatmaps and report: ..."
```
