---
title: "wobbl uninstall"
description: "Remove the MCP registrations and skills Wobbl added to your coding agents, and optionally the whole ~/.wobbl folder."
lead: "uninstall removes exactly what setup added to your agents, and with --all the solvers too."
order: 13
---

```sh
wobbl uninstall [--agents LIST] [--all] [--yes]
```

```sh
wobbl uninstall                    # every agent registration, after a prompt
wobbl uninstall --agents cursor    # just one agent
wobbl uninstall --all --yes        # everything, including ~/.wobbl, with no prompt
```

## Options

| Option | What it does |
|---|---|
| `--agents` | which registrations to remove, a comma list or `all` (the default) |
| `--all` | also deletes `~/.wobbl`, with the solvers, the backups and the results of agent runs |
| `--yes` | skips the confirmation prompt |

Setup records what it installed in `~/.wobbl/agents.json`, and uninstall removes the MCP entry
and the skill for each agent you name without touching the rest of their configuration. It
leaves the `wobbl` command itself in place, which `pipx uninstall wobbl` or
`uv tool uninstall wobbl` removes.
