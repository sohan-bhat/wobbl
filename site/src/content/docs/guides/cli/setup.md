---
title: "wobbl setup"
description: "Install CalculiX and ngspice into ~/.wobbl and register the Wobbl MCP server and skill with your coding agents."
lead: "setup installs the solvers into your home folder and connects every coding agent it finds."
order: 11
---

```sh
wobbl setup [options]
```

```sh
wobbl setup                                  # solvers and every agent it finds
wobbl setup --no-agents                      # solvers only
wobbl setup --no-solvers --agents codex      # connect one agent later
wobbl setup --dry-run                        # show what would change
```

## Options

| Option | What it does |
|---|---|
| `--agents` | a comma list of `claude`, `codex`, `cursor` and `gemini`, or `all`, `detected` (the default) or `none` |
| `--no-agents` | installs the solvers only |
| `--no-solvers` | registers the agents only |
| `--only` | installs only the named solvers, such as `calculix` |
| `--project` | also writes a Cursor rule into `.cursor/rules/wobbl.mdc` in the current folder |
| `--force` | downloads micromamba again and rebuilds the solver environments from scratch |
| `--dry-run` | prints what would change without changing anything |

When setup finds no agents, it says so and suggests `--agents all` to register every one
anyway. Every file it changes is backed up to `~/.wobbl/backups/<time>/` first, and a failed
install points to its log at `~/.wobbl/setup.log`. [Setup](/get-started/setup/) explains what
gets installed where, and the [agent guides](/guides/agents/) show what each agent receives.
