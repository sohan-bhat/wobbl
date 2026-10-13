---
title: "Codex"
description: "How wobbl setup adds the Wobbl MCP server to Codex's config.toml with a long tool timeout, and installs the skill."
lead: "Codex gets a marked block in its config file, with a tool timeout long enough for a full strength study, and the skill."
order: 2
---

```sh
wobbl setup --no-solvers --agents codex
```

## What setup writes

Setup appends this block to `~/.codex/config.toml`, between two marker comments so that it can
find and replace it later without touching the rest of the file.

```toml
# >>> wobbl >>>
[mcp_servers.wobbl]
command = "/full/path/to/wobbl"
args = ["mcp"]
startup_timeout_sec = 30
tool_timeout_sec = 900
# <<< wobbl <<<
```

The 900 second tool timeout covers the longest strength studies, which take about seven
minutes on a 2018 laptop. Running setup again replaces the block instead of adding a second
one, and the old file is backed up to `~/.wobbl/backups/` first. The skill goes to
`~/.codex/skills/wobbl/SKILL.md`.

## By hand

To register it yourself, add the same block to `~/.codex/config.toml` with the output of
`command -v wobbl` as the command, then restart Codex.

## Check and remove

`wobbl doctor` shows `agent codex` as ok when the block and the skill are both there.
`wobbl uninstall --agents codex` strips the block between the markers and deletes the skill
folder, leaving the rest of your config as it was.
