---
title: "Claude Code"
description: "How wobbl setup registers the Wobbl MCP server and skill with Claude Code, and how to do it by hand."
lead: "Claude Code gets the MCP server at user scope and the skill in its skills folder, so Wobbl is there in every project."
order: 1
---

```sh
wobbl setup --no-solvers --agents claude
```

## What setup does

When the `claude` command is on your PATH, setup runs it to register the server at user scope,
removing any older Wobbl entry first.

```sh
claude mcp add --scope user wobbl -- /full/path/to/wobbl mcp
```

When the command is missing, setup writes the same server into `~/.claude.json` under
`mcpServers.wobbl` as a stdio server instead. Either way, the skill goes to
`~/.claude/skills/wobbl/SKILL.md`. Restart Claude Code and ask it about a part.

## By hand

If you would rather not run setup, this one line registers the server by hand.

```sh
claude mcp add --scope user wobbl -- "$(command -v wobbl)" mcp
```

The skill ships inside the package at `wobbl/skill/SKILL.md`, and setup is still the easiest
way to put it in place, since it copies the file and records it for `wobbl uninstall`.

## Long studies

Strength studies take one to seven minutes. If Claude Code stops waiting before a study
finishes, raise its MCP tool timeout, which is in milliseconds, when you start it.

```sh
MCP_TOOL_TIMEOUT=900000 claude
```

Wobbl keeps itself bounded as well. Every study has a time budget, 300 seconds by default, and
when it runs out the last finished mesh level comes back with its checks, so the agent always
gets an answer with a confidence that matches it.

## Check and remove

`wobbl doctor` shows `agent claude` as ok when both the server and the skill are in place.
`wobbl uninstall --agents claude` runs `claude mcp remove --scope user wobbl`, removes the
entry from `~/.claude.json` and deletes the skill folder.
