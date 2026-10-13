---
title: "Cursor"
description: "How wobbl setup adds the Wobbl MCP server and skill to Cursor, and how to add a rule to one project."
lead: "Cursor gets the server in its global MCP file and the skill, and a project can carry its own rule as well."
order: 3
---

```sh
wobbl setup --no-solvers --agents cursor
```

## What setup writes

The server goes into `~/.cursor/mcp.json` under `mcpServers.wobbl`, next to any servers you
already have.

```json
{
  "mcpServers": {
    "wobbl": {
      "command": "/full/path/to/wobbl",
      "args": ["mcp"]
    }
  }
}
```

The skill goes to `~/.cursor/skills/wobbl/SKILL.md`. Setup will not overwrite an `mcp.json`
that is not valid JSON, so if it stops there, fix or remove that file and run it again.

## A rule for one project

Teams that keep agent rules in the repository can add one for Wobbl. Run this from the project
folder.

```sh
wobbl setup --no-solvers --agents cursor --project
```

It writes `.cursor/rules/wobbl.mdc` with the skill's text and `alwaysApply: false`, so Cursor
pulls it in when a conversation is about simulation rather than on every request.

## Check and remove

`wobbl doctor` shows `agent cursor` as ok when the server and the skill are both there.
`wobbl uninstall --agents cursor` removes the `wobbl` entry from `mcp.json` and deletes the
skill folder. A project rule is yours to delete, since it lives in your repository.
