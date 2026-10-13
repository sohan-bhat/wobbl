---
title: "Gemini CLI"
description: "How wobbl setup installs Wobbl into Gemini CLI as an extension with the MCP server and a context file."
lead: "Gemini CLI gets one extension that carries both the MCP server and the skill as its context file."
order: 4
---

```sh
wobbl setup --no-solvers --agents gemini
```

## What setup writes

Setup creates the folder `~/.gemini/extensions/wobbl/` with two files. The first is
`gemini-extension.json`, which declares the server with a 900,000 millisecond timeout so a full
strength study can finish.

```json
{
  "name": "wobbl",
  "version": "0.1.0",
  "description": "Verified engineering simulation",
  "mcpServers": {
    "wobbl": {
      "command": "/full/path/to/wobbl",
      "args": ["mcp"],
      "timeout": 900000
    }
  },
  "contextFileName": "WOBBL.md"
}
```

The second is `WOBBL.md`, the skill's text without its header, which Gemini CLI loads as
context. Restart Gemini CLI to pick the extension up.

## Check and remove

`wobbl doctor` shows `agent gemini` as ok when both files are there, and
`wobbl uninstall --agents gemini` deletes the extension folder.
