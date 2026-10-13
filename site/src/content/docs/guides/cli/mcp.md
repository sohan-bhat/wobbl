---
title: "wobbl mcp"
description: "Run the Wobbl MCP server on stdio, which coding agents start for themselves after wobbl setup registers it."
lead: "mcp runs the MCP server on standard input and output, the way coding agents expect to start it."
order: 14
---

```sh
wobbl mcp
```

You rarely run this yourself, since your agent starts it once `wobbl setup` has registered it.
To register it with an MCP client that setup does not know, give the client the full path of
the `wobbl` executable with the single argument `mcp`, and a tool timeout of at least 900
seconds so that a long strength study can finish.

The server moves anything that native libraries print to stderr, so the protocol stream on
stdout stays clean, and it renders heatmaps off screen. It offers 13 tools, which
[MCP tools](/guides/agents/mcp-tools/) lists with their parameters and results.
