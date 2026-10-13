---
title: "Agents"
description: "Use Wobbl from Claude Code, Codex, Cursor and Gemini CLI through its MCP server and agent skill."
lead: "Ask your coding agent about a part in plain words, and it calls Wobbl's tools, reads the checks before the number and answers with a heatmap and a report."
cards:
  - title: "Claude Code"
    href: "/guides/agents/claude-code/"
    text: "The MCP server registered at user scope and the skill in your Claude folder."
  - title: "Codex"
    href: "/guides/agents/codex/"
    text: "A marked block in config.toml with a 900 second tool timeout, and the skill."
  - title: "Cursor"
    href: "/guides/agents/cursor/"
    text: "The server in mcp.json, the skill, and an optional rule for one project."
  - title: "Gemini CLI"
    href: "/guides/agents/gemini/"
    text: "An extension that carries both the server and the context file."
  - title: "MCP tools"
    href: "/guides/agents/mcp-tools/"
    text: "All 13 tools with their parameters and what each one returns."
  - title: "The skill"
    href: "/guides/agents/skill/"
    text: "The workflow the skill teaches and the rules it never lets an agent skip."
---

`wobbl setup` connects every agent it finds, as [Setup](/get-started/setup/) describes. After
a restart, the agent sees 13 tools and a skill that teaches it the workflow.

## What to ask

Say what the part is made of, how it is held and what loads it. The agent fills in the rest
from `inspect_geometry` and asks you when it cannot tell.

```text
Will this printed PLA shelf bracket hold its share of an 8 kg shelf? It is screwed to the wall through the two holes.
Does this cast motor mount meet our safety factor 2 rule under a 2 kN emergency stop belt pull?
Will a 15 W chip stay under 85 C on this heat sink without a fan?
What is the ripple on this 12 V to 5 V buck at 2 A?
```

## A session

This is the shelf bracket from the examples, driven through the tools.

```text
> Will this printed PLA shelf bracket hold its share of an 8 kg shelf?

agent: inspect_geometry("bracket.py")
agent: solve_structural(path="bracket.py", material="pla",
                        fixtures=["bolt holes d=4.5", "wall x=min"],
                        loads=["40 N down @ top and within 12 of 65,12.5,6"])
       verdict fails, 52.3 MPa on the 0.5 mm inside fillet, safety factor 0.96, confidence high
agent: suggest_fix(result_dir, apply=true)
       3 mm fillet 24.1 MPa (2.08), legs at 10 mm 21 MPa (2.38), gusset 9.82 MPa (5.09)
```

The agent then answers that the bracket cracks at the inside corner, that a gusset brings the
peak down to 9.82 MPa, and that the gusset's own toe needs a small fillet, with the paths to
the heatmaps and the report.

## Long studies

A strength study takes one to seven minutes on a 2018 laptop, which is longer than some agents
wait for a tool by default. Setup registers Codex with a 900 second tool timeout and Gemini CLI
with 900,000 milliseconds. For Claude Code, raise `MCP_TOOL_TIMEOUT`, in milliseconds, if your
setup cuts long calls short, as [Claude Code](/guides/agents/claude-code/) shows. For a quick
first look an agent can also pass `quality="fast"`.
