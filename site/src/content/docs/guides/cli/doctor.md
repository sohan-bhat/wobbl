---
title: "wobbl doctor"
description: "Check that every piece Wobbl needs is installed and working, from gmsh and rendering to the solvers and each agent."
lead: "doctor checks every piece Wobbl needs and says what is missing."
order: 12
---

```sh
wobbl doctor [--json]
```

```text
  ok         wobbl           0.1.0 on Python 3.13.2
  ok         gmsh            4.15.2
  ok         rendering       heatmaps draw with numpy and matplotlib, no OpenGL needed
  ok         micromamba      /Users/you/.wobbl/bin/micromamba
  ok         calculix        2.23 at /Users/you/.wobbl/env/bin/ccx
  ok         ngspice         41 at /Users/you/.wobbl/env/bin/ngspice
  planned    OpenFOAM        OpenFOAM support is planned for Wobbl 0.3.
  planned    MuJoCo          MuJoCo support is planned for Wobbl 0.4.
  planned    Elmer           Elmer support is planned for Wobbl 0.3.
  ok         agent claude    MCP and skill installed
  off        agent codex     MCP no, skill no
```

## What each mark means

| Mark | Meaning |
|---|---|
| ok | installed and working, or for an agent, both the MCP server and the skill are in place |
| missing | needed and not working, with the reason, and `wobbl setup` installs it |
| off | an agent Wobbl is not registered with |
| planned | a solver on the roadmap that is not part of this version |
| on demand | a solver Wobbl installs the first time you run a study of its domain |

Each solver is found the same way a study finds it and then run on the same tiny problem setup
uses, so a solver that is present but broken shows up here. The rendering line draws a small heatmap
the way a report does. An agent that has only one of its two pieces
shows as missing with which one is there, and `wobbl setup --agents` with its name repairs it.
`--json` prints the same information as JSON.
