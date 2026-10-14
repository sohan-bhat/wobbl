---
title: "Command line"
description: "Every wobbl command with what it does, from asking in plain words to setup, doctor and the validation suite."
lead: "One command, wobbl, with a subcommand for each job and a plain words form for quick questions."
cards:
  - title: "wobbl ask"
    href: "/guides/cli/ask/"
    text: "Ask about a part or a circuit in plain words."
  - title: "wobbl run"
    href: "/guides/cli/run/"
    text: "A strength study with explicit material, supports and loads."
  - title: "wobbl inspect"
    href: "/guides/cli/inspect/"
    text: "Holes, fillets, sharp corners, walls and face ids of a part."
  - title: "wobbl thermal"
    href: "/guides/cli/thermal/"
    text: "Steady or transient heat."
  - title: "wobbl circuit"
    href: "/guides/cli/circuit/"
    text: "LED, RC, buck and MOSFET checks, or any netlist."
  - title: "wobbl fix"
    href: "/guides/cli/fix/"
    text: "List fixes, or build them and solve each one again."
---

| Command | What it does |
|---|---|
| [`wobbl PART "question"`](/guides/cli/ask/) | answers a question in plain words, guessing what it leaves out |
| [`wobbl run`](/guides/cli/run/) | runs a strength study with explicit material, supports and loads |
| [`wobbl inspect`](/guides/cli/inspect/) | lists the holes, fillets, sharp corners, walls and faces of a part |
| [`wobbl thermal`](/guides/cli/thermal/) | runs a steady or transient heat study |
| [`wobbl circuit`](/guides/cli/circuit/) | checks an LED, RC network, buck converter or MOSFET, or runs a netlist |
| [`wobbl check`](/guides/cli/check/) | shows the checks and confidence of a finished run |
| [`wobbl report`](/guides/results/files/) | rebuilds and opens the HTML report of a finished run, covered with the result files |
| [`wobbl fix`](/guides/cli/fix/) | lists fixes, or applies them and solves each version again |
| [`wobbl materials`](/guides/cli/materials/) | prints the material table |
| [`wobbl validate`](/guides/cli/validate/) | runs the {{cases}} validation cases |
| [`wobbl setup`](/guides/cli/setup/) | installs the solvers and connects your coding agents |
| [`wobbl doctor`](/guides/cli/doctor/) | checks every piece Wobbl needs |
| [`wobbl uninstall`](/guides/cli/uninstall/) | removes the agent registrations, and with `--all` the solvers too |
| [`wobbl mcp`](/guides/cli/mcp/) | runs the MCP server for an agent |

`wobbl --version` (or `-V`) prints the version, and `-h` or `--help` works on every command.

## Results

Every study prints a readable summary in the terminal and writes its full result to a folder,
`wobbl-out/<part>` unless you pass `-o`. That folder holds `summary.txt`, `report.html`,
`outcome.json`, the heatmaps in `images/` and, for strength, the full field in `result.vtu`.
Add `--json` to print the result as JSON instead, and `-q` to drop the progress lines.
[Files](/guides/results/files/) describes them all.
