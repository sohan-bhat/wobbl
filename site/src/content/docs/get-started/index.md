---
title: "Get started"
description: "Install Wobbl, add CalculiX and ngspice, connect your coding agent and get a first checked answer in a few minutes."
lead: "Wobbl runs real engineering solvers on your parts and circuits and checks every answer before anyone acts on it."
cards:
  - title: "Installation"
    href: "/get-started/installation/"
    text: "Install the wobbl command with pipx or uv on macOS, Linux or Windows."
  - title: "Setup"
    href: "/get-started/setup/"
    text: "Add the solvers to your home folder and connect Claude Code, Codex, Cursor and Gemini CLI."
  - title: "First run"
    href: "/get-started/first-run/"
    text: "Ask a question that needs no files, then run a real part and read what comes back."
  - title: "Platforms"
    href: "/get-started/platforms/"
    text: "Which solvers run natively on each operating system and chip, and what to do when one is missing."
  - title: "FAQ"
    href: "/get-started/faq/"
    text: "Short answers about cost, privacy, accuracy, speed and the files Wobbl can read."
  - title: "Contributing"
    href: "/get-started/contributing/"
    text: "Work on Wobbl itself, run the tests and see how a new solver plugs in."
---

Wobbl is a command line tool, an MCP server and an agent skill, all installed as one
Python package. You give it a part as a STEP, IGES, BREP or STL file, or as a short
parametric Python file, and ask whether it will hold or how hot it gets. For circuits you
describe an LED, an RC network, a buck converter or a MOSFET, or hand it a netlist.

It meshes the part, solves it with CalculiX (or runs ngspice for circuits), refines the mesh
until the answer stops changing and then checks the result with a hand calculation, a force
or energy balance and several more tests. What you get back is the number and where it
happens, a confidence level with the reasons behind it, the assumptions, a heatmap with the
failure point marked and fixes with numbers. Your coding agent gets the same thing as
compact JSON through 13 MCP tools.

## The short version

These four commands install Wobbl, add the solvers, check the install and answer a first
question that needs no files.

```sh
pipx install "git+https://github.com/sohan-bhat/wobbl"
wobbl setup
wobbl doctor
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

The pages below go through each step, and [First run](/get-started/first-run/) moves on to
a real part.
