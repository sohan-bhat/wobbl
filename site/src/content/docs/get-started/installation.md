---
title: "Installation"
description: "Install the wobbl command from PyPI with pipx or uv on macOS, Linux or Windows."
lead: "Wobbl is a Python package, so one command puts the wobbl command on your path."
order: 1
---

## What you need

Wobbl needs Python 3.10 or newer on macOS, Linux or Windows. The package itself brings
gmsh, which carries OpenCASCADE for reading CAD, along with numpy, scipy, matplotlib and
the MCP library, so there is no CAD license or compiler involved. The solvers come in the next
step, `wobbl setup`, which needs about 340 MB of disk and an internet connection once.

## Install

Wobbl is on PyPI, and the cleanest way to install it is with pipx or uv. Both give it an
environment of its own, which keeps its dependencies away from your other Python projects.

```sh
pipx install wobbl
```

If you use uv, this does the same thing.

```sh
uv tool install wobbl
```

Check that the command is there.

```sh
wobbl --version
```

It prints `wobbl, version 0.1.1`.

## Update or remove

To update, run `pipx upgrade wobbl`, or `uv tool upgrade wobbl` if you installed it with uv.
Updating leaves `~/.wobbl` alone, so the solvers and your agent registrations keep working
and nothing has to be downloaded twice.

To remove Wobbl, run `pipx uninstall wobbl` or `uv tool uninstall wobbl`. To remove
everything Wobbl added, including the solvers, run `wobbl uninstall --all` before you
uninstall the package, which [wobbl uninstall](/guides/cli/uninstall/) covers in full.

## Next

Run [`wobbl setup`](/get-started/setup/) to add CalculiX and ngspice and connect your coding
agents.
