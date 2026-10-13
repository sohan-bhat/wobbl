---
title: "Installation"
description: "Install the wobbl command from GitHub with pipx or uv on macOS, Linux or Windows."
lead: "Wobbl is a Python package, so one command puts the wobbl command on your path."
order: 1
---

## What you need

Wobbl needs Python 3.10 or newer on macOS, Linux or Windows. The package itself brings
gmsh, which carries OpenCASCADE for reading CAD, along with numpy, scipy, pyvista and the
MCP library, so there is no CAD license or compiler involved. The solvers come in the next
step, `wobbl setup`, which needs about 340 MB of disk and an internet connection once.

## Install

Wobbl is not on PyPI yet, so install it straight from GitHub. pipx and uv both give it an
environment of its own, which keeps its dependencies away from your other Python projects.

```sh
pipx install "git+https://github.com/sohan-bhat/wobbl"
```

If you use uv, this does the same thing.

```sh
uv tool install "git+https://github.com/sohan-bhat/wobbl"
```

Check that the command is there.

```sh
wobbl --version
```

It prints `wobbl, version 0.1.0`.

## Update or remove

The simplest way to update a GitHub install is to remove the package and install it again.
Removing the package leaves `~/.wobbl` alone, so the solvers and your agent registrations
keep working and nothing has to be downloaded twice.

```sh
pipx uninstall wobbl
pipx install "git+https://github.com/sohan-bhat/wobbl"
```

With uv the commands are `uv tool uninstall wobbl` and the install line above. To remove
everything Wobbl added, including the solvers, run `wobbl uninstall --all` before you
uninstall the package, which [wobbl uninstall](/guides/cli/uninstall/) covers in full.

## Next

Run [`wobbl setup`](/get-started/setup/) to add CalculiX and ngspice and connect your coding
agents.
