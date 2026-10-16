---
title: "Installation"
description: "Install the wobbl command with npm, pipx or uv on macOS, Linux or Windows."
lead: "Wobbl is a Python package, so one command puts the wobbl command on your path."
order: 1
---

## What you need

Wobbl runs on macOS, Linux and Windows. Installing with npm needs Node 18 or newer and no
Python of your own. Installing with pipx or uv needs Python 3.10 or newer. The package brings
gmsh, which carries OpenCASCADE for reading CAD, along with numpy, scipy, matplotlib and
the MCP library, so there is no CAD license or compiler involved. The solvers come in the next
step, `wobbl setup`, which downloads under 100 MB once and leaves about 70 MB on disk.

## Install

The quickest way is npm.

```sh
npm i -g wobbl
```

Wobbl itself is a Python package on PyPI, so npm installs the newest one for good with uv,
or pipx when you have pipx and no uv. When you have neither, it downloads a pinned uv into
`~/.wobbl/uv`, checks it against its published SHA256 and uses that, and uv brings its own
Python. You see one line per step, and the installer's full output goes to
`~/.wobbl/logs/install.log`, which is printed only if something fails. It never edits your
shell profile. If the folder that holds the `wobbl` command is not on your PATH, it prints
the exact line that adds it.

If you already use pipx or uv, install from PyPI directly. Both give Wobbl an environment of
its own, away from your other Python projects.

```sh
pipx install wobbl
```

```sh
uv tool install wobbl
```

To try Wobbl without installing it first, `npx wobbl setup` installs it on the first run and
then runs setup.

Check that the command is there.

```sh
wobbl --version
```

It prints `wobbl, version 0.1.1`.

## Update or remove

To update, run `npm i -g wobbl` again, or `pipx upgrade wobbl`, or `uv tool upgrade wobbl`.
Updating leaves `~/.wobbl` alone, so the solvers and your agent registrations keep working
and nothing has to be downloaded twice.

To remove Wobbl, run `pipx uninstall wobbl` or `uv tool uninstall wobbl`. The npm install put Wobbl in uv or pipx too, so remove it there, then run `npm rm -g wobbl` for the launcher. To remove
everything Wobbl added, including the solvers, run `wobbl uninstall --all` before you
uninstall the package, which [wobbl uninstall](/guides/cli/uninstall/) covers in full.

## Next

Run [`wobbl setup`](/get-started/setup/) to add CalculiX and ngspice and connect your coding
agents.
