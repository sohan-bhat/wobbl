---
title: "Platforms"
description: "Which solvers Wobbl runs natively on macOS, Linux and Windows, and how to fill the gaps."
lead: "CalculiX runs natively everywhere, and ngspice has two gaps that each take one command to close."
order: 4
---

| Platform | CalculiX | ngspice |
|---|---|---|
| macOS Apple Silicon | native | Intel build under Rosetta 2 |
| macOS Intel | native | native |
| Linux x86_64 | native | native |
| Linux arm64 | native | not packaged, install it with your package manager and Wobbl finds it |
| Windows x64 | native | native |

The tests run on Ubuntu, macOS and Windows with Python 3.12 on every push, and on Ubuntu with
Python 3.10 and 3.13 as well, including the end to end studies with the real solvers.

## Apple Silicon

conda-forge builds ngspice for Intel Macs only, so `wobbl setup` installs that build into
`~/.wobbl/env-x86` and macOS runs it through Rosetta 2. If Rosetta is not installed yet,
setup's own solver test fails and tells you to run this.

```sh
softwareupdate --install-rosetta --agree-to-license
```

CalculiX is a native arm64 build, so strength and heat studies run at full speed.

## Linux arm64

conda-forge has no ngspice for this platform. Strength and heat work as usual, and for
circuits you install ngspice yourself, after which Wobbl finds it on your PATH.

```sh
sudo apt install ngspice
```

If it lives somewhere unusual, set `WOBBL_NGSPICE` to the full path of the binary.

## Headless Linux

Wobbl draws its heatmaps with VTK off screen. On a server with no display, VTK needs a few
system libraries for OpenGL and fonts, and the Ubuntu test machines install these.

```sh
sudo apt-get install -y libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgl1 xvfb
```

When VTK cannot render at all, Wobbl falls back to matplotlib for the images and keeps going,
and `wobbl doctor` says which renderer is in use.

## Speed

CalculiX runs on up to four threads. The time budget of a strength study is spent by a cost
model calibrated on a 2018 six core laptop with an Intel i7-8750H, so a faster machine
reaches the same mesh levels and simply finishes sooner. A strength study on a
real part takes one to seven minutes on that laptop, and simple parts take well under a
minute.
