---
title: "FAQ"
description: "Short answers about what Wobbl costs, where your files go, how accurate and fast it is and what it reads."
lead: "Short answers to the questions people ask before they trust a simulation tool with a real part."
order: 5
---

## What does it cost?

Nothing. Wobbl is MIT licensed, and CalculiX, ngspice, gmsh and OpenCASCADE are all free and
open source, so there is no license server and no account.

## Does my part leave my computer?

No. The only command that touches the network is `wobbl setup`, which downloads micromamba
from GitHub and the solver packages from conda-forge. Studies run entirely on your machine.
When you use Wobbl through a coding agent, the agent sees the compact JSON results it asks
for, the same as with any other tool it calls.

## How accurate is it?

{{Cases}} cases with known answers run through the same pipeline you use, and every one lands
inside its tolerance. The {{exact}} whose reference is exact for the modeled setup land within
2%, and the five real parts land within 4.3% of reference models that were refined until
they stopped changing. [Validation](/validation/) has the full table. Each answer also
carries its own checks, so you see on the spot whether a particular run can be trusted.

## How long does a study take?

A circuit check takes from under a second to about eight seconds. A strength study on a real part takes one to
seven minutes on a 2018 laptop, and a simple bracket takes well under a minute. Every study
has a time budget, 300 seconds by default, and when it runs out you still get the last
finished mesh level back with its checks.

## Which files can it read?

STEP, IGES and BREP through OpenCASCADE, watertight STL, and parametric parts written as a
short Python file. Geometry is read in millimeters, so a file drawn in inches needs
`--scale 25.4`. [Geometry](/guides/inputs/geometry/) has the details.

## Do I need a coding agent?

No. Everything works from the terminal, and the agent tools call the same code. The agent is
handy when you want someone else to pick the supports and loads from a description and then
read the checks for you.

## Why is the confidence medium?

Medium means at least one check raised a warning, for example a verdict that flips with an
assumption, a sharp corner in a brittle material or a number still moving between meshes.
The reasons are always listed next to the level, and [Confidence](/guides/results/confidence/)
explains each one.

## Why do I get slightly different numbers on another computer?

The same study gives the same numbers on every run on one machine, digit for digit. A machine
with a different build of gmsh, OpenCASCADE or CalculiX can move them slightly, because
library versions and floating point change where mesh nodes land.

## What does it not model?

Plasticity after yield, buckling, fatigue, impact, vibration, bolt preload and contact between separate
parts, among others. [Limits](/validation/limits/) lists them with what is planned.

## How do I remove it?

`wobbl uninstall --all` removes the agent registrations and the whole `~/.wobbl` folder, and
`pipx uninstall wobbl` or `uv tool uninstall wobbl` removes the command.
