---
title: "Simulations"
description: "The three kinds of study Wobbl runs, strength, heat and circuits, and the solver behind each."
lead: "Wobbl runs three kinds of study, and each one comes back checked and with a confidence level."
cards:
  - title: "Strength"
    href: "/guides/simulations/strength/"
    text: "Static stress and deflection with CalculiX, refined until the peak settles."
  - title: "Heat"
    href: "/guides/simulations/heat/"
    text: "Steady and transient temperatures with convection, radiation and held temperatures."
  - title: "Circuits"
    href: "/guides/simulations/circuits/"
    text: "LEDs, RC networks, buck converters, MOSFETs or any netlist in ngspice."
---

| Study | Solver | Command | Checked by |
|---|---|---|---|
| Strength | CalculiX | `wobbl run` | mesh convergence, force balance, a beam hand calculation, singularities and more |
| Heat | CalculiX | `wobbl thermal` | mesh convergence, energy balance and a lumped hand calculation |
| Circuits | ngspice | `wobbl circuit` | the textbook answer, power balance and a run at half the time step |

You can also ask any of them in plain words, as in `wobbl part.step "will it crack under 200 N?"`,
and Wobbl picks the kind of study from the question, which
[wobbl ask](/guides/cli/ask/) explains.
