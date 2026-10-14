---
title: "Simulations"
description: "The four kinds of study Wobbl runs, strength, heat, circuits and motion, and the solver behind each."
lead: "Wobbl runs four kinds of study, and each one comes back checked and with a confidence level."
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
  - title: "Motion"
    href: "/guides/simulations/motion/"
    text: "Drops, throws and rocket flights through air, wind, thrust and parachutes."
---

| Study | Solver | Command | Checked by |
|---|---|---|---|
| Strength | CalculiX | `wobbl run` | mesh convergence, force balance, a beam hand calculation, singularities and more |
| Heat | CalculiX | `wobbl thermal` | mesh convergence, energy balance and a lumped hand calculation |
| Circuits | ngspice | `wobbl circuit` | the textbook answer, power balance and a run at half the time step |
| Motion | scipy | `wobbl motion` | closed forms, an energy balance, a delta v budget and a run at a tighter tolerance |

You can also ask a strength, heat or circuit question in plain words, as in `wobbl part.step "will it crack under 200 N?"`,
and Wobbl picks the kind of study from the question, which
[wobbl ask](/guides/cli/ask/) explains.
