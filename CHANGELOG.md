# Changelog

## Unreleased

**Motion**
- A new motion domain flies a body from release or ignition to the ground
  through gravity that falls off with altitude, the 1976 standard atmosphere,
  drag that can follow Mach number, wind, a launch rail, a thrust curve and
  drogue and main parachutes, integrated with scipy one phase at a time
  between events. It needs nothing beyond scipy, so it downloads nothing.
- `wobbl motion` takes a round body as one line of flags or a rocket as a
  TOML spec with units, and agents get a `solve_motion` tool and a skill
  section on when to use it.
- Each flight is integrated again with a tolerance ten times tighter,
  compared with closed forms wherever the model reduces to one, closed with
  an energy balance and, for a rocket, with Tsiolkovsky's delta v less its
  gravity, drag and steering losses. Rockets also get Barrowman's stability
  margin, rail exit speed, peak loads, apogee against a target, landing speed
  and drift, with fixes for nose weight, fin span, rail length and the main
  parachute.
- Two new examples, a falling rock and a rocket launch, and six motion
  validation cases against exact closed forms, which make 25 cases in all.

## 0.1.1 (2026-10-13)

**Heat**
- A transient study whose held temperature differs from the part's starting
  temperature now runs. A plate at 20 C set on a surface held at 120 C used to
  stop CalculiX with "increment size smaller than minimum", because a face that
  jumps all at once moves its neighbors by a fixed amount however short the time
  step. The held temperature now ramps in a straight line from the starting
  temperature to its target over a tenth of the time heat takes to cross the
  smallest element at that face, or a thousandth of the run if that is shorter,
  and never quicker than CalculiX's shortest time steps can follow. The result's
  assumptions say how long the ramp was and why.
- A face that jumps gets extra saved times through the ramp and the fast flow of
  heat just after it, and a jump under 100 C gets smaller time steps in
  proportion, so a 20 C jump is followed as closely as a 100 C one.
- The transient energy balance integrates heat flows through a monotone cubic
  instead of straight lines between saved times, counts each held face on its
  own and is judged against the largest flow in the account. A hot face and a
  cold face pushing heat through a part now close within 2%, and the plate on
  the hot surface closes to 0.8%.
- A study that makes no heat says so in its assumptions, instead of saying the
  heat is spread through the part.
- A new validation case, `sudden-face`, compares the temperature 10 mm into an
  aluminum bar whose end jumps to 120 C with the erfc solution for a semi
  infinite solid. Two new end to end tests compare every node with the erfc
  solution and with the Fourier series for a bar between a hot face and a cold
  one.
- The battery pack example was run again, so its result shows the transient
  energy balance.

**Install**
- Installing on an Intel Mac no longer needs Rust and a C compiler, because
  cryptography stays below 49 there, the first version with no Intel Mac
  wheels.

## 0.1.0 (2026-10-13)

First release.

**Strength**
- STEP, IGES, BREP, STL and parametric `.py` parts, read through OpenCASCADE with
  healing only when a solid is broken, touching bodies bonded, tiny faces removed.
- Second order tetrahedral meshes (gmsh) with feature sizing, hotspot refinement,
  automatic fallbacks (another algorithm, straight sided elements, a finer size),
  and every mesh attempt isolated in a child process with a timeout.
- Loads can be forces, pressures, bolt bearing, gravity, or remote forces and
  moments through distributing (RBE3 style) or rigid couplings. Supports can be
  fixed (in all or some directions), bolted holes with washer rings, rollers,
  or wall contact solved by an active set loop with a coarse pre solve and warm
  starts.
- A face selector language picks sides, planes, normals, holes and fillets by
  size, nearest faces, spherical and box patches, and ids.
- 23 sourced materials, including sand cast A356-T6, plus custom properties.
  Each material says how it is made (printed, molded, cast or wrought), so its
  assumptions fit, and printed parts can be checked across layers.
- A `bearing` keyword for a bolt or pin pressing on one side of a hole, as in
  `--load "300 N -x bearing @ hole d=6.6"`. Without it the force spreads evenly
  over the hole faces.

**Verification on every answer**
- Mesh convergence over at least three meshes, force and moment balance, a beam
  theory hand calculation through the critical section, singularity detection
  from exact CAD geometry, bolt bearing, units and deflection sanity, a large
  deflection re-solve when a part bends a good fraction of its wall, and the
  spread between rigid and distributed load attachment.
- Confidence (high, medium, low) with reasons, and never high after a failed
  check.
- The same input gives the same numbers on every run, because meshing runs on
  one thread with a fixed seed and every size comes from the CAD. When the
  rigid and distributed load attachments disagree, the other one is converged
  at its own peak before it is quoted.
- Bounded runtime, so the last finished level is returned when time or a
  solver runs out.

**Seeing and fixing**
- Heatmaps with the failure point marked (VTK off screen, matplotlib fallback),
  convergence plots, plain text summaries and a self contained HTML report.
- Fix suggestions with numbers. Fillets and gussets are applied to any STEP
  through OpenCASCADE, along with thickness and fillet parameters of parametric
  parts and material swaps, and each one is simulated again for a before and
  after table.

**Heat and circuits**
- Steady and transient conduction with convection, radiation, heat sources and
  held temperatures, with an energy balance on steady and transient runs and
  lumped hand calculations.
- ngspice checks for LEDs, RC networks, buck converter ripple and MOSFET junction
  temperature, plus arbitrary netlists, with textbook and time step checks.

**Agents**
- `wobbl setup` installs CalculiX and ngspice into `~/.wobbl` with a pinned
  micromamba and registers an MCP server and skill for Claude Code, Codex, Cursor
  and Gemini CLI, with backups and `wobbl uninstall`. A test parses every
  command in the skill against the real CLI.
- A validation suite of 18 cases, with textbook solutions, NAFEMS LE1 and LE10,
  and five real parts against converged reference models. It runs in about
  30 minutes on a 2018 six core laptop.
