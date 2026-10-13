# Changelog

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
