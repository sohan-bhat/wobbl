# Changelog

## 0.1.0

First release.

**Strength**
- STEP, IGES, BREP, STL and parametric `.py` parts, read through OpenCASCADE with
  healing only when a solid is broken, touching bodies bonded, tiny faces removed.
- Second order tetrahedral meshes (gmsh) with feature sizing, hotspot refinement,
  automatic fallbacks (another algorithm, straight sided elements, a finer size),
  and every mesh attempt isolated in a child process with a timeout.
- Loads: forces, pressures, bolt bearing, gravity, remote forces and moments
  through distributing (RBE3 style) or rigid couplings. Supports: fixed (all or
  some directions), bolted holes with washer rings, rollers, and wall contact
  solved by an active set loop with a coarse pre solve and warm starts.
- A face selector language: sides, planes, normals, holes and fillets by size,
  nearest faces, spherical and box patches, ids.
- 22 sourced materials plus custom properties; printed parts can be checked
  across layers.

**Verification on every answer**
- Mesh convergence over at least three meshes, force and moment balance, a beam
  theory hand calculation through the critical section, singularity detection
  from exact CAD geometry, bolt bearing, units and deflection sanity, a large
  deflection re-solve when a part bends a good fraction of its wall, and the
  spread between rigid and distributed load attachment.
- Confidence (high, medium, low) with reasons; never high after a failed check.
- Bounded runtime: the last finished level is returned when time or a solver
  runs out.

**Seeing and fixing**
- Heatmaps with the failure point marked (VTK off screen, matplotlib fallback),
  convergence plots, plain text summaries and a self contained HTML report.
- Fix suggestions with numbers; fillets and gussets applied to any STEP through
  OpenCASCADE, thickness and fillet parameters of parametric parts, material
  swaps, each re-simulated for a before and after table.

**Heat and circuits**
- Steady and transient conduction with convection, radiation, heat sources and
  held temperatures, with energy balance and lumped hand calculations.
- ngspice checks for LEDs, RC networks, buck converter ripple and MOSFET junction
  temperature, plus arbitrary netlists, with textbook and time step checks.

**Agents**
- `wobbl setup` installs CalculiX and ngspice into `~/.wobbl` with a pinned
  micromamba and registers an MCP server and skill for Claude Code, Codex, Cursor
  and Gemini CLI, with backups and `wobbl uninstall`.
- Validation suite of 18 cases: textbook solutions, NAFEMS LE1 and LE10, and five
  real parts against converged reference models.
