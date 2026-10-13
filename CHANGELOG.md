# Changelog

## Unreleased

**Fixed**
- The same study on the same input now gives the same numbers on every run.
  Before, the cast motor mount's rigid servo case came out at 55.5 MPa on one
  run and 44.2 MPa on another, its main answer moved between 123.4 and 123.9
  MPa, the enclosure shell between 92.5 and 89.8 MPa and the shelf bracket
  between 52.8 and 51.8 MPa. Four sources caused it, and all four are fixed.
  - gmsh meshed faces on parallel threads, which moved nodes from run to run,
    so it now runs on one thread with a fixed seed, and OpenCASCADE booleans
    run serially.
  - The thinnest wall estimate drew random triangles from that mesh, weighted
    them by area twice and took a quantile that jumped between 3 and 5.5 mm.
    It now casts rays from points laid out on the CAD faces and averages the
    thinnest 5% of the surface.
  - The exclusion radius around singular spots came from the first mesh's
    element sizes, and now it comes from the sizes the sizing rules ask for.
  - Refinement stopped when the clock said the next mesh would not fit. The
    time budget is now spent by a cost model of each mesh and solve, with the
    clock only as a backstop at three times the budget.
  A test runs a study twice and compares every number. The cast motor mount
  gave 123.7 MPa and 55.0 MPa for the rigid servo case on four runs back to
  back.
- The large deflection re-solve runs on a coarse mesh of its own in two load
  steps. On the 77k node drone arm truss it was still running after 17
  minutes, and a 90 s timeout made it finish on some runs and not others. It
  now takes 75 s there.
- The load introduction check converges the other attachment idealization at
  its own peak. It used to solve it only on the mesh refined for the main
  answer, so a peak somewhere else came from a coarse mesh. When the verdict
  flips or the peak moves, Wobbl now refines around that peak with the main
  study's convergence rule and time budget and reports the trend, and if the
  budget runs out first, the best mesh reached is labeled as not converged.
  On the cast motor mount the rigid servo case now converges to 55.0 MPa (54.2, 54.3, 55.0
  MPa over three meshes, safety factor 3.00), where before it read 47.2 MPa on
  the mesh refined for the distributed case.
- Custom materials no longer inherit printed PLA's properties. A custom cast
  aluminum listed a printed part assumption and PLA's density. Materials now say
  how they are made (`process=printed`, `molded`, `cast`, `wrought`), the
  assumptions follow it, and what a custom material leaves out comes from the
  nearest library material made the same way.
- When coarsening the first mesh folds a thin wall, the study keeps the first
  mesh instead of failing, and mesh fallbacks shrink the refinements along with
  the global size.
- Material swaps respect a safety factor target the user set.
- `wobbl circuit led` works when a folder named `led` sits in the current
  directory.
- Meshes with flat sliver elements (all four corners in one plane) are rejected
  and remeshed with the next strategy instead of failing in CalculiX.

**Added**
- Sand cast aluminum A356-T6 in the materials library.

**Docs**
- The validation suite takes about 30 minutes (29 on a 2018 six core laptop),
  not 7. All 18 cases pass on the repeatable engine.

## 0.1.0

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
- 22 sourced materials plus custom properties, and printed parts can be checked
  across layers.

**Verification on every answer**
- Mesh convergence over at least three meshes, force and moment balance, a beam
  theory hand calculation through the critical section, singularity detection
  from exact CAD geometry, bolt bearing, units and deflection sanity, a large
  deflection re-solve when a part bends a good fraction of its wall, and the
  spread between rigid and distributed load attachment.
- Confidence (high, medium, low) with reasons, and never high after a failed
  check.
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
  held temperatures, with energy balance and lumped hand calculations.
- ngspice checks for LEDs, RC networks, buck converter ripple and MOSFET junction
  temperature, plus arbitrary netlists, with textbook and time step checks.

**Agents**
- `wobbl setup` installs CalculiX and ngspice into `~/.wobbl` with a pinned
  micromamba and registers an MCP server and skill for Claude Code, Codex, Cursor
  and Gemini CLI, with backups and `wobbl uninstall`.
- A validation suite of 18 cases, with textbook solutions, NAFEMS LE1 and LE10,
  and five real parts against converged reference models.
