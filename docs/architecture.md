# Architecture

Wobbl is layered the way the work flows:

```
question or flags
   |
   v
domain        wobbl/structural, wobbl/thermal, wobbl/circuits
   |            loads, supports, materials -> a solver neutral model (wobbl/fe.py)
   v
solver        wobbl/solvers: CalculiX (structural, thermal), ngspice (circuits)
adapter         writes input files, runs the binary, reads results back
   |
   v
verification  wobbl/verify, plus the domain specific checks
   |            convergence, equilibrium, hand calculations, singularities, sanity -> confidence
   v
report        wobbl/report: heatmaps (pyvista, matplotlib fallback), summary text, HTML
```

## Modules

| Path | Role |
|---|---|
| `units.py` | parse `200 N`, `45 lbf`, `3 ksi`, `5 W/m2K`; one internal system (mm, N, MPa, t, s, mW, C) |
| `materials.py` | sourced material table and custom materials |
| `geometry/cad.py` | STEP, IGES, BREP and STL import through OpenCASCADE, healing only when broken, bonding touching bodies, tiny face removal; every CAD call runs in a child process |
| `geometry/faces.py` | faces, holes, fillets, bosses, sharp inside corners and wall thickness from an oriented boundary mesh |
| `geometry/select.py` | the face selector language |
| `geometry/parametric.py` | `.py` parts with named parameters |
| `geometry/remap.py` | carry selections across geometry edits |
| `mesh/mesher.py` | gmsh second order tets in CalculiX order, feature and hotspot refinement fields, fallbacks |
| `mesh/isolate.py` | run native code in a child process with a timeout |
| `fe.py` | solver neutral static and heat models and results |
| `solvers/` | adapter interface, registry, CalculiX, ngspice, and placeholders for OpenFOAM, MuJoCo and Elmer |
| `structural/` | load and support parsing, model building, the convergence loop, hotspots, verdict and report |
| `thermal/` | heat studies and their checks |
| `circuits/` | LED, RC, buck and MOSFET templates and arbitrary netlists |
| `verify/` | shared checks, beam theory, confidence |
| `fix/` | fix suggestions, OpenCASCADE edits (fillet, gusset), re-runs |
| `report/` | rendering, HTML |
| `mcp/` | MCP server and its tools |
| `setup/` | micromamba solver install, agent registration |
| `validation/` | cases with known answers |

## Adding a solver

A new domain (CFD with OpenFOAM, motion with MuJoCo) is one adapter class in
`wobbl/solvers/` that registers itself with the domains it serves, plus a
domain package that builds its model and checks. `solvers.base.for_domain`
picks the first available adapter, so a second structural solver can be added
next to CalculiX and used as a cross check.

```python
class OpenFOAM(SolverAdapter):
    name = "OpenFOAM"
    domains = ("cfd",)
    def locate(self): return find_tool(["simpleFoam"], "WOBBL_OPENFOAM")
    def solve_flow(self, model, workdir): ...
register(OpenFOAM())
```

## Why CalculiX and gmsh

Both are mature, free, scriptable and packaged on conda-forge for macOS (Intel
and Apple Silicon), Linux and Windows, so `wobbl setup` can install them
without admin rights. gmsh's Python wheel carries OpenCASCADE, which gives STEP
import, healing and geometry edits with no CAD license.

## Isolation

OpenCASCADE and gmsh can hang or crash on bad CAD. Every import, meshing
attempt and CAD edit runs in a child process (`python -m wobbl.mesh.isolate`)
with a timeout; a hang or crash makes Wobbl try the next strategy (another
meshing algorithm, straight sided elements) instead of taking the CLI or MCP
server down. The MCP server also moves file descriptor 1 to stderr so native
libraries cannot corrupt the protocol stream.
