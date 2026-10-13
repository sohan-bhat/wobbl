---
title: "Contributing"
description: "Set up Wobbl for development, run the tests and see how the code is layered and how a new solver plugs in."
lead: "Issues and pull requests are welcome, and this page has what you need to work on Wobbl itself."
order: 6
---

## Set up a checkout

```sh
git clone https://github.com/sohan-bhat/wobbl && cd wobbl
python -m pip install -e ".[dev]"
python -m pytest -m "not solver"      # fast unit tests
wobbl setup --no-agents               # CalculiX and ngspice
python -m pytest -m solver            # end to end tests with the solvers
```

Tests marked `solver` need CalculiX or ngspice, and the ones marked `slow` are long end to end
cases. If you change the physics, add or update a case in `src/wobbl/validation/cases.py` so
that `wobbl validate` covers it.

## How the code is layered

Wobbl is layered the way the work flows. A question or a set of flags goes to a domain package
(`structural`, `thermal` or `circuits`), which turns loads, supports and materials into a
solver neutral model in `fe.py`. A solver adapter in `solvers/` writes the input files, runs
CalculiX or ngspice and reads the results back. The checks in `verify/` and in each domain
turn the results into a confidence level, and `report/` draws the heatmaps and writes the
summary and the HTML page.

| Path | What it holds |
|---|---|
| `units.py` | parses quantities like `200 N`, `45 lbf` or `5 W/m2K` into one internal system (mm, N, MPa, t, s, mW, C) |
| `materials.py` | the sourced material table and custom materials |
| `geometry/` | CAD import and healing, face and feature detection, the face selector language, parametric parts and carrying selections across geometry edits |
| `mesh/` | second order gmsh meshes with feature and hotspot refinement and fallbacks, and the child process isolation |
| `structural/` | load and support parsing, model building, the convergence loop, hotspots, the verdict and the time budget |
| `thermal/`, `circuits/` | heat studies and circuit templates with their checks |
| `verify/` | shared checks, beam theory and confidence |
| `fix/` | fix suggestions, OpenCASCADE edits for fillets and gussets, and the re-runs |
| `mcp/`, `skill/`, `setup/` | the MCP server, the agent skill and `wobbl setup` |
| `validation/` | the cases with known answers and their parts |

## Adding a solver

A new domain, such as CFD with OpenFOAM or motion with MuJoCo, is one adapter class in
`wobbl/solvers/` that registers itself with the domains it serves, plus a domain package that
builds its model and checks. `solvers.base.for_domain` picks the first available adapter, so a
second structural solver could sit next to CalculiX and serve as a cross check.

```python
class OpenFOAM(SolverAdapter):
    name = "OpenFOAM"
    domains = ("cfd",)
    def locate(self): return find_tool(["simpleFoam"], "WOBBL_OPENFOAM")
    def solve_flow(self, model, workdir): ...
register(OpenFOAM())
```

CalculiX and gmsh were picked because both are mature, free, scriptable and packaged on
conda-forge for every platform Wobbl supports, so setup can install them without admin
rights. gmsh's Python wheel also carries OpenCASCADE, which gives STEP import, healing and
geometry edits with no CAD license.

## Isolation

OpenCASCADE and gmsh can hang or crash on bad CAD. Every import, meshing attempt and CAD edit
runs in a child process with a timeout, and a hang or crash makes Wobbl try the next strategy,
another meshing algorithm or straight sided elements, instead of taking the CLI or the MCP
server down with it. The MCP server also moves file descriptor 1 to stderr, so a native
library that prints cannot corrupt the protocol stream.
