# Wobbl

**Simulate anything before you build it.**

Wobbl lets you and your coding agent ask engineering questions about real parts
and get answers you can check. Point it at a STEP or STL file and ask "will this
motor mount crack under 250 N?". Wobbl meshes the part, applies the loads and the
material, runs a real solver, verifies the result, and answers with the number,
a heatmap with the failure point marked, a confidence level, the assumptions it
made and a fix with numbers.

It drives the free engineering solvers that normally take a trained analyst:
[CalculiX](http://www.calculix.de) for stress and heat, [gmsh](https://gmsh.info)
with OpenCASCADE for CAD and meshing, and [ngspice](https://ngspice.sourceforge.io)
for circuits. One install gives your terminal and every coding agent (Claude Code,
Codex, Cursor, Gemini CLI) the same tools.

```
$ wobbl run bracket.py -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
    --load "40 N down @ top and within 12 of 65,12.5,6"

 FAILS  Peak 52.7 MPa on the 0.5 mm inside fillet. The part cracks at about 0.95x this load.

  peak stress   safety factor   target   max deflection
  52.7 MPa      0.95            2.5      3.47 mm             confidence high

 pass  mesh convergence   max principal stress settled: 43.8 -> 51.5 -> 53.3 -> 52.7 MPa over 4 meshes
 pass  force balance      reactions balance the loads to 8.8e-05%
 pass  hand calculation   beam theory through the critical section gives 15.6 MPa nominal; FEA peak
                          is 3.37x that, a stress concentration of about 3.4 from the local shape
 pass  singularities      no sharp corner or support artifact is above the reported stress
 pass  bolt bearing       worst hole carries 6.87 N sideways; bearing stress 0.254 MPa
 pass  units and sanity   peak strain 0.015, max deflection 3.47 mm (2.8% of size)

 Round the 0.5 mm fillet to a 3 mm radius (auto) ... Add a triangular gusset (auto) ...

$ wobbl fix wobbl-out/bracket --apply
                                    peak MPa   safety factor   verdict
  before                            52.7       0.95            fails
  Round the fillet to 3 mm          24.4       2.05            marginal
  Thicken the legs to 10 mm         22.2       2.25            marginal
  Add a gusset across the corner    10.3       4.86            holds
```

![Stress in a printed shelf bracket, peak marked at the inside corner](examples/shelf-bracket/output/images/stress.png)

## Why not just ask a model to do the math

On a textbook shape a careful hand calculation is close: for a flat cantilever
it lands within a few percent of a converged finite element answer. Real parts
are not textbook shapes. Ribs, pockets, bosses, fillets, several holes, thin
walls and off axis loads put the peak where no formula predicts it, and writing
a finite element code from scratch inside a chat runs out of time long before
it gives a trustworthy number. Wobbl does the part that takes an analyst: it
meshes the real geometry, refines where the stress is, checks equilibrium,
compares against beam theory, refuses to report singular peaks as real, and
tells you how much to trust the answer. Typical studies take one to five
minutes.

## Install

Wobbl needs Python 3.10 or newer. Until it is on PyPI, install from GitHub:

```sh
pipx install "git+https://github.com/sohan-bhat/wobbl"
# or
uv tool install "git+https://github.com/sohan-bhat/wobbl"
```

Then install the solvers and connect your agents:

```sh
wobbl setup      # CalculiX and ngspice into ~/.wobbl (no admin rights), plus MCP and skill for detected agents
wobbl doctor     # check that everything works
```

`wobbl setup` downloads a pinned micromamba and installs the solvers from
conda-forge into `~/.wobbl` (about 340 MB). Nothing goes into system folders.

| Platform | CalculiX | ngspice |
|---|---|---|
| macOS Apple Silicon | native | Intel build under Rosetta 2 |
| macOS Intel | native | native |
| Linux x86_64 | native | native |
| Linux arm64 | native | not packaged; install it with your package manager and Wobbl finds it |
| Windows x64 | native | native |

## Use it

```sh
wobbl inspect part.step                                  # faces, holes, fillets, thickness, ids
wobbl part.step "will this 6061 bracket yield under 50 lbf at the tip?"
wobbl run part.step -m petg --fix "bolt holes d=5.5" --load "250 N down at 0,-15,31 @ holes d=3.4 rigid"
wobbl fix wobbl-out/part --apply                         # re-run with fillets, gussets, thicker walls
wobbl thermal pack.step -m li-ion-cell --power "6 W" --film "5 W/m2K @ all" --ambient "35 C" --duration "60 min"
wobbl circuit led --set supply=5V --set resistor=47 --set led=blue
wobbl "what is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz, 10 uH, 22 uF?"
```

Every run writes a folder with a self contained `report.html`, a plain
`summary.txt` an agent can read, `outcome.json`, heatmaps in `images/` and the
full field as `result.vtu` for ParaView. See the [CLI reference](docs/cli.md) for
the load, support and face selector syntax.

## With coding agents

`wobbl setup` registers an MCP server and installs a skill for Claude Code,
Codex, Cursor and Gemini CLI, backing up every file it touches. The agent gets
`inspect_geometry`, `solve_structural`, `verify`, `render`, `suggest_fix`,
`solve_thermal`, `solve_circuit` and a few study building tools, with compact
JSON results and image paths. The skill tells it the workflow and what never to
skip. Details in [docs/agents.md](docs/agents.md).

## How it checks itself

Every strength answer runs mesh convergence (at least three meshes), force and
moment balance, a beam theory hand calculation through the critical section,
singularity detection for sharp corners and idealized supports, bolt bearing,
unit and deflection sanity, and a large deflection re-solve when the part bends a
good fraction of its wall. Heat answers check energy balance and a lumped hand
calculation; circuit answers compare with the textbook and re-run at half the
time step. The checks become a confidence level, high, medium or low, with the
reasons. A result that failed a check is never reported as high confidence.
[docs/verification.md](docs/verification.md) has the details.

## Validation

Thirteen cases with known answers run through the same pipeline users get:
beams, a plate with a hole (Kirsch with Peterson's finite width factor), a
thick walled pressure vessel (Lame), NAFEMS LE1 and LE10, torsion, self weight,
slab conduction, lumped cooling, an LED, an RC circuit and buck ripple. The
results table is in [docs/validation.md](docs/validation.md); run it yourself
with `wobbl validate`.

Wobbl was also run on the harder parts of a bake off against careful hand built
reference models. On a cast motor mount loaded through four bolt holes it found
124 MPa at (43.9, 9.0, 79.6) mm in about a minute; the reference converged to
124 MPa at (43.8, 8.8, 79.8) after ten minutes of solving on its finest mesh.

## Examples

Real products with real outputs (reports, heatmaps, summaries) in `examples/`:

| Example | Question | Domain |
|---|---|---|
| [shelf-bracket](examples/shelf-bracket) | Does a printed PLA bracket hold an 8 kg shelf, and what fixes it? | strength, fixes |
| [motor-mount](examples/motor-mount) | Do the ribs on a printed NEMA 17 mount earn their keep? | strength |
| [enclosure](examples/enclosure) | Does a PC enclosure survive a thumb press, and how far does it dent? | strength, thin walls |
| [drone-arm](examples/drone-arm) | Is a PA12 drone arm strong enough at full throttle? | strength, rigid loads |
| [heat-sink](examples/heat-sink) | Does a 15 W chip stay under 85 C without a fan? | heat |
| [battery-pack](examples/battery-pack) | When does a pack in a sealed box pass 60 C? | transient heat |
| [circuits](examples/circuits) | Will this LED fry? What is the ripple? How hot is the MOSFET? | circuits |

Each folder has a `run.sh` that regenerates its outputs.

## Architecture

Domain (what you ask) to solver adapter (who computes it) to verification
(can we trust it) to report (what you see). Solvers plug in as adapters, so
CFD (OpenFOAM) and motion (MuJoCo) slot in next to CalculiX and ngspice.
[docs/architecture.md](docs/architecture.md).

## Limits today

- Linear elastic materials: Wobbl reports whether yield is reached, not what
  happens after. Large deflection is checked with a nonlinear re-solve.
- Static loads. Fatigue, impact and vibration are not modeled yet.
- Contact only against rigid walls (push but not pull). Bolted joints are
  modeled as clamped holes, without preload.
- One material per study; assemblies of touching bodies are bonded.
- Thin walled parts (2 to 3 mm shells on 100 mm parts) need large tetrahedral
  meshes and take several minutes. Shell elements are on the roadmap.
- Printed plastics are treated as isotropic at 100 percent infill, with a layer
  strength knockdown when you give the print direction.

## Roadmap

- 0.2: shell elements for thin walls, fatigue estimates, bolt preload, a second
  structural solver as a cross check
- 0.3: CFD with OpenFOAM (fan cooling, drag), Elmer for coupled multiphysics
- 0.4: motion with MuJoCo (mechanisms, drop tests)

## License

MIT
