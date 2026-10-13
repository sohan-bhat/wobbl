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

 FAILS  Peak 52.8 MPa on the 0.5 mm inside fillet. The part cracks at about 0.95x this load.

  peak stress   safety factor   target   max deflection
  52.8 MPa      0.95            2.5      3.46 mm             confidence high

 pass  mesh convergence   max principal stress settled: 44.2 -> 51.1 -> 53.4 -> 52.8 MPa over 4 meshes
 pass  force balance      reactions balance the loads to 0.00011%
 pass  hand calculation   beam theory through the critical section gives 15.2 MPa nominal; FEA peak
                          is 3.47x that, a stress concentration of about 3.5 from the local shape
 pass  singularities      no sharp corner or support artifact is above the reported stress
 pass  bolt bearing       worst hole carries 13.1 N sideways; bearing stress 0.485 MPa
 pass  units and sanity   peak strain 0.015, max deflection 3.46 mm (2.8% of size)

 Round the 0.5 mm fillet to a 3 mm radius (auto) ... Add a triangular gusset (auto) ...

$ wobbl fix wobbl-out/bracket --apply
                                    peak MPa   safety factor   verdict
  before                            52.8       0.95            fails
  Round the fillet to 3 mm          24.5       2.04            marginal
  Thicken the legs to 10 mm         21.5       2.32            marginal
  Add a gusset across the corner    10.3       4.84            holds
```

![Stress in a printed shelf bracket, peak marked at the inside corner](examples/shelf-bracket/output/images/stress.png)

## Why not just ask a model to do the math

We tested exactly that. Given a CAD file and a strength question, a capable
coding agent working alone answered in **1 of 10** tries within 15 minutes. On
beam like parts its hand calculations are good, within a few percent. On a
ribbed cast motor mount and a thin walled enclosure, the two parts that actually
fail, it produced nothing in 15 minutes over six tries: it kept writing its own
finite element code and hitting the same traps (meshers folding thin curved
walls, singular peaks at bonded bolts, load idealizations that flip the
verdict).

Wobbl answers those parts with the trap handling built in:

| Part | Converged reference | Wobbl | Confidence |
|---|---|---|---|
| Cast motor mount, belt pull through four bolts | 124.3 MPa at the rib tip | 123.9 MPa, same spot | medium, and it says why: the verdict flips if the servo is treated as rigid |
| ABS enclosure shell pressed near a corner | 93 MPa, 1.93 mm deflection | 89.8 MPa, 1.91 mm | high |
| PA12 truss drone arm, thrust and torque | 11.5 MPa, 4.09 mm rise | 11.53 MPa, 4.08 mm | high |
| PLA bracket, sharp inside corner | about 78 MPa | 80.0 MPa | high |

The full comparison is in [docs/bakeoff.md](docs/bakeoff.md).

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

Eighteen cases with known answers run through the same pipeline users get:
beams, a plate with a hole (Kirsch with Peterson's finite width factor), a thick
walled pressure vessel (Lame), NAFEMS LE1 and LE10, torsion, self weight, slab
conduction, lumped cooling, an LED, an RC circuit, buck ripple, and five real
parts against converged reference models. Every case with an exact textbook or
benchmark answer lands within 2%. The table is in
[docs/validation.md](docs/validation.md); run it yourself with `wobbl validate`.

## Examples

Real products with real outputs (reports, heatmaps, summaries) in `examples/`:

| Example | Question | Domain |
|---|---|---|
| [cast-motor-mount](examples/cast-motor-mount) | Does a cast motor mount meet a safety factor of 2 under a 2 kN belt pull? | strength, remote loads |
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
