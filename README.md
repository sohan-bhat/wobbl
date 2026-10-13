<p align="center">
  <a href="https://sohan-bhat.github.io/wobbl/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="site/assets/logo-dark.svg">
      <img alt="Wobbl" src="site/assets/logo-light.svg" width="280">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Simulate anything before you build it.</strong><br>
  Verified strength, heat and circuit simulation for coding agents and the people who design physical things.
</p>

<p align="center">
  <a href="https://github.com/sohan-bhat/wobbl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sohan-bhat/wobbl/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/sohan-bhat/wobbl"></a>
  <a href="pyproject.toml"><img alt="Python 3.10 or newer" src="https://img.shields.io/badge/python-3.10%2B-3776ab"></a>
</p>

<p align="center">
  <a href="https://sohan-bhat.github.io/wobbl/">Website</a> &nbsp;|&nbsp;
  <a href="docs/getting-started.md">Getting started</a> &nbsp;|&nbsp;
  <a href="docs/cli.md">CLI reference</a> &nbsp;|&nbsp;
  <a href="examples">Examples</a>
</p>

![Stress heatmap of a cast aluminum motor mount under a 2 kN belt pull, peak 124 MPa marked where the rib tip meets the upright](examples/cast-motor-mount/output/images/stress.png)

<p align="center"><sub>A real Wobbl output: a sand cast motor mount under a 2 kN belt pull. Peak 124 MPa at the rib tip, safety factor 1.34 against a target of 2.</sub></p>

## What it is

Wobbl is a command line tool, an MCP server and an agent skill that run real
engineering solvers and check every answer before anyone acts on it. Point it at
a STEP, IGES, STL or parametric Python part, ask a question in plain words or
with explicit loads and supports, and you get back the number, where it happens,
a confidence level with its reasons, the assumptions, a heatmap with the failure
point marked and fixes with numbers.

- **Strength** with [CalculiX](http://www.calculix.de): second order tetrahedral
  meshes from [gmsh](https://gmsh.info) and OpenCASCADE, refined around the peaks
  until the stress settles.
- **Heat**, steady and transient, also with CalculiX.
- **Circuits** with [ngspice](https://ngspice.sourceforge.io): LEDs, RC networks,
  buck converters, MOSFETs or any netlist.

`wobbl fix --apply` edits the part, adding fillets and gussets to any STEP file,
and runs the whole study again. The same tools work from your terminal and from
Claude Code, Codex, Cursor and Gemini CLI.

## Quick start

Wobbl needs Python 3.10 or newer. Until it is on PyPI, install it from GitHub:

```sh
pipx install "git+https://github.com/sohan-bhat/wobbl"
# or: uv tool install "git+https://github.com/sohan-bhat/wobbl"

wobbl setup     # solvers into ~/.wobbl, MCP server and skill for your agents
wobbl doctor    # check every piece
```

`wobbl setup` downloads a pinned micromamba and installs the solvers from
conda-forge into `~/.wobbl` (about 340 MB, no admin rights). Use
`wobbl setup --no-agents` for solvers only, or `--agents claude,cursor` to pick.

A first answer that needs no files:

```sh
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

Then your own part:

```sh
# holes, fillets, sharp corners and face ids
wobbl inspect part.step

# ask in plain words, or give the material, supports and loads yourself
wobbl part.step "will this 6061 bracket yield under 50 lbf at the tip?"
wobbl run part.step -m petg --fix "bolt holes d=5.5" \
  --load "250 N down at 0,-15,31 @ holes d=3.4 rigid"

# build the fixes (fillets, gussets, thicker walls) and run each again
wobbl fix wobbl-out/part --apply
```

Or a bundled example, which takes a few minutes:

```sh
git clone https://github.com/sohan-bhat/wobbl
cd wobbl/examples/shelf-bracket && ./run.sh
```

| Platform | CalculiX | ngspice |
|---|---|---|
| macOS Apple Silicon | native | Intel build under Rosetta 2 |
| macOS Intel | native | native |
| Linux x86_64 | native | native |
| Linux arm64 | native | not packaged; install it with your package manager and Wobbl finds it |
| Windows x64 | native | native |

## Example output

A printed PLA shelf bracket, screwed to a wall, carrying its share of an 8 kg
shelf. Every run writes `summary.txt` (what an agent reads), `report.html`,
`outcome.json`, heatmaps and `result.vtu` for ParaView.

```console
$ cd examples/shelf-bracket
$ wobbl run bracket.py -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
    --load "40 N down @ top and within 12 of 65,12.5,6" -o output
$ cat output/summary.txt
FAILS. Peak 52.8 MPa on the 0.5 mm inside fillet (face 3) at (6.2, 12.2, 6.1). The part cracks at about 0.95x this load.

Confidence: high. passed mesh convergence, force balance, hand calculation, singularities, bolt bearing, units and sanity
Max deflection 3.46 mm at (90, 18.8, 6).

Checks:
  [pass] mesh convergence: max principal stress settled: 44.2 -> 51.1 -> 53.4 -> 52.8 MPa over 4 meshes, last change 1.2%
  [pass] force balance: reactions balance the loads to 0.00011% (forces) and 1.8e-05% (moments)
  [pass] hand calculation: beam theory through the critical section gives 15.2 MPa nominal; FEA peak is 3.47x that, a stress concentration of about 3.5 from the local shape
  [pass] singularities: no sharp corner or support artifact is above the reported stress
  [pass] bolt bearing: worst hole of fix1 (d 4.5 mm) carries 13.1 N sideways and 50 N along its axis; bearing stress 0.485 MPa, safety factor 103.05 on bearing
  [pass] units and sanity: part 123 mm across, peak strain 0.015, max deflection 3.46 mm (2.8% of size): all plausible

Suggested fixes:
  - Stress piles up in the 0.5 mm fillet. A fillet of 3 mm (about half the 6 mm wall) spreads it out. ...
  - The peak sits where two walls meet and bend. A gusset as thick as the wall ...
```

`wobbl fix output --apply` then builds each fix and solves it again:

| Version | Peak stress | Safety factor (PLA, 50 MPa) | Verdict |
|---|---|---|---|
| As drawn, 0.5 mm corner | 52.8 MPa | 0.95 | fails |
| 3 mm fillet | 24.5 MPa | 2.04 | marginal (target 2.5) |
| Legs thickened to 10 mm | 21.5 MPa | 2.32 | marginal |
| Gusset across the corner | 10.3 MPa | 4.84 | holds |

<p>
  <img alt="The bracket as drawn: a thin line of high stress along the 0.5 mm inside corner, 52.8 MPa" src="examples/shelf-bracket/output/images/stress.png" width="49%">
  <img alt="The bracket with a gusset: the peak drops to 10.3 MPa near the gusset toe" src="examples/shelf-bracket/output/fix-gusset/images/stress.png" width="49%">
</p>

## An agent on its own versus Wobbl

Given a CAD file, a strength question and 15 minutes, a capable coding agent
working alone answered in **1 of 10** trials across six
parts. Its hand calculations on beam like parts were good, within 1 to 5%. On the
ribbed cast motor mount and the thin walled enclosure, the two parts that
actually fail, it delivered nothing in six trials: it kept writing its own finite
element code and hitting the same traps (meshers folding thin curved walls,
singular peaks at bonded bolts, load idealizations that flip the verdict).

Wobbl answers those parts in minutes with the trap handling built in:

| Part | Converged reference | Wobbl | Confidence |
|---|---|---|---|
| Cast motor mount, belt pull through four bolts | 124.3 MPa at the rib tip | 123.9 MPa, same spot | medium, and it says why: the verdict flips if the servo is treated as rigid (55.5 versus 124 MPa) |
| ABS enclosure shell pressed near a corner | 93 MPa, 1.93 mm deflection | 89.8 MPa, 1.91 mm | high |
| PA12 truss drone arm, thrust and torque | 11.5 MPa, 4.09 mm rise | 11.53 MPa, 4.08 mm | high |
| PLA bracket, sharp inside corner | about 78 MPa | 80.0 MPa | high |

The full comparison, with times and every trap, is in [docs/bakeoff.md](docs/bakeoff.md).

## Using it from agents

`wobbl setup` registers an MCP server and installs a skill for every coding
agent it finds, backing up each file it touches. `wobbl uninstall` removes
exactly what it added.

| Agent | MCP server | Skill or context |
|---|---|---|
| Claude Code | `claude mcp add --scope user wobbl` | `~/.claude/skills/wobbl/SKILL.md` |
| Codex | `[mcp_servers.wobbl]` in `~/.codex/config.toml` | `~/.codex/skills/wobbl/SKILL.md` |
| Cursor | `mcpServers.wobbl` in `~/.cursor/mcp.json` | `~/.cursor/skills/wobbl/SKILL.md` |
| Gemini CLI | extension at `~/.gemini/extensions/wobbl/` | `WOBBL.md` in the same extension |

The agent gets 13 tools: `inspect_geometry`, `mesh`, `new_study`,
`set_material`, `apply_load`, `add_fixture`, `solve_structural`, `verify`,
`render`, `suggest_fix`, `solve_thermal`, `solve_circuit` and `list_materials`.
Results are compact JSON with image and report paths. The skill teaches the
workflow and what never to skip: read the checks before the number, never quote
an excluded peak, make the supports match the real build.

```text
> Will this printed PLA shelf bracket hold its share of an 8 kg shelf?

agent: solve_structural(path="bracket.py", material="pla",
                        fixtures=["bolt holes d=4.5", "wall x=min"],
                        loads=["40 N down @ top and within 12 of 65,12.5,6"])
       -> verdict fails, 52.8 MPa on the 0.5 mm inside fillet, safety factor 0.95, confidence high
agent: suggest_fix(result_dir, apply=true)
       -> 3 mm fillet: 24.5 MPa (2.04); legs at 10 mm: 21.5 MPa (2.32); gusset: 10.3 MPa (4.84)
```

Strength studies take one to five minutes. Details, timeouts and the full
session are in [docs/agents.md](docs/agents.md).

## What it checks

Every strength answer runs these checks and turns them into a confidence level,
high, medium or low, always with the reasons. A result that failed a check is
never reported as high confidence.

- **Mesh convergence** over at least three meshes, refined around the peaks that matter.
- **Force and moment balance**: reactions must cancel the loads to 0.5%.
- **Hand calculation**: beam theory through the critical section, and a check that the stress concentration is physical.
- **Singularities**: sharp inside corners found from exact CAD normals; peaks there and at idealized supports are excluded, named and tracked.
- **Load introduction**: off part loads are solved both rigid and distributed, each converged at its own peak, and the spread is reported.
- **Bolt bearing** at bolted holes, in place of the idealized peak.
- **Units and sanity**: part size, peak strain and deflection.
- **Large deflection**: a nonlinear re-solve when the part bends a good fraction of its wall.

Heat answers check energy balance and a lumped hand calculation; circuit answers
compare with the textbook and run again at half the time step. Every study has a
time budget, and when it runs out the last finished mesh level comes back with
its checks. The same input gives the same answer on every run: meshing, wall
thickness sampling and the budget are all deterministic. See
[docs/verification.md](docs/verification.md).

## Validation

Eighteen cases with known answers run through the same pipeline you get. Every
case passes its tolerance, and every case whose reference is exact for the
modeled setup lands within 2%.

| Group | Cases | Largest error |
|---|---|---|
| Textbook and benchmark strength (Timoshenko, Kirsch, Lame, NAFEMS LE1 and LE10, torsion, self weight) | 8 | 1.54%, plate with a hole |
| Real parts against converged reference models | 5 | 4.26%, PLA bracket with a 6 mm fillet |
| Heat (slab conduction, lumped cooling) | 2 | 0.44% |
| Circuits (LED, RC, buck ripple) | 3 | 7.70%, buck ripple against an ideal formula that ignores switch and diode drops |

The full table is in [docs/validation.md](docs/validation.md). Run it yourself
with `wobbl validate` (about 30 minutes).

## Examples

Real products with real outputs (reports, heatmaps, summaries) in [`examples/`](examples):

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

## Documentation

- [Getting started](docs/getting-started.md): install, inspect, ask, fix
- [CLI reference](docs/cli.md): every command, the load, support and face selector syntax
- [Agents and MCP](docs/agents.md): what `wobbl setup` registers, the tools, timeouts
- [How it checks itself](docs/verification.md): every check and what sets the confidence
- [Validation](docs/validation.md): the 18 cases and their setups
- [Bake-off](docs/bakeoff.md): a coding agent on its own versus Wobbl
- [Architecture](docs/architecture.md): domains, solver adapters, verification, reports
- [Publishing](docs/publishing.md): release steps for PyPI
- [Changelog](CHANGELOG.md)

## Limits

- **Linear elastic materials.** Wobbl reports whether yield is reached, not what
  happens after. Large deflection is checked with a nonlinear re-solve.
- **Static loads.** Fatigue, impact and vibration are not modeled yet.
- **Rigid wall contact only** (push but not pull). Bolted joints are modeled as
  clamped holes, without preload.
- **One material per study.** Assemblies of touching bodies are bonded.
- **Thin walls are slow.** 2 to 3 mm shells on 100 mm parts need large
  tetrahedral meshes and take several minutes. Shell elements are on the roadmap.
- **Printed plastics** are treated as isotropic at 100 percent infill, with a
  layer strength knockdown when you give the print direction.

Next: shell elements, fatigue estimates and bolt preload in 0.2; CFD with
OpenFOAM in 0.3; motion with MuJoCo in 0.4.

## Contributing

Issues and pull requests are welcome. To work on Wobbl itself:

```sh
git clone https://github.com/sohan-bhat/wobbl && cd wobbl
python -m pip install -e ".[dev]"
python -m pytest -m "not solver"      # fast unit tests
wobbl setup --no-agents               # CalculiX and ngspice
python -m pytest -m solver            # end to end tests with the solvers
```

A new solver plugs in as one adapter class; see
[docs/architecture.md](docs/architecture.md). If you change the physics, add or update a
case in `src/wobbl/validation/cases.py` so `wobbl validate` covers it.

## License

[MIT](LICENSE), copyright Sohan Bhat.

---

<p align="center"><a href="https://sohan-bhat.github.io/wobbl/">sohan-bhat.github.io/wobbl</a></p>
