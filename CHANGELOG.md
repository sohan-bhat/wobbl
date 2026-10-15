# Changelog

## Unreleased

**Any circuit**
- `wobbl circuit` and `wobbl simulate` take any circuit: a SPICE netlist, a KiCad
  schematic or netlist export, or a description in words, with the LED, RC, buck
  and MOSFET templates kept as shortcuts. KiCad schematics are traced as eeschema
  connects them, symbols map onto parts by library, value and part number, pins by
  name, and a rail nothing drives is held at the voltage its name gives.
- A parts library of close to two hundred part numbers (diodes, LEDs, zeners,
  transistors, MOSFETs, linear regulators, op amps and comparators, the 555,
  chargers, the TL431, optocouplers, microcontrollers, batteries, motors and relay
  coils) gives each part a model fitted to its datasheet, its ratings, its package
  and its pinout. A part it does not know gets a generic model, and the report says
  so.
- The analyses follow the circuit and the question: an operating point, a power on
  transient run until it settles, the frequency response and each op amp's phase
  margin by Bode's return ratio, and the supply range with Monte Carlo builds on the
  parts' tolerances.
- Every part is judged against its voltage, current, power and temperature ratings,
  with regulators in dropout, overvolted microcontrollers, half on MOSFETs and
  oscillating loops called out as not working. Fixes use E24 values and library
  parts and are solved again before they are offered.
- Each run checks KCL at every node, the power account, half the time step or
  tighter tolerances, closed form hand calculations wherever a piece reduces to one,
  and where every model came from. Every run writes `playback.json` with the netlist
  and every part's voltage, current and power over time, and hands its vectors to
  the bench and schematic drawings. Board parasitics join the netlist through
  `join_parasitics`.
- Six new validation cases send netlists through the same study, and 44 circuits
  Wobbl was not tuned on all simulate, 41 of them with the verdict an engineer gave
  them beforehand.

**One command for any thing**
- `wobbl simulate THING "question"` takes a CAD part, a netlist or KiCad
  schematic, a mechanism, rocket or motor spec, a fluid case, a result folder, a
  molecule, material or reaction, or a sentence, works out the domain and the
  physics from the question or `--physics`, and runs several physics in one go
  when the question asks several things, each with its own load case.
- Every guess prints with the option that overrides it, `--plan` shows them all
  without solving, and each run ends with the explicit command that repeats it
  with nothing guessed, `request.json` beside every result and a 3D view.
- Physics plug in through a `domain.py` in their package that registers what they
  read and which question words ask for them, with no new command or tool. A
  routing test sends over a hundred things and questions through every domain.
- The help lists `simulate` and the general verbs only. `run`, `thermal`,
  `circuit`, `motion`, `mechanism`, `flow` and the other domain commands still
  work as hidden aliases. The MCP server leads with `simulate` and `physics`, and
  its `solve_*` tools stay as deprecated aliases. The skill teaches the one flow.

**Lighter install**
- Wobbl no longer needs pyvista or VTK. Heatmaps, mode shapes, shell and crack
  pictures come from a small numpy rasterizer with matplotlib labels, so a server
  with no display or OpenGL draws the same pictures as a laptop, and section cuts,
  probes, clipping and surface decimation are numpy too. A fresh environment with
  Wobbl and its dependencies goes from 1006 MB to 392 MB.
- `wobbl setup` moves aside what no solver run needs, checks the solvers still
  run, and deletes its downloads, and CalculiX uses the reference BLAS. A fresh
  `~/.wobbl` goes from 224 MB to 66 MB on an Intel Mac. `WOBBL_KEEP_CACHE=1` keeps
  the downloads.
- Reports are set in Archivo and IBM Plex Mono, embedded as 27 kB of subset woff2,
  and the new pictures are smaller, so the shelf bracket's report goes from
  1.36 MB to 0.98 MB.
- `result.vtu` is written by a small XML writer with 32 bit fields, and the shelf
  bracket's goes from 6.9 MB to 4.4 MB. ParaView opens it as before.

**Mechanisms**
- Any URDF, MJCF or spec runs through `wobbl simulate`, which picks the physics from
  the question: torque against each drive's limits, kinematics (reach and range),
  collision, stability, contact and dynamics. Each searches the whole joint range
  for its worst pose, closes loops by Newton steps and checks itself against a hand
  calculation, and a model left to itself is checked for balance by linearizing
  gravity against its mass matrix.
- Model files load the way people share them: `package://` meshes are found,
  unreadable or missing meshes are dropped with their mass kept, links with no mass
  take it from their shape, ROS and Drake extras are ignored and a home keyframe is
  the starting pose, each written down in the report.
- 27 public robot models, from MuJoCo Menagerie, MuJoCo and DeepMind Control,
  Gymnasium, PyBullet, the ROS URDF tutorials and Drake, run and pass their own
  checks in a test anyone can run with `WOBBL_ZOO=1`.
- `wobbl mechanism` runs robot arms, four bar linkages, gear trains, impacts and
  things that tip in MuJoCo, from a template with `--set` values, a TOML spec of
  bodies, joints, drives, couplings, loops, loads and a move, or an MJCF or URDF
  model. Motors follow a straight torque speed line, and servos follow their move
  by computed torque up to it.
- An arm's servos are judged for a stall against their line and for heat by their
  RMS torque over the cycle, a linkage gets its sweep, toggle positions,
  transmission angle and mechanical advantage, a gear train its output speed and
  torque with mesh losses, an impact its peak force and a stand its tipping point.
  Every fix is solved again before it is suggested.
- Each run is checked against a closed form wherever one applies, at half the time
  step and with an energy account over every force, and its contact depth and
  constraint drift are inspected. Runs write `trajectory.csv` and `playback.json`
  for replaying the run.
- MuJoCo installs itself on first use from its PyPI wheel, 7.8 MB to fetch and
  18.3 MB on disk after pruning, or with `wobbl setup --only mujoco`. Agents get
  `solve_mechanism`, and plain questions about servos, linkages, gearboxes and
  tipping reach the new domain.
- Six validation cases against exact answers, all within 0.05%, and two examples,
  a desktop arm on hobby servos and a windshield wiper linkage.

**Buckling, vibration and fatigue**
- `wobbl buckling` finds the load factor and mode shapes with CalculiX's
  buckling step and cuts the factor by an imperfection knockdown for the shape,
  the European column curves for struts, NASA SP-8007 for thin cylinders, none
  for flat plates and 0.5 for shapes with no test data, and by yielding. It
  checks Euler's column, force balance, mode separation and the meshes.
- `wobbl vibration` finds natural frequencies, mode shapes and effective mass
  with attached masses, judges them against rotor speeds and tones, gives the
  peak stress of a sine sweep (checked against CalculiX's steady state dynamics
  step) and the 3 sigma stress and fatigue of random input such as NASA GEVS.
  Each answer checks the mass, Rayleigh's quotient and a single mode estimate.
- `wobbl fatigue` gives the cycles before a crack starts and a safety factor on
  stress from a strength or vibration result or from given stresses, with
  sourced S-N data per material, Marin factors, notch sensitivity, Goodman,
  Gerber or Soderberg and Miner's rule for a duty cycle.
- All three take any part with any supports and loads, walls included. A sine
  band alone sweeps the loads themselves as harmonic forces, with modal forces
  from CalculiX's static solution under them. Fatigue reads a measured load
  history by rainflow counting (ASTM E1049), and runs the strength study first
  when it is given a part rather than a result. Loads that only stretch a part
  get a plain answer that it does not buckle.
- They run through `wobbl simulate`, with the old commands kept, and Python
  callers get `wobbl.life.api.buckling`, `vibration` and `fatigue`.
- Each run writes a 3D scene its report opens with: the mode shapes that matter
  oscillating in turn, the buckling shape growing as the load climbs, and the
  fatigue damage building up on the part when its strength result kept its
  field.
- Every element's mass is integrated over its curved shape, which coarse meshes
  of round parts need, and every study reaches a second mesh to check
  convergence, thin walls and parts full of holes included.
- A sweep over all eight example parts and eight parts nobody tuned for (thin
  sheet, chunky block, hook, thin can, dome, shaft, an STL mesh and two touching
  bodies) runs modes, swept loads, buckling and two kinds of fatigue on each.
- Agents get `solve_buckling`, `solve_vibration` and `fatigue_life`. Nothing new
  is downloaded, since they run on the CalculiX strength uses.
- Six validation cases, Euler's column, a cantilever's and a plate's modes, a
  sine peak on a tip mass, a force swept through resonance and Shigley's fatigue
  example, and three examples, a drone arm's resonance, a strut's buckling and a
  motor mount's fatigue, each with a `playback.json` for the website to animate.

**Rocket motors and combustion**
- `wobbl motor` burns a solid motor design back to its thrust curve, from
  BATES, tube, end burner, star, finocyl, moon and X core grains, KNSB, KNDX,
  KNSU and MIT APCP propellants with their published burn rate laws, and a
  nozzle. It checks the time step, the mass balance, the closed form pressure
  and the propellant's c* against equilibrium, judges the casing's hoop stress
  at peak pressure, the port to throat ratio, core mass flux and nozzle
  separation, simulates each fix and writes `thrust.csv` and a RASP
  `motor.eng`. A rocket spec for motion flies a design with
  `design = "motor.toml"`.
- `wobbl flame` and `wobbl propellant` give flame temperatures and CEA style
  c* and Isp from Cantera on NASA's thermodynamic data, and agents get
  `solve_motor`, `solve_flame` and `solve_propellant`.
- Cantera installs itself on first use from its PyPI wheel, 6.3 MB to fetch
  and 15.7 MB on disk after pruning, or ahead of time with
  `wobbl setup --only cantera`.
- Eight new validation cases, against Turns, NASA CEA, Nakka's PROPEP run,
  closed forms and two MIT static fires, and two examples, a KNSB motor and a
  propane flame.
- `wobbl runaway` finds the surrounding temperature at which a charged
  lithium ion cell's own decomposition outruns its cooling, by Semenov's
  criterion and a transient with the reactants used up, and can take its
  surroundings from a heat study. A LiCoO2 18650 runs away at 156 C against
  the published 155 C, and a new example checks the battery pack's hottest
  cell. Agents get `solve_runaway`.
- Every motor, flame, propellant and runaway run writes `playback.json`, its
  own data thinned for a viewer to animate.

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

**Credibility evidence**
- `wobbl uncertainty` carries load, material, geometry and boundary condition
  scatter through a finished run and returns the probability of failure with
  its 95% interval, the spread of the answer and Sobol indices. Strength runs
  scale exactly, several loads scatter one at a time through an exact sum of
  one solve per load on the final mesh, and Poisson's ratio, geometry, heat and
  circuit inputs get new solves and a polynomial chaos surrogate with its leave
  one out error.
- `wobbl validate-against` compares a run with gauge, temperature, circuit or
  scope readings as ASME V&V 20 does, adds the area metric, calibrates the
  uncertain inputs and says which load attachment the readings favour.
- `wobbl credibility` writes a V&V 40 style assessment into `report.html` and
  a standalone `credibility.html`, and `wobbl test-plan` names the fewest
  physical tests that most reduce the risk. Wobbl supplies the evidence and
  certifies nothing.
- Elastic supports, as in `--fix "elastic 50 N/mm3 @ z=min"`, hold a face on
  a bed of springs, so a mount that gives a little can be modeled and its
  stiffness can scatter. A cantilever on an elastic root matches beam theory
  on a Winkler bed to 0.01%.
- Four new agent tools and a skill section, six validation cases,
  and the cast mount credibility example with clearly synthetic readings.

**Crack growth**
- Crack growth follows a fatigue crack through any part. Ask
  `wobbl simulate part.step "how long until a crack from the hole grows
  critical?"`, or start from a finished strength or fatigue result, whose
  part, material, loads, supports and governing point it reuses, and a
  fatigue result's duty cycle and required life with them. `wobbl crack` and
  the `grow_crack` tool stay as the direct way in.
- The starting crack is automatic unless one is given: the damage tolerance
  flaw of MIL-A-83444 and JSSG-2006 at the most stressed point, a corner crack
  at a hole, a surface crack elsewhere or a crack through a thin wall, made
  smaller when the part leaves no room. A through, corner, surface or center
  crack can also start at any faces or point.
- The crack is cut into the CAD and gmsh meshes it with a fine tube along the
  front and quarter point elements. K_I, K_II and K_III come from the opening
  of the crack faces, and every front point turns by the maximum tangential
  stress angle and moves at its own growth rate, with a fresh mesh each step,
  so the path comes from the geometry. The path ends when the crack cuts
  through the wall it grows across.
- Growth follows the NASGRO equation with threshold and toughness, or Paris,
  with sourced constants for 7075-T6, the common aluminum alloys and three
  classes of steel. Other metals get BS 7910's steel law scaled by their
  stiffness, flagged as estimated, and plastics take measured constants.
  Loads cycle at an R ratio, as a file of cycles or as blocks such as
  `100% x 3650 R=0`, and Wheeler retardation is used when its exponent is
  given.
- The K table from the path is the surrogate for a Monte Carlo over the
  starting flaw, the growth constants, the load and the toughness, so 20,000
  parts take seconds. Results give B50, B10 and B1 with their sampling bands,
  the chance of failing before a target life and damage tolerance inspection
  times, with fixes for the stress at the origin, the material and the
  inspection interval.
- Each run checks K on a coarser front mesh, against a weight function or
  Newman and Raju, the force balance, the integration step, the Monte Carlo
  sample count and, when retardation is on, a cycle by cycle reference. It
  writes `playback.json`, a compact record of the part, every front and the
  spread of lives for a viewer to animate.
- Five validation cases, a center and an edge crack against Tada, a surface
  crack against Newman and Raju, Paris life against its closed form and the
  turn of a crack at 45 degrees, a run over the strength and fatigue examples
  it was not tuned for, and the bracket crack example.

**Magnetic, electric, sound and board fields on your own geometry**
- `wobbl simulate` runs fields on any CAD file and on real boards. Each body of
  a STEP, IGES, BREP, STL or parametric part gets a material from `--assign`
  lines such as `plunger: steel 1010`, `coil: coil 800 turns 0.6 A`, `magnet:
  n42 +z`, `electrode: conductor 5 kV`, `spacer: ptfe` or `box: rigid`, or from
  its name and the question, every guess shown with its override.
- Magnetics solve the vector potential on the edges of a 3D mesh with nonlinear
  steel, magnets in any direction and coils of any shape, whose current follows
  the winding's own path. The force and torque on a body come from the Maxwell
  stress tensor and are checked by virtual work on a stretched mesh, by action
  and reaction across all bodies and by the energy balance.
- Electrostatics give the capacitance matrix, the peak field per region and a
  breakdown margin, with charge, reciprocity and energy checks. Sound finds the
  modes of the air in a room or inside an enclosure, cut out of the CAD, and
  the level a vibrating body makes in free air.
- A KiCad (4 to 10) or Gerber board is read with its stackup, tracks, vias,
  pads and pours. Every net's traces are solved as cross sections over the
  planes really under them, coupled neighbors give crosstalk, and the result
  writes `parasitics.json` for the circuit run of the same board. Given its
  netlist, the board study runs it with and without the layout.
- The parametric solenoid, plates, trace, dipole and room studies stay for
  devices described in words, and the old field commands are hidden aliases.
- GetDP installs itself the first time a field study runs (11 MB download,
  26 MB kept). Every run writes `playback.json` with the fields on a slice.
- A generality run over 14 real boards and varied CAD scenes, and examples of a
  solenoid actuator, a buck converter's layout on two boards and a speaker box.

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
- Every circuit run draws its bench, the circuit wired up with the run's own values
  and readings in `setup.svg`, and its schematic in `schematic.svg`, both in the
  report. Any netlist is laid out on a breadboard as plain data the 3D viewer can
  share, and the website plays each bench with the measured current moving along
  its wires.

**Agents**
- `wobbl setup` installs CalculiX and ngspice into `~/.wobbl` with a pinned
  micromamba and registers an MCP server and skill for Claude Code, Codex, Cursor
  and Gemini CLI, with backups and `wobbl uninstall`. A test parses every
  command in the skill against the real CLI.
- A validation suite of 18 cases, with textbook solutions, NAFEMS LE1 and LE10,
  and five real parts against converged reference models. It runs in about
  30 minutes on a 2018 six core laptop.
