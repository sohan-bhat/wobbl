# Changelog

## Unreleased

**Mechanisms**
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
- Agents get `solve_buckling`, `solve_vibration` and `fatigue_life`. Nothing new
  is downloaded, since they run on the CalculiX strength uses.
- Five validation cases, Euler's column, a cantilever's and a plate's modes, a
  sine peak on a tip mass and Shigley's fatigue example, and three examples, a
  drone arm's resonance, a strut's buckling and a motor mount's fatigue, each
  with a `playback.json` for the website to animate.

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
- Four new agent tools and a skill section, five validation cases,
  and the cast mount credibility example with clearly synthetic readings.

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
