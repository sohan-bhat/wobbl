---
name: wobbl
description: Simulate anything before it is built, with Wobbl. Use when someone asks whether a physical part will hold, crack, bend, yield, overheat, buckle, resonate, survive a drop or a shaker test, or how many cycles it lasts (brackets, mounts, enclosures, drone arms, heat sinks, battery packs, composite layups), wants stress, deflection, safety factor or temperature numbers, asks whether a metal part bends for good, where a fatigue crack goes and when it fails the part, asks whether a circuit works or whether anything on a board burns (any SPICE netlist, KiCad schematic or circuit in words, from LED arrays and 555 timers to op amp stages, regulators, chargers and microcontroller boards, LED current, regulator ripple or MOSFET heating), asks how something dropped, thrown or launched moves (fall time, impact speed, range, or a rocket's apogee, stability margin and landing speed), designs a solid rocket motor, asks for a flame temperature, a propellant's Isp or what a reaction makes, asks whether a lithium ion cell can run away, whether servos can lift a load, how a linkage swings, what a gear train gives or whether something tips over, asks about pipe pressure drop, drag, propellers, drones or boat hulls, solenoid or magnet force, capacitance and breakdown, trace impedance and crosstalk, antenna range, room modes or muffler loss, or how likely a design is to fail and which tests are still needed. Wobbl reads the thing, works out the physics, runs real solvers (CalculiX, ngspice, GetDP, Cantera, MuJoCo, OpenFOAM and scipy), checks its own answer and shows the user their scenario playing out in 3D with a confidence level and fixes.
---

# Simulate it with Wobbl before it is built

Wobbl has one command for everything, `wobbl simulate` (MCP `simulate`). Hand it the thing and the
question in plain words, and it works out what the thing is, which physics the question needs and how to
set them up, runs each one on a real solver with its own checks, and says what it assumed and how to
override every guess. Use it instead of hand calculations, and never write your own finite element code,
circuit solver or physics loop.

## The flow

1. **Hand Wobbl the thing and the question.**
   `wobbl simulate bracket.step "will it hold 200 N and stay under 60 C at 12 W?"`
   (MCP `simulate` with `thing` and `question`). The thing can be a CAD file (STEP, STL or a parametric
   Python part), a SPICE netlist or KiCad schematic, a mechanism spec, URDF or MJCF model, a rocket or motor
   spec, a finished result folder, a solver case folder, a molecule, element, material or reaction as a name,
   formula, SMILES or equation, or a sentence that describes it, as in
   `wobbl simulate "a 9 cm rock dropped from 300 m"`. A question that asks several things runs several
   physics in one go, each with its own load case, so "will it survive a 1 m drop and hold a 10 kg shelf"
   runs a drop and a strength study side by side.
2. **Open the 3D view for the user straight away.** Every run ends with an interactive view of the user's
   own scenario playing out with the result on it, the wall and the shelf, the floor for a drop, the
   breadboard for a circuit. The CLI opens it in the browser and prints its path, and every MCP result says
   where it is as `view` (simulation.html for a simulate run). From an agent, open it at once with
   `open <path>` on macOS, `xdg-open <path>` on Linux or `start "" <path>` on Windows, or pass the result
   folder to wobbl view. Then tell the user what they are seeing, which props
   stand for the supports and loads, what plays out (the load coming on, the flight, the heat building) and
   where the failure shows if there is one.
3. **Check what Wobbl read and what it guessed.** The run prints the route it took, the values it read
   from the question and every guess, each with the option that replaces it, and ends with the explicit
   command that repeats the run with nothing guessed (MCP `decisions`, `assumptions` and `explicit`). When a
   guess does not match what the user said or built, run again with that option. Before a long run, add
   `--plan` (MCP `plan_only=true`) to see the route and every guess without solving, and `--estimate` to
   see the time, memory and disk it needs.
4. **Give Wobbl what you know.** Pass the material, loads and supports the user gave rather than leaving
   them to guesses. For a part, look first with `wobbl inspect bracket.step --json` (MCP
   `inspect_geometry`) for the bounding box (mm), holes, fillets, sharp inside corners, wall thickness and
   face ids, then write
   - loads: `"200 N down @ top"`, `"50 lbf -z @ x=max"`, `"0.2 MPa @ top"`, `"3 g down"`,
     `"300 N -x bearing @ hole d=6.6"` (a bolt or pin pressing on one side of the hole, where the
     same load without `bearing` spreads evenly over the hole faces),
     `"2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid"` (force at a point off the part, carried
     by the faces), `"0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25"`.
   - supports: `"holes d=4.5"` (fixed), `"bolt holes d=6.6"` (hole plus washer ring), `"x=min"`,
     `"wall x=min"` (can push, can lift off), `"roller y=0"`, `"fixed-z faces 7"` (only some directions),
     `"elastic 50 N/mm3 @ z=min"` (a bed of springs, for a mount that gives a little).
   - selectors: `top`, `bottom`, `x=min`, `z>10`, `normal -z`, `holes d<6`, `hole near 55,0,10`,
     `fillets`, `face 12`, `within 15 of 50,-25,35` (a circular patch), combined with `and` or `or`.

   They go in as `-m`, `--load` and `--fix`, as in
   `wobbl simulate bracket.step --physics stress -m 6061-t6 --fix "bolt holes d=6.6" --load "200 N down @ top" --target-sf 2 --json`
   (MCP `material`, `loads`, `fixtures`). Any other value a physics reads is a flag of its own, such as
   `--power "12 W"`, `--rpm "4000-9000 rpm"` or `--height "1.2 m"` (MCP `options`), and
   `wobbl simulate --list` (MCP `physics`) lists every domain, its physics and their options. Materials come
   from `wobbl materials` (pla, petg, abs, asa, nylon, pa12, pc, 6061-t6, 7075-t6, mild-steel, ss304,
   ti-6al-4v, wood and more, or `custom:E=3000,nu=0.35,yield=40`). Parametric Python parts (`PARAMS` plus
   `build(occ, **params)`) take `--param thickness=6`.
5. **Read the verification before the number.** Every result has `confidence` (high, medium, low),
   `confidence_reasons` and `checks`. Look at the images to confirm the peak is where you expect.
6. **If it fails or is marginal,** read the fixes and run `wobbl fix wobbl-out/bracket --apply` on the
   result folder the run printed (MCP `suggest_fix` with `apply=true`) to solve the part again with fillets,
   gussets, thicker walls or another material and show before and after.
7. **Report** the verdict, the governing number with units and location, the margin against the target,
   the confidence and its reasons, the key assumptions, and the view and report paths.

## What each kind of thing gets, and how to quote it

**Parts.** A CAD file gets static strength (the default), heat, vibration, buckling, fatigue, plasticity,
drops, composite layups, crack growth, contact between bodies and bolted joints, read from the question or
named with `--physics`. A thin walled part goes to a shell model on its own inside the strength study,
checked against a coarse solid.
Heat runs as `wobbl simulate pack.step "does it stay under 60 C at 12 W in a sealed box?"`, with the
cooling read from words like sealed box, a small fan or water cooling, or given as
`--film "5 W/m2K @ all"`, and `--duration "30 min"` makes it transient.
A drop runs as `wobbl simulate phone.step "will it survive a 1.2 m drop onto tile?" -m pc --mass "45 g @ 0,0,5"`.
Its options are height (a bare number is meters), floor (concrete, steel, tile, stone, wood, vinyl, carpet,
foam, grass, or layer:E=5,t=10 in MPa and mm), orientation (worst by default, which tries every face and corner
landing, or all, corners, edges, flat, or a direction such as -z), mass for a battery or board, and watch for
where the accelerometer sits. It answers with the peak stress away from the contact patch, the safety factor,
the g at the accelerometer filtered at 10 kHz and the worst landing, checked by momentum and energy.
A composite part runs as
`wobbl simulate arm.step "is it strong enough?" --physics composite --layup "[0/±45/90]s" --fix "x=min" --load "40 N +z @ x=max"`
with plies listed from the inner face out and a ply material of t300-5208, im7-8552, as4-3501-6 or
e-glass-epoxy. It gives the load factor at first ply failure by Tsai-Wu and Hashin, the ply and mode that go
first, and layup changes checked with lamination theory.
Buckling runs as `wobbl simulate strut.step "will it buckle under 2 kN?" -m 6061-t6 --fix "z=min" --load "2 kN -z @ z=max" --target-sf 3`.
It cuts the linear factor by an imperfection knockdown for the shape and by yielding. Struts follow the
European column curves, thin cylinders NASA SP-8007, flat plates keep their factor and shapes with no
published data take 0.5, which lowers the confidence. Quote the design factor rather than the raw linear one.
Vibration runs as
`wobbl simulate arm.py "does it resonate with the props?" -m pa12 --fix "bolt holes d=3.2 and x<0" --mass "60 g at 150,0,20 @ top and within 15 of 150,0,15" --rpm "4000-9000 rpm" --blades 2`
and judges the natural frequencies against the rotor's once per turn and blade pass bands, keeping every mode
that moves mass 20% clear by default. Give each motor, battery or payload as a mass, because it sets the low
modes. A shaker sweep and random vibration go in as `--sine "2 g z 10-500 Hz"` and
`--psd gevs-qual --axis z`, and loads with a band alone, as in `--sine "10-500 Hz"`, sweep as harmonic forces.
Fatigue runs on a part with its loads, as in
`wobbl simulate bracket.step "will it last a million cycles at 300 N?" -m 6061-t6 --fix "bolt holes" --ratio -1`,
or on a finished strength or vibration result, as in
`wobbl simulate wobbl-out/bracket "how many cycles at R=0?" --physics fatigue --ratio 0 --cycles 1e6`.
A duty cycle goes in as `--block "100% x 2e4" --block "60% x 1e6" --repeats 10`, and a measured load
history as `--history loads.csv`, counted by rainflow with each value a multiple of the studied load.
When a part reaches yield, or someone asks whether it bends for good or what load collapses it,
`wobbl simulate hook.py "will 3 kN bend it for good?" --physics plastic -m mild-steel --fix "bolt holes d=8.5" --load "3 kN -z @ within 5 of 22,0,-16"`
runs the plastic study. It finds the load multiple at first yield, applies and removes the load with the
material's true stress strain curve to measure the permanent set and peak plastic strain, and ramps an
elastic perfectly plastic copy until it collapses. A measured curve goes in `--curve`, as a tensile test
like `engineering 250@0.00125, 400@0.2`, as true stress at plastic strain, or as a CSV file. Its verdict
separates `holds`, `yields` (yields locally but carries the load), `marginal` and `collapses`. Quote the
collapse load and the permanent set, not the elastic peak, and check the ductility line. Ductile plastics
run with a caveat that lowers the confidence, and brittle materials are refused because they have no
plastic reserve.
For an assembly whose bodies touch, slide or are bolted, ask about contact or bolts and Wobbl keeps the
bodies apart where they meet, pairing every face that lies on another, as in
`wobbl simulate clamp.step "does the jaw slip on the rod under 2 kN?" --physics contact -m mild-steel --fix "body 1 and bottom" --load "2 kN +x @ body 2 and top"`.
Name faces of one body with `body 2 and top`. Bolts come from holes that run through two parts or more,
sized from the hole and tightened to 70% of their proof load unless told otherwise, as in
`wobbl simulate flange.step "does it stay closed under 30 kN?" --physics bolts -m mild-steel --fix "body 1 and bottom" --load "30 kN +z @ body 2 and top" --bolt "M8 8.8 torque 20 N*m"`,
and a hole narrower than its bolt is read as tapped. Each bolt is checked as VDI 2230 does, for proof
stress, opening, slip, head pressure and thread stripping, beside the hand calculation.
A fatigue crack runs as
`wobbl simulate lug.step "how long until a crack from the pin hole grows critical?" -m 7075-t6 --fix "bolt holes d=6.6" --load "4 kN 0.866,0.5,0 bearing @ hole d=8" --target-life 100000`,
or on a finished result, as in `wobbl simulate wobbl-out/bracket "how long until a crack grows critical?"`.
It grows the front step by step with a fresh mesh each time and integrates the NASGRO equation along it.
The crack is the damage tolerance flaw at the most stressed point unless `--crack` gives one, such as
`"through 1.27 mm at hole d=8"`, and `--r-ratio` or `--spectrum` give the cycles. Quote the nominal life
with B10, B1 and the inspection interval, never one part's failure cycle.

**Circuits.** Pass whatever the user has, a SPICE netlist, a KiCad schematic or a description in words,
as in `wobbl simulate amp.cir "does it work?"`, `wobbl simulate board.kicad_sch "will anything overheat?" --set J2.load=2A`
or `wobbl simulate "three red LEDs in parallel from a 9 V battery, each with 330 ohm"`. Wobbl judges every
part against its voltage, current, power and temperature ratings, says whether the circuit works, and solves
its fixes again before offering them. Parts named in a netlist (1N4148, 2N3904, IRLZ44N, LM358, NE555,
AMS1117-3.3 and many more) get their library models and ratings, and a note after a part's `;` gives its
package or rating, as in `R1 a b 220 ; 0805` or `C1 out 0 10u ; 16V X5R`. Set part values and ratings with
`--set`, as in `--set R1.power=1W` or `--set ambient=40C`, read a signal with `--probe "v(out)"`, and ask
for the operating point, the waveform over time (ripple, inrush, timing), the frequency response and
stability, or the spread across part tolerances and the supply range in the question or with `--physics`
(operating-point, transient, ac or tolerance). When a description would take more than a sentence, write
the netlist yourself and pass it. A board's trace parasitics come from the field physics below and feed
back into a converter run.

**Mechanisms.** Any URDF or MJCF model, a spec of bodies, joints, drives and a move, or words, as in
`wobbl simulate arm.urdf "can it hold 2 kg at full reach?"`, `wobbl simulate go1.xml "what does each foot carry, and is it stable?"`
or `wobbl simulate "can two mg996r servos lift 500 g with a robot arm?"`. It runs in MuJoCo with each drive
held to its torque and speed limit, and the question picks the physics: torque, kinematics (reach and range),
collision, stability, contact or dynamics. Each searches the whole range for its worst pose and checks itself
against a hand calculation. A model that names no motors takes a catalog servo on every driven joint, as in
`wobbl simulate arm.urdf "can it lift 500 g?" --servo mg996r`, a fixed base arm stands loose on a floor with
`wobbl simulate arm.urdf "will it tip over?" --base free`, and a model left to itself runs as
`wobbl simulate pendulum.xml "does it fall over?" --passive`. A CAD assembly needs a short spec naming each
part's solid in the STEP file (`cad = "arm.step"`, then `[[part]]` with `name`, `solid`, `parent`, `material`
and a `joint` with its `axis` and the point `at` it sits, in mm in the assembly's coordinates). Read the loader's notes on meshes it could not
read or masses it took from shapes, and read stall and heat joint by joint, since a servo that clears a
single move can still cook when the move repeats.

**Flights.** A body in words, a rocket spec or a solid motor spec, as in
`wobbl simulate "a 9 cm rock dropped from 300 m"`, `wobbl simulate rocket.toml "what is the apogee?"` and
`wobbl simulate motor.toml "does the casing hold the chamber pressure?"`. A flight goes through gravity that
falls off with altitude, the 1976 standard atmosphere, drag, wind, thrust and parachutes, checked against
closed forms, an energy balance and a delta v budget. A rocket's verdict comes from the stability margin,
the speed off the rail, apogee against its target and the landing speed, and every run writes
`trajectory.csv`. A motor design also runs from flags, as in
`wobbl simulate "solid motor" --physics motor --propellant knsb --grain "bates d=48 core=18 L=80 x4" --throat "13 mm" --exit "33 mm" --casing "6061-t6 od=54 wall=1.6 L=400"`,
and is judged on the casing's hoop stress at peak pressure, erosive burning and nozzle separation. The burning
comes from the propellant's burn rate law rather than resolved flame chemistry, so say so when quoting a
pressure. Wobbl models given designs only and gives no guidance on making or handling propellants.

**Fluids.** A pipe or duct, a body in a stream, a propeller or a hull, as in
`wobbl simulate "40 m of 25 mm pipe carrying 30 L/min of water, what is the pressure drop?"` or
`wobbl simulate apcsf_10x4.7 "can a 1.2 kg quad with 4 rotors hover?"`. Pipes follow Darcy Weisbach on the
Moody chart, drag comes from published curves (a result near the drag crisis comes back with low
confidence), props from their measured UIUC curves checked with blade element momentum theory, and hulls
from Holtrop and Mennen.

**Fields.** A solenoid, electromagnet or magnet, capacitor plates, a coax or a PCB trace, an antenna, a room
or a muffler, as in `wobbl simulate solenoid "does it pull 8 N over its stroke?" --turns 800 --current 0.6A --stroke 6mm --need 8N`
or `wobbl simulate "2.4 GHz dipole" "where does it resonate?" --frequency 2.4GHz --length 58mm`, solve with
GetDP, fetched the first time (an 11 MB download). A magnetic force comes from the Maxwell stress tensor and is
checked by virtual work, so read that check before quoting it. A patch antenna is sized with the transmission
line model rather than a full wave solve, so say so when you quote it.

**Chemistry.** A fuel, a reaction, a propellant or a cell, as in
`wobbl simulate "CH4 + 2 O2 -> CO2 + 2 H2O" "how hot is the flame?"`,
`wobbl simulate knsb "what Isp at 5 MPa?" --area-ratio 8` or `wobbl simulate 18650 "can it run away at 60 C?"`,
runs on NASA's thermodynamic data with Cantera, which installs itself the first time (about 6 MB). Runaway
kinetics are for a fully charged LiCoO2 cell, so say the answer is indicative for other chemistries.

**Evidence on any result.** When someone asks how likely a design is to fail, which input matters most,
whether the model matches their tests or which physical tests they can skip, build the evidence on a
finished result folder from any domain. `wobbl uncertainty wobbl-out/part --vary "load lognormal 10%" --vary "yield 8%"`
(MCP `quantify_uncertainty`) gives the probability of failure with its 95% interval, the spread of the
answer and Sobol indices that rank the inputs. Any number the run was made from can scatter, by the name
request.json gives it, written as a percentage, an `sd`, a `tol` band or a range, as in
`"thickness tol 0.2"` or `"film uniform 3..8"`. `wobbl credibility wobbl-out/part --influence high --consequence medium`
(MCP `assess_credibility`) maps the evidence to a V&V 40 style assessment and writes `credibility.html` for a
reviewer, MCP `validate_against_tests` compares the run with test readings as ASME V&V 20 does, and
`wobbl test-plan wobbl-out/part` (MCP `plan_tests`) writes the smallest set of tests that would most reduce
the risk.

**Big problems.** `--estimate` predicts a run's time, memory and disk before it starts, and
`--background` or `--remote NAME` run it as a job here or on a machine added with
`wobbl jobs remote add box me@box.local`. `wobbl jobs status` follows it, and `wobbl jobs cancel` and
`wobbl jobs resume` stop it and carry it on from its last checkpoint.

## Never skip

- **Open the view for the user.** They asked to see what happens, so show them their scenario.
- **Never quote a low confidence number as the answer.** Say what failed and rerun with
  `--quality fine` or a longer `--budget`, or fix the setup.
- **Never quote an excluded peak.** Peaks at sharp inside corners, at the edge of fixed faces and at
  rigid couplings are idealization artifacts, and Wobbl lists them under `singularities`. The real
  answer is the governing value. If the material is brittle and a sharp corner is flagged, recommend
  a fillet and run again.
- **Check units.** Files are read in mm. If the bounding box looks 25.4x or 1000x off, pass `--scale`.
- **Check the supports match reality.** A fixed face is perfectly rigid. Bolted holes, walls that
  can lift off and sliding faces each have their own type, so pick the one that matches the build.
- **Do not use `--quality fast` for a final answer**, since it is for quick looks.
- **State the assumptions** Wobbl returns, especially guesses it made from a plain question, and run
  again with the override whenever one is wrong.
- **Call a fatigue life an estimate.** Lives scatter by a factor of three to ten at the same stress, so
  lead with the safety factor on stress and say which S-N data and mean stress model were used.
- **Never quote a raw linear buckling factor as the margin.** Quote the design factor after the knockdown,
  and say which shape class set it.
- **Quote an impact's peak force as an estimate.** Rigid parts meet through a contact spring that Wobbl
  sets from Hertz theory, so the bounce and the momentum hold whatever the stiffness while the force
  follows it, as the contact softness check shows.
- **Quote a wireless range as a spread.** It comes from an empirical path loss model, so give the median and
  the 90% figure together with the environment it assumed.
- **Never call a model validated, certified or approved.** Wobbl supplies credibility evidence and
  proposes levels, and the reviewer, customer or regulator decides what it is enough for. Say plainly
  when readings are synthetic, and quote a failure probability with its interval.
