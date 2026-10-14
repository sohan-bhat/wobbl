---
name: wobbl
description: Run verified engineering simulations on CAD parts, circuits and moving bodies with Wobbl. Use when someone asks whether a physical part will hold, crack, bend, yield, overheat or fail (brackets, mounts, enclosures, drone arms, heat sinks, battery packs), whether it buckles, resonates near a motor or propeller speed, survives a shaker or random vibration test or how many cycles it lasts in fatigue, wants stress, deflection, safety factor or temperature numbers, asks about LED current, regulator ripple or MOSFET heating, or asks whether a metal part bends for good or collapses past yield, or asks how something dropped, thrown or launched moves (fall time, impact speed, range, or a rocket's apogee, stability margin, top speed, landing speed and drift), designs a solid rocket motor (chamber pressure against the casing, thrust curve, total impulse and class), asks for a flame temperature or a propellant's c* and Isp, asks whether servos can lift a load or a robot arm can make a move, how far a linkage swings and where it binds, what a gear train gives, how hard two parts hit or whether something tips over, or how likely a part is to fail under load and material scatter, whether a model matches test readings and which physical tests are still needed, or asks about solenoid, electromagnet or magnet force, capacitance and breakdown, PCB trace impedance, crosstalk and switch node ringing, antenna tuning and wireless range, room modes, muffler loss or panel noise. Wobbl meshes the CAD, runs CalculiX, ngspice, GetDP, Cantera, MuJoCo or a scipy integrator, checks its own answer and returns a picture, a confidence level and fixes.
---

# Wobbl: simulate before you build

Wobbl wraps real solvers (gmsh, CalculiX, ngspice, MuJoCo, and scipy for motion) behind
one CLI and an MCP server. It refines the mesh until the answer stops moving, checks equilibrium,
compares against beam theory, flags stress singularities and reports a
confidence level. Use it instead of hand calculations for real geometry, and
never write your own finite element code.

## Workflow

1. **Look at the part first.** `wobbl inspect part.step --json` (MCP: `inspect_geometry`).
   Note the bounding box (units are mm), holes with diameters, fillets, sharp inside corners
   and wall thickness. Use these to write selectors.
2. **Decide loads and supports explicitly.** Ask the user if they are unclear. Write them as:
   - loads: `"200 N down @ top"`, `"50 lbf -z @ x=max"`, `"0.2 MPa @ top"`, `"3 g down"`,
     `"300 N -x bearing @ hole d=6.6"` (a bolt or pin pressing on one side of the hole, where the
     same load without `bearing` spreads evenly over the hole faces),
     `"2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid"` (force at a point off the part, carried
     by the faces), `"0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25"`.
   - supports: `"holes d=4.5"` (fixed), `"bolt holes d=6.6"` (hole plus washer ring), `"x=min"`,
     `"wall x=min"` (can push, can lift off), `"roller y=0"`, `"fixed-z faces 7"` (only some directions).
   - selectors: `top`, `bottom`, `x=min`, `z>10`, `normal -z`, `holes d<6`, `hole near 55,0,10`,
     `fillets`, `face 12`, `within 15 of 50,-25,35` (a circular patch), combined with `and` / `or`.
   Parts can also be parametric Python files (`PARAMS` plus `build(occ, **params)`). Pass
   `--param thickness=6` to change them, and `wobbl fix --apply` changes their thickness and
   fillet parameters directly.
3. **Run the study.**
   `wobbl run part.step -m 6061-t6 --fix "bolt holes d=6.6" --load "200 N down @ top" --target-sf 2 --json`
   (MCP: `solve_structural`). Materials: `wobbl materials` (pla, petg, abs, asa, nylon, pa12, pc,
   6061-t6, 7075-t6, mild-steel, ss304, ti-6al-4v, wood, and more; or `custom:E=3000,nu=0.35,yield=40`).
4. **Read the verification before the number.** Every result has `confidence` (high, medium, low),
   `confidence_reasons` and `checks`. Look at the `stress` image (`images.stress`) to confirm the
   peak is where you expect.
5. **If it fails or is marginal,** read `fixes` and run `wobbl fix wobbl-out/part --apply` with the
   result folder the run printed (MCP: `suggest_fix` with `apply=true`) to re-run with fillets,
   gussets, thicker walls or another material and show before and after.
6. **Report** to the user: the verdict, the governing number with units and location, the safety
   factor against the target, the confidence and its reasons, the key assumptions, the image path and
   the HTML report path.

For heat, run `wobbl thermal part.step -m 6061-t6 --power "12 W" --film "5 W/m2K @ all" --ambient "25 C"`
(MCP `solve_thermal`). The material is required, and adding `--duration "30 min"` makes the run transient.
For circuits, give each template value its own `--set`, as in
`wobbl circuit led --set supply=5V --set resistor=150 --set led=red`, or run your own netlist with
`wobbl circuit board.cir --probe "v(out)"` (MCP `solve_circuit`).

For anything that falls, flies or launches, use motion (MCP `solve_motion`) instead of writing your own
integrator. It flies the body through gravity that falls off with altitude, the 1976 standard
atmosphere, drag, wind, thrust and parachutes, and checks the flight against closed forms, an energy
balance and a delta v budget. A dropped or thrown body is one line, as in
`wobbl motion --name rock --diameter "9 cm" --density "2700 kg/m3" --cd 0.6 --height "300 m"` or
`wobbl motion --name ball --mass "145 g" --diameter "7.4 cm" --speed "40 m/s" --angle 35`.
A rocket is a TOML spec with its motor's thrust curve, nose, fins, balance point, rail and
parachutes, run with `wobbl motion rocket.toml`, and the MCP tool also takes the TOML as text. For a
rocket the verdict comes from the stability margin, the speed off the rail, apogee against its target
and the landing speed, and the fixes give the nose weight, fin span, rail length or main parachute that
would pass. Every run writes `trajectory.csv` with one row at every event.

For air and water flow, use the flow commands (MCP `solve_flow`) instead of guessing a thrust, a drag
coefficient or a resistance. A drone or propeller question is one line, as in
`wobbl flow prop apcsf_10x4.7 --mass "1.2 kg" --rotors 4`, which finds the hover speed and shaft power
from the prop's measured UIUC curves, checks them with blade element momentum theory and judges the
thrust to weight ratio at the prop's rated top speed. Give the size in inches for a prop outside the
bundled set, as in `wobbl flow prop 10x4.7 --rpm 6000`. A displacement boat's calm water resistance
and engine power come from Holtrop and Mennen with
`wobbl flow hull --length "12 m" --beam "3.6 m" --draft "0.9 m" --displacement "16 t" --speed "8 kn" --power "60 kW"`.
Pressure drop through a pipe or duct with its fittings comes from Darcy Weisbach on the Moody chart, as in
`wobbl flow pipe --length "40 m" --diameter "25 mm" --flow "30 L/min" --fittings "3 elbow90, gate-valve" --max-dp "50 kPa"`,
and drag on a sphere, cylinder, plate or cube from published drag curves, as in
`wobbl flow drag sphere --size "22 cm" --speed "20 m/s"`. A drag result near the drag crisis comes back
with low confidence, because roughness and turbulence decide it there.

For a solid rocket motor design, run `wobbl motor motor.toml` (MCP `solve_motor`) instead of working the
ballistics by hand. The spec lists the propellant (knsb, kndx, knsu, ocean-water or cherry-limeade, or its
own burn rate and combustion data), the grains from the forward end (bates, tube, endburner, star, finocyl,
moon or xcore), the nozzle and the casing, as in examples/knsb-motor/motor.toml, and the same design also
runs as flags, as in
`wobbl motor --propellant knsb --grain "bates d=48 core=18 L=80 x4" --throat 13 --exit 33 --casing "6061-t6 od=54 wall=1.6 L=400"`.
Wobbl burns the grains back, integrates chamber pressure and thrust, and checks the time step, the mass
balance, the closed form pressure at the peak and the propellant's c* against Cantera's equilibrium. The
verdict comes from the casing's hoop stress at peak pressure against its yield, the port to throat ratio
and core mass flux for erosive burning, and nozzle separation, and each fix is simulated with its new peak
pressure and impulse. The run writes `thrust.csv` and a RASP `motor.eng`, and a rocket spec for motion can
fly the design directly with `design = "motor.toml"` under `[motor]`. For a flame, run
`wobbl flame methane --phi 1` or `wobbl flame propane --phi 0.9 --limit "1800 C"` (MCP `solve_flame`) for
the adiabatic flame temperature and the equilibrium CO and NO. For propellant performance, run
`wobbl propellant knsb --pressure "5 MPa" --area-ratio 8` or give a composition, as in
`wobbl propellant "ap:70,al:16,htpb:14" --pressure "6.9 MPa" --area-ratio 10` (MCP `solve_propellant`),
for c* and Isp in shifting and frozen flow the way NASA's CEA reports them. Cantera installs itself the
first time (about 6 MB). The burning in a motor comes from the propellant's burn rate law rather than
resolved flame chemistry, so say so when quoting a pressure, and keep the casing's safety factor margin.
Wobbl models given designs only and gives no guidance on making or handling propellants.

When someone asks how likely a part is to fail, which input matters most, whether the model matches
their tests, or which physical tests they can skip, build the evidence on a finished run's folder.
`wobbl uncertainty wobbl-out/part --vary "load lognormal 10%" --vary "yield 8%" --vary "E 3%"`
(MCP `quantify_uncertainty`) gives the probability of failure with its 95% interval, the spread of the
answer and Sobol indices that rank the inputs. Scatter is written as a percentage, an `sd`, a `tol`
band or a range, as in `"thickness tol 0.2"` or `"film uniform 3..8"`. Load, stiffness and strength
scatter on a strength run scale the solution exactly and take seconds, while geometry, heat and circuit
inputs need new solves and a polynomial chaos surrogate whose leave one out error comes back with the
answer. `wobbl validate-against wobbl-out/part readings.csv` (MCP `validate_against_tests`) compares the
run with strain, displacement, temperature or circuit readings as ASME V&V 20 does, adds the area metric
and calibrates the uncertain inputs. `wobbl credibility wobbl-out/part --influence high --consequence medium`
(MCP `assess_credibility`) maps all of it to a V&V 40 style assessment, writes `credibility.html` for a
reviewer and a test plan, which `wobbl test-plan wobbl-out/part` (MCP `plan_tests`) also writes alone.

For buckling, vibration and fatigue, use the life commands instead of hand formulas.
`wobbl buckling strut.step -m 6061-t6 --fix "z=min" --load "2 kN -z @ z=max" --target-sf 3`
(MCP `solve_buckling`) finds the factor on the load at which the part buckles and its mode shapes, then
cuts it by an imperfection knockdown for the shape and by yielding. Struts follow the European column
curves, thin cylinders NASA SP-8007, flat plates keep their factor and shapes with no published data take
0.5, which lowers the confidence. Quote the design factor rather than the raw linear one, because linear
buckling overpredicts real collapse, most of all for thin shells.
`wobbl vibration arm.py -m pa12 --fix "bolt holes d=3.2 and x<0" --mass "60 g at 150,0,20 @ top and within 15 of 150,0,15" --rpm "4000-9000 rpm" --blades 2`
(MCP `solve_vibration`) finds the natural frequencies and judges them against the rotor's once per turn
and blade pass bands, keeping every mode that moves mass 20% clear by default. Give each motor, battery or
payload as a mass, because it sets the low modes. A shaker sweep and random vibration go in as
`wobbl vibration bracket.step -m 6061-t6 --fix "bolt holes" --sine "2 g z 10-500 Hz"` and
`wobbl vibration bracket.step -m 6061-t6 --fix "bolt holes" --psd gevs-qual --axis z`, which give the peak
and the 3 sigma stress and, for random input, a fatigue estimate over the test.
`wobbl fatigue wobbl-out/bracket --ratio 0 --cycles 1e6` (MCP `fatigue_life`) turns a finished strength
result into the cycles before a crack starts and a safety factor on stress, a duty cycle goes in as
`wobbl fatigue wobbl-out/bracket --block "100% x 2e4" --block "60% x 1e6" --repeats 10`, and stresses you
already know run as `wobbl fatigue --material 4140 --amplitude "200 MPa" --mean "100 MPa" --kt 2 --radius "1 mm"`.
It reads a vibration result folder the same way. Crack growth after a crack starts is a separate question.

When a metal part reaches yield in a strength run, or someone asks whether it will bend for good or
what load would collapse it, run the plastic study with the same loads and supports, as in
`wobbl plastic hook.py -m mild-steel --fix "bolt holes d=8.5" --load "3 kN -z @ hole d=20"` (MCP
`solve_plastic`). It finds the load multiple at first yield, applies and removes the load with the
metal's true stress strain curve to measure the permanent set and peak plastic strain, and ramps an
elastic perfectly plastic copy until it collapses. The verdict separates `holds` (stays elastic),
`yields` (yields locally but carries the load), `marginal` (collapse below the target margin, 1.5 by
default) and `collapses`, and the fixes are aimed at the hinge where it gives way. Quote the collapse
load and the permanent set, not the elastic peak, and check the ductility line, since a peak plastic
strain near the metal's elongation means it may tear first. Plastics and brittle materials are refused,
so judge them with a strength run.

For mechanisms, use the mechanism command (MCP `solve_mechanism`) instead of working torques by hand or
writing a physics loop. It runs MuJoCo with each motor's torque speed line, compares the answer with a
closed form wherever one exists, runs it again at half the time step, closes an energy account and solves
every fix again before suggesting it. Whether hobby servos lift a load through a pick and place is
`wobbl mechanism arm --set servo=mg996r --set payload=500g`, with `shoulder_servo`, `links`, `pick`,
`place` and `move_time` set to match the build, and `--set mount=free` also checks whether the base tips.
A four bar linkage such as a wiper is
`wobbl mechanism fourbar --set ground=120 --set crank=30 --set coupler=125 --set rocker=45 --set load=2`,
which gives the sweep, the transmission angle, the dead points and the torque at the crank. A gear train is
`wobbl mechanism gears --set teeth=12:36,10:40 --set motor=fa-130 --set load=0.005`, and impacts and
tipping are `wobbl mechanism impact --set mass=200g --set drop=500mm --set material=pla --set limit=300`
and `wobbl mechanism tip --set payload=500g --set reach=120 --set force=3`. Anything else goes in a TOML
spec with bodies, joints, drives, couplings, loops, loads and a move, or in an MJCF or URDF model, and runs
as `wobbl mechanism robot.toml`. Bare numbers are mm, g, degrees, N m, rpm and seconds, and servo ratings
read as written, like `11 kg*cm` or `0.14 s/60deg`. Read the stall and heat numbers joint by joint, since
a servo that clears a single move can still cook when the move repeats.

For magnetic, electric, radio and sound questions, use the field commands, which solve with GetDP. It is
fetched the first time one of them runs (an 11 MB download). Each takes a template and its values with units,
one `--set` per value. `wobbl magnetic solenoid --set turns=800 --set current=0.6A --set stroke=6mm --set need=8N`
(MCP `solve_magnetic`) gives a plunger's pull over its whole stroke with nonlinear steel, its inductance and
whether the steel saturates, and `electromagnet` and `magnet` do the same for a pot core lifting a plate and a
permanent magnet holding one. The force comes from the Maxwell stress tensor and is checked by virtual work,
so read that check before quoting it.
`wobbl electric plates --set diameter=100mm --set gap=2mm --set voltage=3kV` (MCP `solve_electric`) gives the
capacitance and the peak field against the dielectric strength, and `coax` and `sphere` cover cables and high
voltage gaps.
`wobbl board microstrip --set width=0.3mm --set height=0.17mm --set length=20mm` (MCP `solve_board`) gives a
trace's impedance from a field solve of its cross section, with its inductance and capacitance per length, and
`coupled` gives the near and far end crosstalk between two traces. To see what the board does to a converter,
pass those parasitics into the circuit run, as in
`wobbl circuit buck --set vin=12 --set vout=5 --set current=2 --set loop_inductance=6n --set switch_capacitance=300p`,
or point `board=` at the board result folder. Wobbl then simulates the switch node with and without them and
tries a snubber.
`wobbl antenna dipole --set frequency=2.4GHz --set length=58mm` (MCP `solve_antenna`) sweeps a dipole or a
monopole with a full wave solve for its resonance, impedance, S11 and gain. Adding
`--set tx_power=10dBm --set sensitivity=-90dBm --set environment=office` gives the range with its spread, and
`wobbl antenna link` does the link budget alone. `patch` sizes a patch antenna with the transmission line model
rather than a full wave solve, so say that when you quote it.
`wobbl sound room --set size=5x4x2.7` (MCP `solve_sound`) finds a room's modes and flags the ones that crowd
together, `enclosure` does the same for a box measured in millimeters, `muffler` gives an expansion chamber's
transmission loss and `panel` gives the sound pressure level of a vibrating panel treated as a baffled piston,
which is an upper estimate.

## Never skip

- **Never quote a low confidence number as the answer.** Say what failed and rerun with
  `--quality fine` or a longer `--budget`, or fix the setup.
- **Never quote an excluded peak.** Peaks at sharp inside corners, at the edge of fixed faces and at
  rigid couplings are idealization artifacts; Wobbl lists them under `singularities`. The real
  answer is the governing value. If the material is brittle and a sharp corner is flagged, recommend
  a fillet and re-run.
- **Check units.** Files are read in mm. If the bounding box looks 25.4x or 1000x off, pass `--scale`.
- **Check the supports match reality.** A fixed face is perfectly rigid. Bolted holes, walls that
  can lift off and sliding faces each have their own type; pick the one that matches the build.
- **Do not use `--quality fast` for a final answer**; it is for quick looks.
- **State the assumptions** Wobbl returns, especially guesses it made from a plain question.
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
