"""Wobbl's MCP server (stdio). Start it with 'wobbl mcp'; 'wobbl setup' registers it with your agents."""
from __future__ import annotations

import os
import sys

from . import tools

INSTRUCTIONS = """Wobbl runs verified simulations of anything a person is about to build. Use one tool for all of it.
1. Call simulate(thing, question). thing is whatever describes it, a CAD file (STEP, STL or a parametric .py), a
   SPICE netlist, a mechanism spec, URDF or MJCF, a rocket or motor spec, a fluid case, a molecule, element,
   material or reaction as a name, formula, SMILES or equation, or a sentence describing it. Wobbl works out the
   domain and the physics from the question, runs several physics in one call when the question needs them,
   and checks each answer.
2. Open the 3D view for the user straight away. The result's view is an HTML page of their own scenario playing
   out with the result on it (open on macOS, xdg-open on Linux, start on Windows). Tell them what they are watching.
3. Read decisions and assumptions. Each one carries the override that replaces it. When a guess does not match
   what the user said or built, call simulate again with physics and options set from those overrides, or
   reuse explicit.options, which holds every value the run used. Fill material, loads, fixtures and options
   precisely when you know them rather than leaving Wobbl to guess.
4. Read confidence, confidence_reasons and checks before quoting any number. Never quote a low confidence result
   as final, and never quote peaks listed as excluded artifacts.
5. physics() lists every domain, its physics and their options. plan_only=True shows the route and the guesses
   without running, and estimate=True predicts time, memory and disk first for anything big.
After a run, the general tools work on any result folder. inspect_geometry describes a CAD part with face ids
for loads and fixtures, verify and render reread a result, suggest_fix gives fixes with numbers and applies
strength fixes, quantify_uncertainty, validate_against_tests, assess_credibility and plan_tests build
credibility evidence (Wobbl never certifies a model), estimate_cost and the job tools run big problems in the
background or on a remote machine, and list_materials gives the material data. The older solve_* tools remain
as deprecated aliases of simulate."""


def _protect_stdout():
    """Keep the JSON-RPC stream clean: native libraries that print go to stderr instead."""
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = os.fdopen(saved, "w", buffering=1, encoding="utf-8")
    except OSError:
        pass


# the tools agents should reach for; every other tool is a deprecated alias of simulate and says so
GENERAL = {"simulate", "physics", "inspect_geometry", "verify", "render", "suggest_fix", "quantify_uncertainty",
           "validate_against_tests", "assess_credibility", "plan_tests", "estimate_cost", "start_job", "job_status",
           "job_logs", "cancel_job", "resume_job", "list_materials", "solve_submodel"}


def _deprecate(srv) -> None:
    """Mark each scenario tool registered after this point as a deprecated alias, so the tool list leads agents
    to simulate while older calls keep working."""
    import inspect
    plain = srv.tool

    def tool(*args, **kw):
        def deco(fn):
            if fn.__name__ not in GENERAL and not kw.get("description"):
                doc = inspect.cleandoc(fn.__doc__ or "")
                kw["description"] = ("Deprecated alias of simulate, which reads the same study from a thing and a "
                                     "question. " + doc)
            return plain(*args, **kw)(fn)
        return deco
    srv.tool = tool


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
    srv = Server("wobbl", instructions=INSTRUCTIONS)

    @srv.tool()
    def simulate(thing: str = "", question: str = "", physics: list[str] | None = None, domain: str = "",
                 kind: str = "", material: str = "", loads: list[str] | None = None,
                 fixtures: list[str] | None = None, options: dict | None = None, plan_only: bool = False,
                 quick: bool = False, estimate: bool = False, background: bool = False, remote: str = "") -> dict:
        """Simulate anything from what describes it and the question in plain words. thing is a CAD file, a
        netlist, a spec, a URDF or MJCF model, a fluid case folder, a finished result folder, a name, formula,
        SMILES or equation, or a sentence. Wobbl picks the domain and the physics (several when the question needs
        them, as in "will it survive a drop and stay under 60 C"), runs each with its checks and returns view, the
        3D page of the scenario to open for the user at once, then verdict, plan, decisions and assumptions each
        with its override, explicit (every value used, ready to pass back), steps with each physics' result, and
        result_dir. Set physics (e.g. ["stress", "heat"]), domain and kind to override the reading. material,
        loads (e.g. "200 N down @ top") and fixtures (e.g. "bolt holes d=4.5") are for parts, and options holds
        any other value a physics reads, as {"power": "12 W", "rpm": "6000"}. A CLI override '--name value' is
        options[name]. plan_only shows the route and guesses without running, estimate predicts cost, quick
        skips images and fixes, and background or remote run it as a job."""
        from .simulate import simulate_tool
        return simulate_tool(thing, question, physics, domain, kind, material, loads, fixtures, options, plan_only,
                             quick, estimate, background, remote)

    @srv.tool()
    def physics(domain: str = "") -> dict:
        """Every domain simulate reads (part, circuit, mechanism, flight, fluid, field, chemistry and any added
        since), the kinds of thing each takes, and each physics with its aliases and options."""
        from .simulate import physics_tool
        return physics_tool(domain)

    _deprecate(srv)

    @srv.tool()
    def inspect_geometry(path: str, scale: float = 1.0) -> dict:
        """Describe a STEP/STL part: bounding box (mm), holes, fillets, sharp inside corners, wall thickness,
        the largest faces with ids, and example face selectors to use in loads and fixtures."""
        return tools.inspect_geometry(path, scale)

    @srv.tool()
    def mesh(path: str, size: float | None = None) -> dict:
        """Mesh a part with second order tetrahedra and report nodes, elements and element size. Studies mesh
        automatically; use this to check that a file meshes at all."""
        return tools.mesh_part(path, size)

    @srv.tool()
    def new_study(path: str, material: str = "", question: str = "") -> dict:
        """Start a strength study on a part. Then call add_fixture, apply_load and solve_structural."""
        return tools.new_study(path, material, question)

    @srv.tool()
    def set_material(study_id: str, material: str) -> dict:
        """Set the material: pla, petg, abs, asa, nylon, pa12, pc, 6061-t6, 7075-t6, mild-steel, ss304,
        ti-6al-4v, wood, ... or custom:E=3000,nu=0.35,yield=40 (MPa)."""
        return tools.set_material(study_id, material)

    @srv.tool()
    def apply_load(study_id: str, load: str) -> dict:
        """Add a load, e.g. '200 N down @ top', '50 lbf -z @ x=max', '0.2 MPa @ top', '3 g down',
        '300 N -x bearing @ hole d=6.6' (a bolt or pin pressing on one side of the hole, where the same load
        without bearing spreads evenly over the hole faces), '2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid',
        '0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25'."""
        return tools.apply_load(study_id, load)

    @srv.tool()
    def add_fixture(study_id: str, fixture: str) -> dict:
        """Add a support: 'holes d=4.5' (fixed), 'bolt holes d=6.6' (hole and washer ring), 'x=min',
        'wall x=min' (compression only), 'roller y=0', 'elastic 50 N/mm3 @ z=min' (a bed of springs)."""
        return tools.add_fixture(study_id, fixture)

    @srv.tool()
    def solve_structural(study_id: str = "", path: str = "", material: str = "", loads: list[str] | None = None,
                         fixtures: list[str] | None = None, target_sf: float | None = None,
                         quality: str = "normal", budget: float = 300.0, question: str = "",
                         print_direction: str | None = None, shell: bool = False, layup: str = "",
                         ply: str = "t300-5208") -> dict:
        """Run the verified static strength study (mesh convergence, equilibrium, beam theory check,
        singularity detection). Returns verdict, governing stress and location, safety factor, deflection,
        confidence with reasons, checks, assumptions, fixes, image paths and the HTML report path.
        Simple shapes finish in under a minute, and real parts take one to seven minutes on a 2018 laptop.
        shell=True solves a thin walled part (enclosure, cover, tube) as shells, checked against a coarse solid.
        layup (e.g. "[0/±45/90]s", plies from the inner face out) makes the walls a composite of ply material
        ply (t300-5208, im7-8552, as4-3501-6, e-glass-epoxy) and answers with the load factor at first ply
        failure by Tsai-Wu and Hashin, the governing ply and layup changes that reach the target."""
        return tools.solve_structural(study_id, path, material, loads, fixtures, target_sf, quality, budget,
                                      question, print_direction, shell, layup, ply)

    @srv.tool()
    def verify(result_dir: str) -> dict:
        """Verification checks, confidence and excluded artifact peaks of a finished study."""
        return tools.verify(result_dir)

    @srv.tool()
    def render(result_dir: str) -> dict:
        """Image paths (stress heatmap with the failure point marked, zoom, safety factor, deflection,
        convergence) and the HTML report of a finished study."""
        return tools.render(result_dir)

    @srv.tool()
    def suggest_fix(result_dir: str, apply: bool = False, which: list[str] | None = None) -> dict:
        """Concrete fixes with numbers for a finished study. For a strength result they cover fillet radius, wall
        thickness, gussets and material, and with apply=True Wobbl builds the automatic ones, solves each again
        and returns before and after. Pass which to pick kinds from fillet, thicken, gusset and material.
        Heat, circuit and motion results return their fixes with a note to call solve_thermal, solve_circuit or
        solve_motion again with the new value."""
        return tools.suggest_fix(result_dir, apply, which)

    @srv.tool()
    def solve_thermal(path: str, material: str, power: str = "", heat_faces: str = "", film: str = "",
                      ambient: str = "25 C", fixed_temp: str = "", duration: str = "", limit: str = "",
                      radiation: float = 0.0) -> dict:
        """Steady or transient heat conduction with convection: e.g. path=battery.step, material=li-ion-cell,
        power='12 W' (spread through the part, or over heat_faces), film='5 W/m2K @ all', ambient='25 C',
        duration='30 min' for transient, limit='60 C'. Returns max temperature, where, checks and images."""
        from ..thermal.tools import solve_thermal_tool
        return solve_thermal_tool(path, material, power, heat_faces, film, ambient, fixed_temp, duration, limit,
                                  radiation)

    @srv.tool()
    def solve_circuit(template: str = "", netlist: str = "", params: dict | None = None) -> dict:
        """Circuit checks with ngspice. template='led' (supply, resistor, led color or vf, current rating),
        'buck' (vin, vout, current, frequency, inductor, capacitor), 'mosfet' (current, rds_on, theta_ja,
        ambient, switching), 'rc' (r, c, v); or netlist=SPICE text with params.vectors to read."""
        from ..circuits.tools import solve_circuit_tool
        return solve_circuit_tool(template, netlist, params or {})

    @srv.tool()
    def solve_mechanism(template: str = "", spec: str = "", params: dict | None = None, question: str = "") -> dict:
        """Mechanisms run in MuJoCo and checked against closed forms, half the time step, an energy account and
        contact depth. template='arm' (servo or shoulder_servo and so on, payload, links, link_mass, pick, place,
        lift, move_time, mount=free with base_mass and base_size) says whether hobby servos lift a load through a
        pick and place. 'fourbar' (ground, crank, coupler, rocker in mm, motor or stall_torque and no_load_speed, or
        speed, load on the rocker, sweep) gives swing, dead points, transmission angle and motor torque. 'gears'
        (teeth like '12:36,10:40', efficiency, motor, load, speed) gives output speed and torque. 'impact' (mass,
        speed or drop, target_mass, restitution, radius, material, limit) gives peak contact force. 'tip' (base,
        base_mass, mass, height, payload, reach, force, accel) says whether it tips, and 'pendulum' swings one. Or
        pass spec, a TOML spec path or its text, or an MJCF or URDF path, or just a question. Bare numbers are mm,
        g, degrees, N m, rpm and s.
        Returns verdict, headline, per joint torque and speed against each motor, events, confidence with checks,
        fixes that were solved again, images and the trajectory CSV. Runs take a few seconds to half a minute."""
        from ..mechanism.tools import solve_mechanism_tool
        return solve_mechanism_tool(template, spec, params or {}, question)

    @srv.tool()
    def solve_motion(spec: str = "", name: str = "", mass: str = "", density: str = "", diameter: str = "",
                     cd: float | None = None, height: str = "", speed: str = "", angle: str = "", wind: str = "",
                     question: str = "") -> dict:
        """Drops, throws and rocket flights through gravity that falls off with altitude, the 1976 standard
        atmosphere, drag, wind, thrust and parachutes, integrated with scipy and checked against closed forms, an
        energy balance and a delta v budget. A round body takes diameter='9 cm' with mass='1 kg' or
        density='2700 kg/m3', cd (default 0.47), height='300 m', speed='30 m/s', angle='35 deg' and wind. A rocket
        takes spec, a TOML file path or the TOML text itself, with [rocket], [nose], [fins], [motor], [launch],
        [drogue], [main] and [targets] tables as in examples/rocket-launch/rocket.toml, and wind overrides the
        spec's. Returns verdict, headline, events, apogee, peak Mach, stability margin, rail exit and landing
        speed, drift, confidence with checks, fixes and the trajectory CSV. Runs take a second or two."""
        from ..motion.tools import solve_motion_tool
        return solve_motion_tool(spec, name, mass, density, diameter, cd, height, speed, angle, wind, question)

    @srv.tool()
    def solve_motor(spec: str = "", propellant: str = "", grains: list[str] | None = None, throat: str = "",
                    exit: str = "", casing: str = "", efficiency: float | None = None, question: str = "") -> dict:
        """Solid rocket motor internal ballistics from a design: chamber pressure, mass flow and thrust over the
        burn, total impulse and motor class, checked by time step convergence, mass balance, the closed form
        pressure and a Cantera equilibrium cross check of the propellant. Judges the casing (hoop stress at peak
        pressure against yield and burst), Kn, erosive burning (port to throat ratio, core mass flux) and nozzle
        separation, and simulates each fix. Give spec, a TOML path or text as in examples/knsb-motor/motor.toml,
        or propellant (knsb, kndx, knsu, ocean-water, cherry-limeade), grains like ['bates d=48 core=18 L=80 x4'],
        throat='13 mm', exit='33 mm' and casing='6061-t6 od=54 wall=1.6 L=400 sf=2' (bare numbers are mm).
        Returns the thrust curve CSV and a RASP .eng that solve_motion reads. Runs take a few seconds."""
        from ..chem.tools import solve_motor_tool
        return solve_motor_tool(spec, propellant, grains, throat, exit, casing, efficiency, question)

    @srv.tool()
    def solve_flame(fuel: str, oxidizer: str = "air", phi: float = 1.0, pressure: str = "1 atm",
                    temperature: str = "25 C", limit: str = "", question: str = "") -> dict:
        """Adiabatic flame temperature and equilibrium products (CO, NO) of methane, propane, butane, hydrogen,
        ethane, ethylene, acetylene, methanol, carbon monoxide or natural gas in air or oxygen, on NASA's
        thermodynamic data with Cantera. Checked against GRI-Mech 3.0's data, the no dissociation bound and the
        flammability limits. limit='1800 C' gets the leaner mixture that keeps the flame under it."""
        from ..chem.tools import solve_flame_tool
        return solve_flame_tool(fuel, oxidizer, phi, pressure, temperature, limit, question)

    @srv.tool()
    def solve_propellant(propellant: str, chamber_pressure: str = "6.895 MPa", area_ratio: float | None = None,
                         exit_pressure: str = "", ambient: str = "1 atm", question: str = "") -> dict:
        """Rocket propellant performance, CEA style: chamber temperature, c* and Isp at ambient and in vacuum for
        shifting and frozen expansion. propellant is knsb, kndx, knsu or a composition by mass of kno3, sorbitol,
        dextrose, sucrose, ap, al, htpb, lox, lh2, lch4 or rp1, e.g. 'ap:70,al:16,htpb:14' or 'lox:6,lh2:1'.
        Give area_ratio, or exit_pressure (default the ambient). Matches NASA CEA to about 0.1%."""
        from ..chem.tools import solve_propellant_tool
        return solve_propellant_tool(propellant, chamber_pressure, area_ratio, exit_pressure, ambient, question)

    @srv.tool()
    def solve_runaway(cell: str = "18650", ambient: str = "25 C", film: str = "", emissivity: float = 0.8,
                      from_result: str = "", question: str = "") -> dict:
        """Lithium ion thermal runaway onset: the surrounding temperature at which a charged cell's own
        decomposition heat outruns its cooling, by Semenov's criterion and a transient with the reactants used up
        (Hatchard's LiCoO2 kinetics). cell is 18650, 21700, 26650, 14500 or '18x65' mm, film is the cooling as
        '25 W/m2K' or words like 'sealed box' or 'a small fan', and from_result takes the surroundings from the
        hottest point of a solve_thermal result. Returns the onset, the margin and fixes. Runs in seconds."""
        from ..chem.tools import solve_runaway_tool
        return solve_runaway_tool(cell, ambient, film, emissivity, from_result, question)

    @srv.tool()
    def solve_flow(kind: str, params: dict | None = None) -> dict:
        """Air and water flow. kind='pipe' takes length, flow='30 L/min', diameter='25 mm' (or width and height
        for a duct), fluid (air, water, seawater), material or roughness, fittings='3 elbow90, gate-valve', rise
        and max_dp='50 kPa'. kind='drag' takes shape (sphere, cylinder, disk, square-plate, cube), size, speed,
        length for a cylinder's span, fluid and max_drag. kind='prop' takes prop (a UIUC key like apcsf_10x4.7, or
        '10x4.7' in inches), and rpm, thrust='4 N' or mass='1.2 kg' with rotors=4 to hover, speed, altitude and
        max_rpm. kind='hull' takes length, beam, draft, displacement ('16 t' or '15.6 m3'), speed='8 kn' and
        power='60 kW'. Values take units. Returns the verdict, the key numbers, confidence with checks against
        measurements or hand calculations, fixes and image paths. Each runs in seconds."""
        from ..flow.tools import solve_flow_tool
        return solve_flow_tool(kind, params or {})

    @srv.tool()
    def quantify_uncertainty(result_dir: str, vary: list[str], samples: int = 200000,
                             solves: int | None = None) -> dict:
        """Carry input scatter through a finished study. vary lists inputs with their scatter, e.g. ['load 10%',
        'yield 8%', 'E uniform 70000..75000', 'thickness tol 0.2', 'film uniform 3..8']. Returns the probability
        of failure with its 95% interval, the spread of the answer and Sobol indices ranking the inputs. Load,
        stiffness and strength scatter on a strength study scale the solution exactly and take seconds; geometry,
        heat and circuit inputs need new solves and a surrogate whose error is reported."""
        return tools.quantify_uncertainty(result_dir, vary, samples, solves)

    @srv.tool()
    def validate_against_tests(result_dir: str, tests: str, vary: list[str] | None = None,
                               load_uncertainty: float = 0.01) -> dict:
        """Compare a finished study with physical test readings. tests is a CSV with columns name, quantity
        (strain, displacement, temperature or a circuit value), value, unit, uncertainty, x, y, z, direction, load
        and time. Returns the ASME V&V 20 comparison error and validation uncertainty of each reading, the area
        metric, and a Bayesian calibration of the uncertain inputs (from quantify_uncertainty or vary, e.g.
        ['E 5%']). Readings marked synthetic in the file are reported as synthetic."""
        return tools.validate_against_tests(result_dir, tests, vary, load_uncertainty)

    @srv.tool()
    def assess_credibility(result_dir: str, influence: str = "medium", consequence: str = "medium",
                           question: str = "") -> dict:
        """Map a finished study's evidence to an ASME V&V 40 style credibility assessment. influence (low, medium,
        high) is how much the decision leans on the model, consequence how bad a wrong decision would be. Returns
        each factor's level against a proposed goal with its evidence, the test plan and credibility.html for a
        reviewer. Run quantify_uncertainty and validate_against_tests first for the strongest evidence. Wobbl
        supplies evidence and does not certify anything."""
        return tools.assess_credibility(result_dir, influence, consequence, question)

    @srv.tool()
    def plan_tests(result_dir: str) -> dict:
        """The smallest set of physical tests that would most reduce the risk of a decision: input measurements by
        Sobol index, a validation test with gauge positions, load steps and predicted readings, and a gauge where
        idealizations disagree. Also says which usual tests the simulation evidence can or cannot stand in for."""
        return tools.plan_tests(result_dir)

    @srv.tool()
    def solve_buckling(path: str, material: str, loads: list[str], fixtures: list[str],
                       target_sf: float | None = None, shape: str = "auto", quality: str = "normal") -> dict:
        """Linear buckling with CalculiX: the factor on the loads at which the part buckles, its mode shapes, and a
        design factor after an imperfection knockdown for the shape (column curve, NASA SP-8007 cylinders, plates,
        domes) and yielding. Loads and fixtures are written as for solve_structural, e.g. loads=['2 kN -z @ z=max'],
        fixtures=['z=min']. Checks Euler for columns, force balance and mesh convergence."""
        from ..life.tools import solve_buckling_tool
        return solve_buckling_tool(path, material, loads, fixtures, target_sf, shape, quality)

    @srv.tool()
    def solve_vibration(path: str, material: str, fixtures: list[str], masses: list[str] | None = None,
                        rpm: list[str] | None = None, blades: int = 2, frequencies: list[str] | None = None,
                        sine: str = "", psd: str = "", axis: str = "z", duration: str = "",
                        damping: float | None = None, modes: int = 15, target_sf: float | None = None,
                        quality: str = "normal", loads: list[str] | None = None) -> dict:
        """Natural frequencies and mode shapes of any part with CalculiX, judged against what shakes it. masses are
        parts it carries, e.g. ['60 g at 150,0,25 @ top and within 15 of 150,0,15']. rpm like ['4000-9000 rpm']
        with blades checks rotor and blade pass resonance, and frequencies adds tones to avoid. sine='2 g z 10-500 Hz'
        gives the peak stress of a base sweep, and with loads like ['5 N z @ top'] a band alone, sine='10-500 Hz',
        sweeps those loads as harmonic forces (both checked against CalculiX steady state dynamics). psd='gevs-qual'
        or breakpoints gives the 3 sigma stress of random input along axis, with a fatigue estimate over duration.
        Returns frequencies, effective mass, the resonance table, stresses, checks and fixes."""
        from ..life.tools import solve_vibration_tool
        return solve_vibration_tool(path, material, fixtures, masses, rpm, blades, frequencies, sine, psd, axis,
                                    duration, damping, modes, target_sf, quality, loads)

    @srv.tool()
    def fatigue_life(result_dir: str = "", material: str = "", amplitude: str = "", mean: str = "",
                     kt: float | None = None, radius: str = "", ratio: float = 0.0, cycles: str = "1e7",
                     blocks: list[str] | None = None, repeats: float = 1.0, duration: str = "",
                     surface: str = "auto", reliability: float = 0.9, mean_stress: str = "goodman",
                     target_sf: float | None = None, history: str = "", path: str = "",
                     fixtures: list[str] | None = None, loads: list[str] | None = None,
                     quality: str = "normal") -> dict:
        """Stress life fatigue: cycles to crack initiation and the safety factor on stress at the required cycles.
        Works from a strength result_dir, from any part path with its fixtures and loads (solved first), from a
        vibration result over duration, or from material with amplitude, mean and kt given directly. The load
        cycles at load ratio ratio (0 on and off, -1 reversed), through blocks like ['100% x 2e4', '60% x 1e6'], or
        through a measured history (a file or numbers, multiples of the studied load) counted by rainflow, repeated
        repeats times. Uses sourced S-N data, Marin factors, Peterson notch sensitivity, Goodman (or gerber,
        soderberg) and Miner's rule."""
        from ..life.tools import fatigue_life_tool
        return fatigue_life_tool(result_dir, material, amplitude, mean, kt, radius, ratio, cycles, blocks, repeats,
                                 duration, surface, reliability, mean_stress, target_sf, history, path, fixtures,
                                 loads, quality)

    @srv.tool()
    def solve_plastic(path: str, material: str, loads: list[str], fixtures: list[str], target: float | None = None,
                      quality: str = "normal", curve: str = "auto") -> dict:
        """Elastic plastic study of a metal part past first yield, with the same loads and fixtures as
        solve_structural. Returns the load multiple at first yield, the collapse (limit) load multiple, the
        permanent set and peak plastic strain after the load is applied and removed, and a verdict that separates
        holds (stays elastic), yields (yields locally but carries the load), marginal and collapses, with checks
        (collapse load over two meshes, a fully plastic section from beam theory, balance at collapse, ductility)
        and fixes aimed at the collapse mechanism. target is the margin wanted on collapse (default 1.5 for most
        metals). Takes one to five minutes."""
        from ..plastic.tools import solve_plastic_tool
        return solve_plastic_tool(path, material, loads, fixtures, target, quality, curve)

    @srv.tool()
    def solve_magnetic(template: str, params: dict | None = None) -> dict:
        """Magnetostatics with nonlinear steel (GetDP). template='solenoid' (turns, current, coil_id, coil_od, length,
        plunger_d, frame, clearance, stroke, material, need in N) gives the pull force over the stroke;
        'electromagnet' (diameter, height, pole_d, wall, base, turns, current, plate, stroke, need) the holding
        force on a plate; 'magnet' (diameter, height, grade n35/n42/n52/ferrite, cup, plate, stroke, need) a
        permanent magnet's pull. Returns force, inductance, saturation and demagnetization, checked by virtual
        work against the stress tensor, energy balance and three meshes. Values take units, e.g. '6 mm'."""
        from ..fields.tools import solve_field_tool
        return solve_field_tool("magnetic", template, params)

    @srv.tool()
    def solve_electric(template: str, params: dict | None = None) -> dict:
        """Electrostatics (GetDP): capacitance, peak field and breakdown margin. template='plates' (diameter, gap,
        thickness, voltage, dielectric, edge round or square), 'coax' (inner_d, outer_d, offset, voltage,
        dielectric) or 'sphere' (diameter, gap to a grounded plane, voltage). margin sets the wanted factor
        below breakdown (default 2). Checked by charge balance, charge against energy and a closed form."""
        from ..fields.tools import solve_field_tool
        return solve_field_tool("electric", template, params)

    @srv.tool()
    def solve_board(template: str, params: dict | None = None) -> dict:
        """PCB trace cross sections (GetDP): impedance, effective permittivity, inductance and capacitance per
        length, and with length= the totals to feed solve_circuit (buck params loop_inductance and
        switch_capacitance). template='microstrip', 'stripline', 'coplanar' (gap, ground_width), 'coupled' (spacing,
        length, rise, swing, max_crosstalk for near and far end crosstalk) or 'coax' (inner_d, outer_d). Common:
        width, height (dielectric thickness), thickness (copper), dielectric (fr4, ro4350b, ...) or er, target
        impedance and tolerance. Checked against Hammerstad and Jensen, Cohn and conformal mapping formulas."""
        from ..fields.tools import solve_field_tool
        return solve_field_tool("board", template, params)

    @srv.tool()
    def solve_antenna(template: str, params: dict | None = None) -> dict:
        """Antennas and range. template='dipole' or 'monopole' (frequency, length, diameter, gap, impedance) runs an
        axisymmetric full wave sweep (GetDP) for resonance, impedance, S11, pattern and gain; 'patch' (frequency,
        height, dielectric or er, width, length) sizes a patch with the transmission line model; 'link'
        (frequency, tx_power, sensitivity, tx_gain, rx_gain, losses, environment office/home/factory/free/
        outdoor/urban/suburban, distance, reliability) is a link budget with Friis and ITU-R P.1238 or log
        distance path loss and shadowing. Add tx_power and sensitivity to any antenna for its range."""
        from ..fields.tools import solve_field_tool
        return solve_field_tool("antenna", template, params)

    @srv.tool()
    def solve_sound(template: str, params: dict | None = None) -> dict:
        """Acoustics (GetDP). template='room' (size '5x4x2.7' in m, fmax, t60) or 'enclosure' (size in mm) finds
        the standing wave modes and judges their spread; 'muffler' (pipe_d, chamber_d, chamber_length, fmin, fmax,
        target in dB over band '100-400') gives the transmission loss of an expansion chamber; 'panel' (diameter,
        velocity in m/s or acceleration or displacement, distance, limit in dB) the sound pressure level of a
        vibrating panel as a baffled piston. Checked against exact box modes, plane wave theory, an energy
        balance and the exact piston."""
        from ..fields.tools import solve_field_tool
        return solve_field_tool("sound", template, params)

    @srv.tool()
    def solve_submodel(result_dir: str, at: list[float] | None = None, radius: float | None = None,
                       size: float | None = None) -> dict:
        """Refine only around the peak of a finished result, with the refinement its domain registered. For a
        solve_structural result the part is cut down to a ball around the peak (or around at=[x, y, z] in mm),
        meshed finely and held on its cut where the global solution moved (CalculiX *SUBMODEL). Returns the local
        peak against the global one, and two checks on the cut, the peak's distance from it in feature sizes and
        the stress agreement across it. Quote the local peak only when trusted is true. Use it when the global
        mesh is too coarse at the peak and a finer whole model would be too big."""
        from ..jobs.workload import refine
        try:
            return refine(result_dir, at, radius, size)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), **({"hint": e.hint} if getattr(e, "hint", "") else {})}

    @srv.tool()
    def estimate_cost(command: list[str] | None = None, path: str = "", solver: str = "", nodes: float = 0,
                      elements: float = 0, end_time: float = 0.0, time_step: float = 0.0, iterations: int = 0,
                      procedure: str = "static", remote: str = "", threads: int = 0,
                      max_hours: float | None = None) -> dict:
        """Predict run time, peak memory and disk before anything runs, for any Wobbl command of any domain
        (command=['run', 'part.step', '-m', 'pla', '--fix', 'x=min', ...], or thermal, flow, mechanism and the
        rest), a CalculiX .inp, LS-DYNA .k or OpenRadioss .rad deck or an OpenFOAM case (path), or a run given by
        size (solver='openradioss', elements=1.5e6, end_time=0.2). Returns the verdict (fits, long, too long, does
        not fit, or unknown before a command's first run here), the problems and options such as a background job,
        a coarser first pass, a submodel, a remote machine or a priced cloud machine. Call it before any solve that
        might be big."""
        from ..jobs.tools import estimate_cost as run
        return run(command, path, solver, nodes, elements, end_time, time_step, iterations, procedure, remote,
                   threads, max_hours)

    @srv.tool()
    def start_job(path: str = "", command: list[str] | None = None, remote: str = "", threads: int = 0,
                  checkpoint: bool = True, cloud: str = "", instance: str = "", region: str = "us-east-1",
                  key_name: str = "", security_group: str = "", confirm: str = "",
                  max_hours: float | None = None) -> dict:
        """Run a CalculiX deck (path) or a Wobbl command (command=['run', 'part.step', '-m', 'pla', ...]) in the
        background, here or on a remote added with wobbl remote add. Checkpoints after every step, refuses a run
        that will not fit and returns the job id. With cloud='aws' it returns a priced plan and confirm_token and
        starts nothing; only after the user agrees, call again with confirm=<token>."""
        from ..jobs.tools import start_job as run
        return run(path, command, remote, threads, checkpoint, cloud, instance, region, key_name, security_group,
                   confirm, max_hours)

    @srv.tool()
    def job_status(job_id: str = "") -> dict:
        """State, progress, ETA, predicted time and the peaks found so far for a job (the newest by default)."""
        from ..jobs.tools import job_status as run
        return run(job_id)

    @srv.tool()
    def job_logs(job_id: str = "", lines: int = 40) -> dict:
        """The newest lines a job's solver printed."""
        from ..jobs.tools import job_logs as run
        return run(job_id, lines)

    @srv.tool()
    def cancel_job(job_id: str = "") -> dict:
        """Stop a job. Its checkpoint stays, so resume_job carries on from the last finished step."""
        from ..jobs.tools import cancel_job as run
        return run(job_id)

    @srv.tool()
    def resume_job(job_id: str = "") -> dict:
        """Carry on a cancelled or interrupted job from its last checkpoint, or from the start when it has none."""
        from ..jobs.tools import resume_job as run
        return run(job_id)

    @srv.tool()
    def solve_casting(path: str, alloy: str = "a356", risers: list[str] | None = None,
                      chills: list[str] | None = None, mold: str = "sand", pour: str = "", mold_temp: str = "",
                      quality: str = "normal") -> dict:
        """Sand casting: where the metal freezes last, whether the risers can feed it as it shrinks, and the
        shrinkage porosity left behind (Niyama criterion). risers like ['d=40 h=60 at 60,30,20'] (mm, the center
        of the riser's foot on the casting), chills like ['steel 30x30x10 at 10,30,-5']. alloy a356 or pure-al,
        pour '720 C'. Checks Chvorinov's rule, meshes, time step and energy balance; fixes give a riser size and
        place to pass back. Takes two to eight minutes."""
        from ..process.tools import solve_casting_tool
        return solve_casting_tool(path, alloy, risers, chills, mold, pour, mold_temp, quality)

    @srv.tool()
    def grow_crack(path: str, material: str = "", loads: list[str] | None = None,
                   fixtures: list[str] | None = None, crack: str = "auto", r_ratio: float | None = None,
                   spectrum: str = "", law: str = "nasgro", growth: str = "", kc: float | None = None,
                   wheeler: float = 0.0, target_life: float | None = None, max_pof: float = 0.01,
                   detectable: str = "1.27 mm", samples: int = 20000, steps: int = 12, budget: float = 1200.0,
                   quality: str = "normal", question: str = "") -> dict:
        """Fatigue crack growth with CalculiX. path is a CAD file, or a strength or fatigue result_dir, whose part,
        material, loads, fixtures, governing point and duty cycle it reuses. crack='auto' starts the damage
        tolerance flaw at the most stressed point; or 'auto at 30,0,5', 'through 1.27 mm at hole d=8', 'corner 1 mm
        at hole d=8', 'surface a=1 c=2 at hottest'. Reads K_I, K_II and K_III along the front, turns and advances it
        step by step with remeshing, so the path comes from the geometry, and integrates the NASGRO or Paris law
        (metals without data get constants estimated from their stiffness; growth='C=..,n=..,dkth=..,kic=..'
        overrides). Loads are the peak of each cycle, with r_ratio, a spectrum file of 'max,min' factors or
        blocks like '100% x 3650 R=0; 70% x 1e8 R=0.7'.
        Returns the nominal life, B50, B10 and B1 from a Monte Carlo over the starting flaw, growth constants, load
        and toughness, the probability of failing before target_life, inspection times, where the crack starts,
        checks (SIF mesh convergence, handbook K, force balance, integration step, sample count), fixes and the
        path image. One part's failure cycle cannot be predicted, only the spread. Takes five to twenty minutes."""
        from ..fracture.tools import grow_crack_tool
        return grow_crack_tool(path, material, loads, fixtures, crack, r_ratio, spectrum, law, growth, kc, wheeler,
                               target_life, max_pof, detectable, samples, steps, budget, quality, question)

    @srv.tool()
    def list_materials() -> dict:
        """Built in materials with stiffness, strength limit, density, conductivity and temperature limit."""
        return tools.list_materials()

    return srv


def main():
    _protect_stdout()
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    build_server().run()
