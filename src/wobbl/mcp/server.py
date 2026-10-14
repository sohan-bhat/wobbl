"""Wobbl's MCP server (stdio). Start it with 'wobbl mcp'; 'wobbl setup' registers it with your agents."""
from __future__ import annotations

import os
import sys

from . import tools

INSTRUCTIONS = """Wobbl runs verified engineering simulations. Typical flow for a strength question:
1. inspect_geometry(path) to see faces, holes, fillets and thickness (mm).
2. solve_structural(path, material, loads, fixtures, target_sf) with loads like "200 N down @ top" and
   fixtures like "bolt holes d=6.6"; it refines the mesh until converged and checks itself.
3. Read confidence, confidence_reasons and checks before quoting any number. Never quote a low
   confidence result as final, and never quote peaks listed as excluded artifacts.
4. Open images.stress to see where the peak is. If the verdict is fails or marginal, call
   suggest_fix(result_dir, apply=true) for before and after numbers.
Heat questions: solve_thermal. Circuits: solve_circuit. Falling, thrown and rocket flights: solve_motion."""


def _protect_stdout():
    """Keep the JSON-RPC stream clean: native libraries that print go to stderr instead."""
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = os.fdopen(saved, "w", buffering=1, encoding="utf-8")
    except OSError:
        pass


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
    srv = Server("wobbl", instructions=INSTRUCTIONS)

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
        'wall x=min' (compression only), 'roller y=0'."""
        return tools.add_fixture(study_id, fixture)

    @srv.tool()
    def solve_structural(study_id: str = "", path: str = "", material: str = "", loads: list[str] | None = None,
                         fixtures: list[str] | None = None, target_sf: float | None = None,
                         quality: str = "normal", budget: float = 300.0, question: str = "",
                         print_direction: str | None = None) -> dict:
        """Run the verified static strength study (mesh convergence, equilibrium, beam theory check,
        singularity detection). Returns verdict, governing stress and location, safety factor, deflection,
        confidence with reasons, checks, assumptions, fixes, image paths and the HTML report path.
        Simple shapes finish in under a minute, and real parts take one to seven minutes on a 2018 laptop."""
        return tools.solve_structural(study_id, path, material, loads, fixtures, target_sf, quality, budget,
                                      question, print_direction)

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
    def list_materials() -> dict:
        """Built in materials with stiffness, strength limit, density, conductivity and temperature limit."""
        return tools.list_materials()

    return srv


def main():
    _protect_stdout()
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    build_server().run()
