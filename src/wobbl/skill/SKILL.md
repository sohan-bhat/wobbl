---
name: wobbl
description: Run verified engineering simulations on CAD parts and circuits with Wobbl. Use when someone asks whether a physical part will hold, crack, bend, yield, overheat or fail (brackets, mounts, enclosures, drone arms, heat sinks, battery packs), wants stress, deflection, safety factor or temperature numbers, or asks about LED current, regulator ripple or MOSFET heating. Wobbl meshes the CAD, runs CalculiX or ngspice, checks its own answer and returns a heatmap, a confidence level and fixes.
---

# Wobbl: simulate before you build

Wobbl wraps real solvers (gmsh, CalculiX, ngspice) behind one CLI and an MCP
server. It refines the mesh until the answer stops moving, checks equilibrium,
compares against beam theory, flags stress singularities and reports a
confidence level. Use it instead of hand calculations for real geometry, and
never write your own finite element code.

## Workflow

1. **Look at the part first.** `wobbl inspect part.step --json` (MCP: `inspect_geometry`).
   Note the bounding box (units are mm), holes with diameters, fillets, sharp inside corners
   and wall thickness. Use these to write selectors.
2. **Decide loads and supports explicitly.** Ask the user if they are unclear. Write them as:
   - loads: `"200 N down @ top"`, `"50 lbf -z @ x=max"`, `"0.2 MPa @ top"`, `"3 g down"`,
     `"300 N -x @ hole d=6.6"` (bolt bearing), `"2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6 rigid"`
     (force at a point off the part, carried by the faces), `"0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25"`.
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
