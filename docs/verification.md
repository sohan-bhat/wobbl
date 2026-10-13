# How Wobbl checks itself

A finite element number is only as good as the mesh, the supports and the
setup behind it. Wobbl runs these checks on every answer and turns them into a
confidence level. It never reports high confidence after a failed check.

## Strength

**Mesh convergence.** The first mesh is coarse with curved features resolved.
Wobbl solves, finds every stress peak, refines the mesh around the ones that
matter (whole fillets when they are tight, balls around other peaks) and solves
again, until the governing stress changes by less than 3% between meshes
(`--quality fine`: 2%, `fast`: 5%). At least three meshes are required for a
pass; two meshes that agree can both be too coarse. The check fails above 10%
change. If the number is still moving but even the highest peak seen on any
mesh leaves comfortable margin, the check is a warning and the report says the
verdict holds either way.

**Equilibrium.** Reaction forces and moments at the supports must cancel the
applied loads (gravity and rigid attachments included) to 0.5%.

**Hand calculation.** Wobbl cuts the part with a plane through the governing
point, square to the line toward the load. When every support is on one side
and every load on the other, the cut carries the whole load like a beam: Wobbl
computes the section's area and second moments, the axial force, bending
moment and torsion, and the nominal stress `N/A + M c / I`. The FEA peak divided
by that nominal stress is the local stress concentration. Between 0.85 and 4 is
physical (plain beam region up to fillets and holes); far outside that range is
flagged for a look at loads, units and supports.

**Singularities.** Linear elasticity predicts infinite stress at perfectly sharp
inside corners, at the edge of an infinitely rigid support and where a rigid
attachment grabs the part. Those peaks climb without limit as the mesh is
refined. Wobbl finds sharp inside corners from the CAD (exact normals, so a
small fillet is not mistaken for a corner), tracks every peak through the
refinement levels, and reports the governing stress from points at least a
fixed distance away (a fifth of the wall thickness or one element). The excluded
peaks are listed with their trend, for example `20.5 -> 27.3 -> 29.7 MPa,
singular`. In a brittle material a sharp corner lowers confidence and the fix
list asks for a fillet.

**Load introduction.** Loads that act off the part (a belt pull at a pulley,
thrust and torque on a motor) reach it either through a rigid attachment or
spread over the faces without stiffening them (a distributing coupling, like
an RBE3). The two idealizations can give very different peaks: on a cast motor
mount with a belt pull shared by four flange bolts, the spread load puts 124 MPa
on a rib tip while a rigid servo body moves the peak into the bolt holes and
leaves the rib at 36 MPa. Wobbl solves the other idealization on the final mesh,
reports both, and lowers confidence when the verdict depends on the choice.
Peaks where a load is pushed into holes are reported separately, like support
artifacts.

**Bolt bearing.** At bolted and fixed holes, where the stress is set by the
support idealization, Wobbl reports the honest number instead: the reaction
force through each hole divided by its projected area (diameter times length),
against the bearing limit.

**Units and sanity.** Part size (0.001 mm or 20 m parts usually mean a unit
mistake), peak strain, stress far above or below the material limit, and
deflection: over 6% of the part size is a warning (linear theory gets less
accurate), over 20% is a failure.

**Bounded runtime.** Every study has a time budget (`--budget`, by
default 300 s). If a refinement level runs out of time, fails to mesh or the solver
fails, Wobbl stops and returns the last finished level with its checks and
heatmaps; the convergence check says how far it got. An agent always gets an
answer back, with a confidence that matches it.

**Contact.** Wall supports are solved by an active set loop: a coarse linear
pre solve finds the contact patch, each mesh level starts from it, and the loop
stops when fewer than 2% of the contact nodes change state.

## Heat

- convergence of the hottest point (or the coolest one when the hottest is a
  held temperature)
- energy balance: heat made plus heat through held temperature faces (CalculiX
  reaction flux) equals heat shed by convection and radiation, to 1%
- hand calculation: in steady state the area averaged cooled surface
  temperature must equal `T_air + P / (h A)`; the Biot number says whether the
  part is nearly isothermal. In a transient, the lumped capacitance curve with
  time constant `rho c V / (h A)` is compared with the FEA mean temperature
- a note when radiation is off but would shed significant heat

## Circuits

- the textbook answer: the diode equation solved by hand for LEDs, `1 - e^-1`
  for RC, triangular ripple current through C and ESR for buck converters,
  `I_rms^2 R` for MOSFET conduction
- time step convergence: transients are re-run at half the step
- power balance: source power equals the sum of the dissipated powers
- for MOSFETs, the junction temperature loop with on resistance rising with
  temperature, which also detects thermal runaway

## Confidence

| Level | Means |
|---|---|
| high | every check passed or did not apply |
| medium | at least one warning: still converging, a sharp corner in a brittle part, a peak set by the support idealization well above the reported value, an unusual hand calculation ratio |
| low | at least one check failed |

The reasons are always listed next to the level, in the terminal, the JSON
and the report.
