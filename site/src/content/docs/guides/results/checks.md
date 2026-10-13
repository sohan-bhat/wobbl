---
title: "Checks"
description: "Every check Wobbl runs on strength, heat and circuit answers, what it compares and the thresholds that set its status."
lead: "A finite element number is only as good as the mesh, the supports and the setup behind it, so Wobbl tests all three on every answer."
order: 2
---

| Check | What it asks | From a real run |
|---|---|---|
| Mesh convergence | does the stress stop moving as the mesh refines | `43 -> 50.9 -> 53 -> 52.3 MPa over 4 meshes` |
| Force balance | do the reactions cancel the loads | `reactions balance the loads to 0.00014%` |
| Hand calculation | is the peak a physical multiple of beam theory | `15.2 MPa nominal, a stress concentration of about 3.4` |
| Singularities | is the peak at a sharp corner or an idealized support | `15.2 -> 20.8 -> 20.6 MPa, excluded` |
| Bolt bearing | what does each bolted hole really carry | `2.7 N sideways, bearing stress 0.1 MPa` |
| Units and sanity | are the size, strain and deflection plausible | `peak strain 0.015, deflection 2.8% of size` |
| Large deflection | does a nonlinear solve agree | `the peak moves +2.3%, linear theory holds` |
| Load introduction | does the answer flip if the attached part is rigid | `rigid servo 55 MPa over 3 meshes, safety factor 3.00` |

## Mesh convergence

Wobbl compares the governing stress on the last two meshes. It passes when the change is
under the tolerance, 3% by default, 2% with `--quality fine` and 5% with `fast`, and when at
least three meshes are behind it, because two coarse meshes can agree and both be off. Two
meshes that agree give a warning, a change up to 10% gives a warning that says to expect about
that much error, and anything above 10% fails. When the number is still moving but even the
highest value seen on any mesh, plus the last swing, leaves a safety factor 20% above the
target, the warning adds that the verdict holds either way. A largest deflection that still
changes by more than 5% also lowers the confidence.

## Force balance

The reaction forces and moments at the supports must cancel the applied loads, gravity and
rigid attachments included. Within 0.5% passes, within 2% is a warning and anything more
fails, since the solution is then not in equilibrium. When rollers at an angle carry part of
the load through constraint equations, the check reports what the supports carry directly as
information instead.

## Hand calculation

Wobbl cuts the part with a plane through the governing point, square to the line toward the
load. When every support sits on one side of the cut and every load on the other, the section
carries the whole load like a beam, so Wobbl works out its area and second moments, the axial
force, bending moment and torsion, and the nominal stress. The FEA peak divided by that
nominal stress is the local stress concentration.

| Ratio | Status |
|---|---|
| 0.85 to 1.2 | pass, a plain beam region |
| 1.2 to 6 | pass, a stress concentration from the local shape such as a fillet or a hole |
| 0.6 to 0.85 | info, the part is stiffer than a plain beam there |
| anything else | warn, so check the loads, units and supports |

The check is skipped under self weight, which acts everywhere, and when no single section
carries the load, with the reason given.

## Singularities

Linear elasticity predicts infinite stress at a perfectly sharp inside corner, at the edge of
a perfectly rigid support, where a rigid attachment grabs the part and where a load is pushed
into a hole. Those peaks keep climbing as the mesh gets finer. Wobbl finds sharp inside
corners from the exact CAD normals, so a small fillet is not mistaken for a corner, and tracks
every peak through the mesh levels. The answer is taken from points at least a set distance
away, a fifth of the typical wall or the element size the sizing rules ask for at the peak,
whichever is larger. Both come from the geometry, so the distance is the same on every run.

The excluded peaks are listed with their trend, for example `15.2 -> 20.8 -> 20.6 MPa` at a
boss root. The check passes when nothing excluded is above the answer and is information when
something is. It becomes a warning when a sharp corner sits in a brittle material or an
excluded peak is more than one and a half times the answer, and a sharp corner in a brittle
material also puts a fillet on the fix list.

![Stress heatmap of a printed motor mount without its ribs, with a band of high stress along the base of the upright and two spots labeled artifact](/img/motor-mount-noribs.webp "Wobbl excludes the spots labeled artifact, where an idealized support or a sharp corner sets the stress, and reports 41.3 MPa from the fillet.")

## Bolt bearing

At bolted and fixed holes, the stress right at the hole is set by the support idealization,
so Wobbl reports the honest number instead. It takes the reaction force through each hole,
divides it by the projected area, diameter times length, and compares that with the bearing
limit, which is one and a half times yield for ductile metals and the plain limit otherwise.
The worst hole of each support is reported, and it passes when its safety factor meets the
target.

## Units and sanity

This check catches unit mistakes and setups outside linear theory. A part under 1 mm or over
20 m across, or a stress over 30 times or under a ten thousandth of the material's limit,
gives a warning. A largest deflection over 6% of the part size gives a warning because linear
results get less accurate, and over 20% fails.

## Large deflection

When the part bends at least 30% of its thinnest wall or 3% of its size, Wobbl solves it again
with large deflection theory in two load steps, on the first mesh or on a coarser mesh of about
20,000 nodes when the first one is big, and reports how far the peak and the deflection move.
Within 10% passes and linear theory holds. Beyond that it warns and scales the converged answer
by the ratio, saying whether the linear answer is conservative. On the
[drone arm](/examples/drone-arm/) the peak moves 2.3%, and on the enclosure shell in the
bake-off large deflection theory lowers it 11%. Studies with wall supports skip this check.

## Load introduction

Loads that act off the part reach it through a distributing coupling or a rigid attachment,
and the two can give very different peaks. Wobbl first solves the other idealization on the
final mesh, and when the verdict flips or the peak moves, it refines around that other peak
with the same convergence rule and time budget as the main study. If the budget runs out
first, the best mesh reached is labeled as not converged. The check passes when the other
peak is within 0.7 to 1.4 times the answer with the same verdict, is information when the
verdict is the same but the numbers differ more, and warns when the verdict flips.

## Heat checks

| Check | What it compares |
|---|---|
| Mesh convergence | the temperature rise at the hottest point, which has to settle within 2% over two meshes, or the coolest point when the hottest is a held temperature |
| Energy balance | in steady state, heat in against heat out by convection, radiation and held temperature faces, passing within 1% |
| Hand calculation | a lumped model of the whole part, against the surface average in steady state or the mean temperature over time in a transient, with the Biot number that says whether the part is nearly uniform |
| Units and sanity | that the temperatures are physical, with a note on how much heat radiation could shed when it is off and the part runs above 70 C |

## Circuit checks

| Check | What it compares |
|---|---|
| Hand calculation | the textbook answer, such as the diode equation solved by hand for an LED, `1 - e^-1` for an RC network, the ideal triangular ripple for a buck converter and `I_rms^2 R` for a MOSFET |
| Power balance | the supply's power against the sum of what the parts burn |
| Time step convergence | the same transient at half the time step, passing within 1% |
| Steady state | for a buck converter, that the average output stopped drifting between the last two 20 cycle windows |
| Thermal loop | for a MOSFET, the junction temperature with the on resistance rising as it heats, which fails on thermal runaway |
