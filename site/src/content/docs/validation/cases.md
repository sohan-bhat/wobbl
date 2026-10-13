---
title: "How the cases are set up"
description: "The part, the loads, the supports and the reference answer behind each of Wobbl's 19 validation cases."
lead: "Each case builds its part, meshes it, refines until the answer settles and runs its checks, and the number is then compared with a known answer."
order: 1
---

## Textbook formulas

The cantilever cases use a 200 x 20 x 10 mm steel bar clamped at one end with 100 N at the
other. The tip deflection is compared with Timoshenko beam theory, which counts both bending and
shear, and the stress on the top surface halfway along with `M c / I`. The clamped end makes the
3D model slightly stiffer than beam theory, by under 1%.

The plate with a hole is 400 x 200 x 4 mm with a 20 mm hole, pulled at 10 MPa and held by
rollers on three faces. The peak at the hole is compared with Kirsch's solution with Peterson's
finite width correction, a stress concentration of 3.035 on the gross section.

The thick walled pressure vessel is a quarter of a tube with radii of 50 and 100 mm under
100 MPa of internal pressure, in plane strain with symmetry rollers on all the cut faces. Its
von Mises stress at the bore is compared with Lame's solution.

The torsion case is a 20 mm steel shaft, 100 mm long, clamped at one end with 50 N m applied
as a distributed moment at the other. The surface von Mises stress is compared with
`sqrt(3) x 16 T / (pi d^3)`.

The self weight case is a 1 m steel cantilever, 20 x 20 mm, under 1 g, and its tip deflection is
compared with `q L^4 / (8 E I)`.

## Industry benchmarks

NAFEMS LE1 is the elliptic membrane, a quarter of an elliptical ring with 10 MPa of outward
pressure on its outer edge and symmetry on both axes. sigma_yy at point D, at (2000, 0), is
compared with the target of 92.7 MPa.

NAFEMS LE10 is the thick plate, a 600 mm thick elliptical ring under 1 MPa on its top face,
with the outer edge held in x and y and the outer mid plane line held in z, which Wobbl models
as a 2 mm band. sigma_yy at point D on the top surface is compared with the target of
-5.38 MPa.

## Real parts from the bake-off

These are the parts from the [bake-off](/validation/bakeoff/). A sand cast motor mount carries a
belt pull off the part through four bolts. A molded ABS enclosure shell with bosses, a snap
tongue and a USB opening is pressed near a corner. A PA12 drone arm with Warren truss windows
takes thrust and torque from a rigid motor. A PLA L bracket, once with a sharp 0.5 mm inside
corner and once with a 6 mm fillet, is screwed to a wall it can lift off.

The references are gmsh and CalculiX models built by hand and refined over four to six meshes
of up to 400,000 nodes. Their tolerances follow how settled each reference was, so the sharp
bracket, whose reference was still moving 5% on its last mesh, gets 8%. The parts ship with
Wobbl in `src/wobbl/validation/parts/`.

## Heat

The slab is a 100 mm aluminum bar held at 100 C on one end and cooled by 50 W/(m2 K) air at
20 C on the other, insulated elsewhere. The temperature of the cooled face is compared with the
series resistance `L/k + 1/h`.

The lumped case is a 20 mm aluminum cube at 100 C cooling in 10 W/(m2 K) air for ten minutes.
Its mean temperature is compared with lumped capacitance, which is exact to well under 1% here
because the Biot number is 2e-4.

The sudden face case is a 100 mm aluminum bar at 20 C whose end is held at 120 C from the start,
insulated elsewhere. The temperature rise 10 mm in after 2 s is compared with the erfc solution
for a semi infinite solid, `100 erfc(x / (2 sqrt(alpha t)))`, which is exact here because the
heat has not reached the far end yet. Like every held face that starts away from the part's
temperature, the end ramps to 120 C over a few milliseconds instead of jumping at once.

## Circuits

A red LED on 5 V through 150 ohm is compared with the diode equation solved by Newton iteration
with the same model parameters. An RC network's voltage at one time constant is compared with
`1 - e^-1`. A 12 V to 5 V buck converter's output ripple is compared with the ideal triangular
ripple current through the output capacitor and its ESR.
