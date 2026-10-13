# Validation

18 cases with known answers, run through the same pipeline you get.

Wobbl builds each part, meshes it, refines until the answer settles and runs its
checks, and the number is compared with a textbook or benchmark value. Run them
with:

```sh
wobbl validate                 # all cases, about 30 minutes
wobbl validate --only lame,led # a subset
```

## Results

| Case | Quantity | Reference | Expected | Wobbl | Error | Tolerance | Wobbl's confidence | Result |
|---|---|---|---|---|---|---|---|---|
| Cantilever tip deflection | tip deflection | Timoshenko beam theory | 0.8016 mm | 0.7956 mm | 0.74% | 3.0% | high | pass |
| Cantilever bending stress at mid span | sigma_xx at mid span | Euler-Bernoulli, M c / I | 30 MPa | 29.94 MPa | 0.21% | 3.0% | high | pass |
| Plate with a hole in tension | peak von Mises at the hole | Kirsch with Peterson finite width Kt | 30.35 MPa | 29.89 MPa | 1.54% | 4.0% | high | pass |
| Thick walled pressure vessel | von Mises at the bore | Lame, plane strain | 231.3 MPa | 232.3 MPa | 0.43% | 3.0% | high | pass |
| NAFEMS LE1 elliptic membrane | sigma_yy at point D | NAFEMS LE1 target 92.7 MPa | 92.7 MPa | 92.92 MPa | 0.23% | 4.0% | high | pass |
| NAFEMS LE10 thick plate under pressure | sigma_yy at point D | NAFEMS LE10 target -5.38 MPa | -5.38 MPa | -5.412 MPa | 0.59% | 5.0% | medium | pass |
| Round shaft in torsion | von Mises at the surface | tau = 16 T / (pi d^3) | 55.13 MPa | 55.6 MPa | 0.85% | 3.0% | high | pass |
| Cantilever under its own weight | tip deflection | q L^4 / (8 E I) | 1.443 mm | 1.441 mm | 0.16% | 3.0% | high | pass |
| Cast motor mount with a belt pull through four bolts | peak von Mises at the rib tip | converged reference FEA, 278k nodes | 124.3 MPa | 123.7 MPa | 0.49% | 5.0% | medium | pass |
| ABS enclosure shell pressed near a corner | peak von Mises at the corner boss | converged reference FEA, 281k nodes | 93 MPa | 92.98 MPa | 0.02% | 10.0% | medium | pass |
| PA12 truss drone arm at full thrust | peak von Mises at the first window corner | converged reference FEA, 403k nodes | 11.5 MPa | 11.49 MPa | 0.11% | 8.0% | high | pass |
| PLA shelf bracket, 0.5 mm inside corner | max principal at the corner | converged reference FEA with wall contact | 78 MPa | 80.48 MPa | 3.18% | 8.0% | high | pass |
| PLA shelf bracket, 6 mm fillet | max principal at the fillet | converged reference FEA with wall contact | 25.3 MPa | 26.38 MPa | 4.26% | 5.0% | medium | pass |
| Conduction through a slab into convection | cooled face temperature | 1D series resistance | 97.67 C | 97.67 C | 0.00% | 1.0% | high | pass |
| Aluminum cube cooling in air | mean temperature after 10 min | lumped capacitance (Biot 2e-4) | 58.01 C | 58.27 C | 0.44% | 1.0% | high | pass |
| Red LED on 5 V through 150 ohm | LED current | diode equation solved by Newton | 20 mA | 20 mA | 0.00% | 0.5% | high | pass |
| RC charging | voltage at one time constant | V (1 - e^-1) at t = RC | 3.161 V | 3.161 V | 0.00% | 0.2% | high | pass |
| Buck converter output ripple | peak to peak ripple | ideal triangular current through C and ESR | 7.949 mV | 8.561 mV | 7.70% | 15.0% | high | pass |

The full run took 29 minutes on a 2018 six core laptop (Intel i7-8750H), and
every number above comes from it. Every case whose reference is exact for the
modeled setup lands within 2%. Medium confidence is Wobbl telling you what the
number leans on, not a miss: on LE10 the 2 mm band that stands in for the
benchmark's line support makes a peak far above the answer (excluded and
named), on the cast motor mount the verdict flips if the servo is treated as
rigid instead of a bolt group, on the enclosure shell large deflection theory
lowers the peak by 11% (the linear answer is the conservative one, and the
reference is linear too), and on the filleted bracket the bonded screw bore
reads nearly twice the real peak.
The buck converter's reference is the ideal waveform formula, which ignores the
switch and diode drops that ngspice models, so it gets a wider tolerance.

## The cases

**Cantilever tip deflection and mid span stress.** A 200 x 20 x 10 mm steel bar
clamped at one end with 100 N at the other. Deflection against Timoshenko beam
theory (bending plus shear), stress at the top surface half way along against
`M c / I`. The clamped end makes the 3D model slightly stiffer than beam theory,
by under 1%.

**Plate with a hole.** A 400 x 200 x 4 mm plate with a 20 mm hole, pulled at 10
MPa, held by rollers on three faces. The peak at the hole against Kirsch's
solution with Peterson's finite width correction, Kt = 3.035 on the gross
section.

**Thick walled pressure vessel.** A quarter of a 50/100 mm radius tube under 100
MPa internal pressure in plane strain (symmetry rollers on all cut faces). Von
Mises stress at the bore against Lame's solution.

**NAFEMS LE1.** The elliptic membrane benchmark: a quarter of an elliptical ring,
10 MPa outward pressure on the outer edge, symmetry on both axes. sigma_yy at
point D (2000, 0) against the target 92.7 MPa.

**NAFEMS LE10.** The thick plate benchmark: a 600 mm thick elliptical ring under
1 MPa on its top face, outer edge held in x and y, the outer mid plane line held
in z (modeled as a 2 mm band). sigma_yy at point D on the top surface against the
target -5.38 MPa.

**Torsion.** A 20 mm round shaft, 100 mm long, clamped at one end with 50 N m
applied as a distributed moment at the other. Surface von Mises stress against
`sqrt(3) x 16 T / (pi d^3)`.

**Self weight.** A 1 m steel cantilever, 20 x 20 mm, under 1 g. Tip deflection
against `q L^4 / (8 E I)`.

**Slab conduction.** A 100 mm aluminum bar held at 100 C on one end and cooled
by 50 W/(m2 K) air at 20 C on the other, insulated elsewhere. Temperature of the
cooled face against the series resistance `L/k + 1/h`.

**Lumped cooling.** A 20 mm aluminum cube at 100 C cooling in 10 W/(m2 K) air for
ten minutes. Mean temperature against lumped capacitance (Biot number 2e-4, so
the lumped answer is exact to well under 1%).

**Real parts.** A sand cast motor mount loaded through four bolts by a belt pull
off the part, a molded ABS enclosure shell with bosses, a snap tongue and a USB
opening pressed near a corner, a PA12 drone arm with Warren truss windows under
thrust and torque from a rigid motor, and a PLA L bracket with a sharp (0.5 mm)
and a filleted (6 mm) inside corner, screwed to a wall it can lift off. The
references are hand built gmsh plus CalculiX models converged over four to six
meshes of up to 400,000 nodes (see [bakeoff.md](bakeoff.md)); the parts ship in
`src/wobbl/validation/parts/`. Their tolerances reflect the references' own
convergence (the sharp bracket's reference was still moving 5% on its last mesh).

**LED, RC, buck.** A red LED on 5 V through 150 ohm against the diode equation
solved by Newton iteration with the same model parameters; an RC network's
voltage at one time constant against `1 - e^-1`; a 12 V to 5 V buck converter's
output ripple against the ideal triangular ripple current through the output
capacitor and its ESR.

## Real parts

Wobbl was also run on the harder parts of a bake off where reference answers
came from hand built, carefully converged models (gmsh plus CalculiX, meshes up
to 400,000 nodes). See [bakeoff.md](bakeoff.md).

The raw results are in [validation-results.json](validation-results.json).
