# Limits

This page now lives on the website, at
[sohan-bhat.github.io/wobbl/validation/limits](https://sohan-bhat.github.io/wobbl/validation/limits/).
The summary here matches the code as it is today.

## Materials and geometry

- Materials are linear elastic, so the strength study reports whether yield is reached but not what
  happens after. Permanent set and the collapse load have their own physics, `--physics plasticity`.
- Each study uses one material.
- Large deflection is checked with a nonlinear re-solve of the part.
- A thin walled part goes to a shell model on its midsurface, checked against a coarse solid. `--shell`
  asks for that outright, and `--solid` keeps the solid.
- Printed plastics are treated as isotropic at 100 percent infill, with a layer strength knockdown when
  the print direction is given.
- Elastic supports are linear springs with the same stiffness in every direction, so they pull on the
  part as readily as they push.

## Dynamics, fatigue and fracture

- A drop is one landing on a floor. Its peak force rests on a contact spring set by Hertz theory, so
  quote it as an estimate.
- Buckling is linear. Knockdowns from test data stand in for imperfections.
- Vibration is linear with one damping ratio for every mode.
- Fatigue covers crack initiation by the stress life method.
- Crack growth is linear elastic fracture mechanics. Crack tip plasticity, crack faces pressing on each
  other and the environment are not modeled, and plastics need measured growth constants.
- Contact between bodies is its own physics, `--physics contact`, with friction. Bolted joints take their
  preload from the torque, 70 percent of proof load unless you say otherwise, and are checked as VDI 2230
  does.

## Motion, flight and motors

- Motion is a point mass in a vertical plane over a flat Earth. It has no roll, no pitch oscillation and
  no sideways wind, and Barrowman's stability margin holds below about Mach 0.8.
- Solid motors burn by their published burn rate laws, with every surface lit at once. Erosive burning,
  throat erosion and slag are flagged but not modeled. Against two MIT static fires the predicted peak
  pressure was 7 percent low on one and 12 percent high on the other.

## Mechanisms

- Mechanisms are rigid bodies with stiff spring contacts and joints without backlash.
- Servos follow their move as a well tuned controller would, and motors follow a straight torque speed
  line. A rotor inertia is assumed where a datasheet gives none.
- A hobby servo's continuous rating is taken as a third of its stall torque.

## Antennas

- A straight wire dipole or a monopole on an infinite plane is solved as a body of revolution, and a patch asked
  only for its size comes from the transmission line model without a solve. Everything else is a 3D full wave
  solve: first order edge elements on a tetrahedral mesh, copper as perfect conductors of zero thickness, free
  space closed by an absorbing sphere (a second order condition), a lumped port that is a strip carrying 1 A.
- The solve is slow. A normal run is 12 or so frequencies on 80 to 130 thousand unknowns and takes 25 to 60
  minutes here. The run predicts that time from its mesh before solving, gives up frequencies and then mesh
  fineness when it is over its budget (15 minutes unless `--set budget=` says otherwise) and refuses when it is
  far over.
- Resonances come out 2 to 3% low at normal quality against independent axisymmetric solves of the same shapes (a
  dipole 2.6% low, a monopole on a finite plate 3.2% low, a disk patch on FR4 1.4% low on a coarser mesh). Part of
  that is the mesh, which the finer mesh check measures and extrapolates, and the rest is the feed strip.
  A trace on a board with 0.8 mm copper can be worse (a 1.2% shift on the finer mesh, 13% in impedance).
- The transmission line model of a patch is optimistic. For a wide patch on FR4 the 3D solve resonates 7.5% below
  it, and for Balanis's 10 GHz example 15% below. The solve itself is within 3% of independent solves of disk
  patches and within 0.5% of a closed cavity, so the model's 2 to 4% claim does not hold for these patches. No
  independent full wave reference for a rectangular patch was available to settle which is right.
- Vias, conductor loss, copper thickness and connectors are not modeled, and a board's copper is one sheet at the
  middle of its layer.
