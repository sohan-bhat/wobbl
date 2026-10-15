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
