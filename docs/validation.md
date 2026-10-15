# Validation

The results now live on the website, at
[sohan-bhat.github.io/wobbl/validation](https://sohan-bhat.github.io/wobbl/validation/), with
the setup of each case on
[its own page](https://sohan-bhat.github.io/wobbl/validation/cases/).

Ninety five cases with known answers run through the same pipeline you use. Strength has 14 of them,
covering textbook formulas, the NAFEMS LE1 and LE10 benchmarks and five real parts against converged
reference models. The rest cover chemistry (13), circuits (9), motion (6), mechanisms (6), uncertainty (5),
fracture (5), manufacturing (5), vibration (4), combustion (4), motors (4), heat (3), plasticity (2), boards
(2), sound (2), and one each for buckling, fatigue, batteries, contact, bolted joints, magnetics,
electrostatics, antennas, drops, shells and composites. Every case passes its tolerance, and every case
whose reference is exact for the modeled setup lands within 2%. `wobbl validate` runs them all in about
30 minutes. The raw numbers are in [validation-results.json](validation-results.json).
