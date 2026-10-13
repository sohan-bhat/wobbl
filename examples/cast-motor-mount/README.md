# Cast motor mount

Does a cast motor mount meet a safety factor of 2 under a 2 kN belt pull?

This is a sand cast A356-T6 aluminum mount for a servo driving a conveyor belt, with a yield
strength of 165 MPa and a house rule of safety factor 2 on yield for castings because of
porosity. The base bolts flat to a steel frame, and the servo bolts to the upright with four M6
bolts. Its pulley sits 30 mm behind the upright, and an emergency stop pulls the belt with 2 kN,
30 degrees below horizontal. This is one of the parts a coding agent working alone could not
answer in 15 minutes in the
[bake-off](https://sohan-bhat.github.io/wobbl/validation/bakeoff/).

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress | 123.7 MPa where the +X rib tip meets the upright, and the reference model gives 124.3 MPa at the same spot |
| Safety factor | 1.33, below the required 2, so Wobbl calls it marginal because the casting does not yield but fails the house rule |
| Mesh convergence | 122.2, 123.6, 123.7 MPa over three meshes |
| Rigid servo case | 54.2, 54.3, 55.0 MPa over three meshes refined around its own peak, safety factor 3.00 |
| Equilibrium | forces balance to 0.00001%, moments to 0.000001% |
| Time | 5 minutes in all on a 2018 six core laptop, the rigid servo case included |
| Repeatability | four runs back to back gave these numbers to the last digit |

The answer depends on how the servo carries the load, and Wobbl says so. The belt pull is a
force at a point off the part, and Wobbl's default spreads it over the four bolt holes without
adding stiffness, the way a bolt group shares a load. Treating the servo body as rigid instead
moves the peak to the face the servo bolts against, about 2 mm outside a lower bolt hole, where
it converges to 55.0 MPa with a safety factor of 3.00, which passes. Wobbl solves both, refines
the rigid case around its own peak until it settles, reports the spread and marks the result
medium confidence with that reason. A servo flange on four bolts behaves much closer to the
bolt group in practice, so the part should be redesigned with thicker ribs or a larger blend at
the rib tip.
