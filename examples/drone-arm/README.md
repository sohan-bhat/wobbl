# Drone arm

Is a printed PA12 drone arm strong enough at full throttle?

This quadcopter arm is printed in PA12 by Multi Jet Fusion, a 150 mm tapered beam with three
lightening windows, bolted into the frame through two holes in its root block. The motor sits on
a 30 mm pad at the tip, and at full throttle it pushes up 25 N and twists the pad with 0.5 N m of
reaction torque. The motor is stiff, so both loads go in through a rigid attachment on the pad.

The rule here for printed parts that vibrate is a safety factor of 3.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress | 9.46 MPa on the underside of the arm below the first window, safety factor 5.1 against 48 MPa (meets the rule of 3) |
| Tip deflection | 4.2 mm |
| Mesh convergence | 9.44, 9.53, 9.46 MPa over three meshes |
| Hand calculation | beam theory through the critical section gives 6.69 MPa, and the local shape concentrates it 1.4x |
| Equilibrium | force and moment balance to 0.0002% |
| Large deflection | the arm bends a good share of its wall, so Wobbl also solves it with large deflection theory, which moves the peak +2.3% and shows linear theory holds |
| Load attachment | spread over the pad instead of rigid, the peak is the same 9.46 MPa at the same spot |

The windows are cut with square corners, and Wobbl excludes those corners from the answer as
singular (they read 14.4 MPa on the finest mesh and depend on the mesh). Even that value meets
the rule, but because it is more than one and a half times the answer the result is marked
medium confidence. A 1 to 2 mm radius in the window corners is cheap in a printed part and makes
the number trustworthy.

For a harder version of this question, a Warren truss arm, see the `drone-arm-truss`
[validation case](https://sohan-bhat.github.io/wobbl/validation/cases/), which matches a
converged reference model to within 0.2%.
