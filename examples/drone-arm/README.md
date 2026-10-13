# Drone arm: printed PA12 at full throttle

A quadcopter arm printed in PA12 by Multi Jet Fusion: a 150 mm tapered beam with
three lightening windows, bolted into the frame through two holes in its root
block. The motor sits on a 30 mm pad at the tip. At full throttle it pushes up
25 N and twists the pad with 0.5 N m of reaction torque; the motor is stiff, so
both loads go in through a rigid attachment on the pad.

The rule for printed parts that vibrate here is a safety factor of 3.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress | 10.1 MPa at the top edge of the first window, safety factor 4.8 against 48 MPa (meets the rule of 3) |
| Tip deflection | 4.2 mm |
| Mesh convergence | 10.3, 10.0, 10.1 MPa over three meshes |
| Hand calculation | beam theory through the critical section gives 7.1 MPa; the window corner concentrates it 1.4x |
| Equilibrium | force and moment balance to 0.0001% |

The windows are cut with square corners, and Wobbl excludes those corners from
the answer as singular (they read 14.4 MPa on the finest mesh and depend on the
mesh). Even that value meets the rule, but a 1 to 2 mm radius in the window
corners is cheap in a printed part and makes the number trustworthy.

For a harder version of this question (a Warren truss arm), see the
`drone-arm-truss` validation case, which matches a converged reference model
to within 0.4%.
