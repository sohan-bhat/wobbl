# Cast motor mount: does it meet a safety factor of 2?

A sand cast A356-T6 aluminum mount for a servo driving a conveyor belt (yield
165 MPa, and a house rule of safety factor 2 on yield for castings because of
porosity). The base bolts flat to a steel frame. The servo bolts to the upright
with four M6 bolts; its pulley sits 30 mm behind the upright, and an emergency
stop pulls the belt with 2 kN, 30 degrees below horizontal. This is one of the
parts a coding agent working alone could not answer in 15 minutes
([docs/bakeoff.md](../../docs/bakeoff.md)).

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress | 123.4 MPa where the +X rib tip meets the upright (reference model: 124.3 MPa, same spot) |
| Safety factor | 1.34, below the required 2, so the casting **fails the rule** (it does not yield) |
| Mesh convergence | 121.9, 123.5, 123.4 MPa over three meshes, about 1.5 minutes in all |
| Equilibrium | forces balance to 0.00002%, moments to 0.000003% |

**The answer depends on how the servo carries the load, and Wobbl says so.** The
belt pull is a force at a point off the part. Wobbl's default spreads it over the
four bolt holes without adding stiffness (a distributing coupling, like a bolt
group sharing load). Treating the servo body as rigid instead moves the load
into the bolt holes and leaves the rib at 47 MPa, safety factor 3.5, which
passes. Wobbl solves both, reports the spread, and marks the result medium
confidence with that reason. A plain number would have hidden that the verdict
flips with the assumption; in practice a servo flange on four bolts behaves much
closer to the bolt group, so the part should be redesigned (thicker ribs or a
larger blend at the rib tip).
