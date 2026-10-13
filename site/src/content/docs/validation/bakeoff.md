---
title: "Bake-off"
description: "A coding agent on its own against Wobbl on six real parts, each with a CAD file, a strength question and a 15 minute limit."
lead: "Six parts, a strength question for each and a time limit, answered by a coding agent on its own and by Wobbl."
order: 2
wide: true
---

A coding agent got six parts to check, each with a CAD file and a question about whether it
would hold. It had two tries per part and 15 minutes for each, and every answer was compared
against a careful reference simulation that was refined until the numbers stopped changing.

On its own, the agent answered only once in ten tries. That answer was a quick hand calculation
on the simplest part, a flat bar arm, and it landed within 2% in about a minute. Every other
time it started writing its own simulation code and ran out of time, even though twice it
already had the right number and just never reported it.

When it was told to stick to hand calculations, it did well on simple beam shapes and got within
1 to 5%, but it came in 29% low on a bracket with a sharp inside corner. On the two parts that
actually fail, the ribbed motor mount and the thin enclosure, it still had nothing after 15
minutes.

Wobbl ran the same parts with the same loads on a 2018 laptop, and the times below include every
check.

| Part | Reference | Wobbl | Difference | Confidence | Time |
|---|---|---|---|---|---|
| Cast motor mount, a 2 kN belt pull through four bolts | 124.3 MPa at the +X rib tip | 123.7 MPa at the same spot | 0.5% | Medium, because the part passes if the servo is treated as rigid, and Wobbl points that out | About 6.6 minutes, with the rigid servo case converged too |
| ABS enclosure shell, 250 N near a corner | 93 MPa at the corner boss fillet and 1.93 mm of deflection | 93.0 MPa at the same fillet and 1.92 mm | 0.02% and 0.6% | Medium, because large deflection theory lowers the peak 11%, and Wobbl says so in the result | About 6 minutes, since the thin walls need 166k nodes |
| PA12 truss drone arm, 40 N of thrust and 0.8 N m of torque | 11.5 MPa at the first window corner, with the motor rising 4.09 mm | 11.49 MPa at the same corner and 4.08 mm | 0.1% and 0.3% | High | About 7 minutes |
| PLA bracket with a 0.5 mm inside corner, resting on a wall | About 78 MPa, from a reference still moving 5% at 77.5 | 80.5 MPa | 3.2% | High | 38 seconds |
| PLA bracket with a 6 mm fillet, resting on a wall | 25.3 MPa | 26.4 MPa | 4.3% | Medium, because the bonded screw bore reads 48.9 MPa, which Wobbl sets aside and names | 24 seconds |
| 6061 bar arm with a root fillet and a hole | 165.1 MPa (23,949 psi) | 164.2 MPa (23.8 ksi) | 0.5% | High | 52 seconds |

The laptop has a six core i7 and 16 GB of memory. The first five parts are cases of
[`wobbl validate`](/guides/cli/validate/), and the bar arm was a plain `wobbl run`. The
references took one to ten minutes of solving per mesh on their finest meshes, on top of hours of
setup by hand.

## The traps

Every time the agent tried to build its own simulation, it ran into the same few problems. Its
mesher folded elements or stalled on thin curved walls, its peaks landed on bonded bolt holes where the
stress has no real value, and the way it attached the loads changed the verdict. Wobbl handles
each of these by default.

| Trap | What Wobbl does |
|---|---|
| A run that hits its time limit returns nothing | Every study has a budget, and when it runs out the last finished mesh level comes back with its checks, its confidence and its heatmap |
| gmsh's curved quadratic elements fold on thin curved walls, and its optimizer runs for minutes or crashes | Folded elements are caught and that mesh switches to straight sided quadratic elements at once, and every mesh attempt runs in a child process with a timeout and falls back to other algorithms and finer sizes |
| Bonded bolt bores and the edges of fixed faces look like the peak | Zones next to supports, rigid attachments and loads pushed into holes are kept out of the reported value and listed with their trend, and bolted holes get a bearing check instead |
| A belt pull at a pulley or the thrust and torque of a motor have no face to sit on | Remote loads go in through a distributing coupling like an RBE3 or a CalculiX rigid body, and both the force and the moment balance are checked |
| Treating the servo as rigid or as a bolt group flipped the motor mount's verdict | Wobbl solves the other idealization too, refines it around its own peak until it settles, reports the spread and lowers the confidence when the verdict depends on it |
| Sharp inside corners from the CAD never converge | Sharp corners are found from exact CAD normals, so fillets and blends count as smooth, the peaks there are labeled singular with their trend, and the fix list gives a fillet radius |
| Refining the whole mesh went to 3.5 million nodes and pushed the laptop into swap | Wobbl starts coarse and refines only around the peaks that matter, within node and time limits |
| Wall contact did not settle on the coarsest mesh | A coarse linear solve finds the contact patch first, and every mesh level starts from the one before |

The bake-off's own reference for the motor mount held the servo rigid with bonded bolts and got
36 MPa. That is a different idealization from Wobbl's rigid variant, which converges to
55.0 MPa. Both sit well below the 124 MPa of the bolt group, which Wobbl uses by default and
which a servo flange on four bolts behaves much closer to.
