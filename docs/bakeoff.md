# Bake off: a coding agent on its own versus Wobbl

Before building Wobbl we asked a plain question: if you give a capable coding
agent a CAD file and a strength question, does it answer as well as a careful,
converged simulation? Six parts were tested, two trials each, with a 15 minute
limit, against reference answers from hand built gmsh plus CalculiX models
converged over four to six meshes (up to 400,000 nodes, one to twenty minutes of
solving per mesh).

## What the agent did on its own

- It delivered an answer in **1 of 10** standard trials. That answer was a hand
  calculation on the simplest part (a flat bar arm), 2% from the reference, in 75 s.
- In the other 9 trials it started writing its own finite element code (mesher,
  element kernel, solver, validation test) and was stopped at 15 minutes with
  nothing to show the user. Twice it already held a correct number internally.
- With "hand calculations only", it was within 1 to 5% on beam like parts, 29%
  low on a bracket with a sharp inside corner, and still gave **no answer in 15
  minutes** on the ribbed cast motor mount and the thin walled enclosure, the
  two parts that actually fail.
- It hit the same traps every time: gmsh's high order optimizer running for
  minutes or crashing on thin curved walls, singular peaks at bonded bolt holes,
  and load idealizations that flip the verdict.

So the physics knowledge is there. What is missing is a fast, reliable way to
turn it into a checked number. That is what Wobbl is for.

## What Wobbl does on the same parts

Each part runs through `wobbl run` with the loads and supports from the prompt,
on a shared 2018 laptop (6 core i7, 16 GB). Times are wall clock for the whole
study, including every check and the heatmaps.

| Part | Reference | Wobbl | Difference | Wobbl's confidence | Time |
|---|---|---|---|---|---|
| Cast motor mount, 2 kN belt pull through four bolts | 124.3 MPa at the +X rib tip | 123.9 MPa at the same spot | 0.3% | high | about 2.5 min |
| ABS enclosure shell, 250 N near a corner | 93 MPa at the corner boss fillet, 1.93 mm deflection | 92.5 MPa at the same fillet, 1.92 mm | 0.5%, 0.6% | high | about 13 min (thin walls, 166k nodes) |
| PA12 truss drone arm, 40 N thrust and 0.8 N m | 11.5 MPa at the first window corner, motor rises 4.09 mm | 11.49 MPa at the same corner, 4.08 mm | 0.1%, 0.4% | high | about 12 min |
| PLA bracket, 0.5 mm inside corner, wall contact | about 78 MPa (77.5, still moving 5%) | 79.8 MPa | 2.4% | high | 43 s |
| PLA bracket, 6 mm fillet, wall contact | 25.3 MPa | 25.7 MPa | 1.4% | medium (the bonded screw bore reads 64 MPa, excluded and named) | 38 s |
| 6061 bar arm with root fillet and hole | 165.1 MPa (23,949 psi) | 165.0 MPa | 0.1% | high | about 1 to 2 min |

The references took one to ten minutes of solving per mesh on their finest
meshes, plus hours of setup by hand; the agent on its own produced nothing on
the motor mount and enclosure in 15 minutes.

## The traps, and what Wobbl does about each

| Trap found in the bake off | What Wobbl does by default |
|---|---|
| A run that hits a time limit returns nothing | every study has a budget; when it runs out, the last finished mesh level comes back with its checks, confidence and heatmap |
| gmsh curved quadratic elements fold on thin curved walls, and its optimizer runs for minutes or crashes | folded elements are detected and that mesh switches to straight sided quadratic elements at once; every mesh attempt runs in a child process with a timeout and falls back to other algorithms and finer sizes |
| Bonded bolt bores and fixed face edges look like the peak | zones next to supports, rigid attachments and loads pushed into holes are excluded from the reported value, listed with their trend, and bolted holes get a bearing stress check instead |
| A belt pull at a pulley or thrust and torque on a motor have no face to sit on | remote loads go in through an RBE3 style distributing coupling or a CalculiX rigid body, and both force and moment balance are checked |
| Rigid servo versus bolt group load share flipped the motor mount's verdict (36 versus 124 MPa) | Wobbl re-solves with the other attachment idealization, reports the spread, and lowers confidence when the verdict depends on it |
| Sharp re-entrant CAD edges never converge | sharp inside corners are found from exact CAD normals (fillets and blends are recognized as smooth), peaks there are labeled singular with their trend, and the fix list gives a fillet radius |
| Uniform refinement went to 3.5 million nodes and swapped | a coarse first mesh, then refinement only around the peaks that matter, with node and time limits |
| Wall contact did not settle on the coarsest mesh | a coarse linear pre solve finds the contact patch and every level starts from the previous one |
