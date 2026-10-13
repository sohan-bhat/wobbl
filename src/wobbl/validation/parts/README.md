# Validation parts

Real parts from a bake off between a plain coding agent and careful, hand built
reference models (gmsh plus CalculiX, converged over four to six meshes of up to
400,000 nodes). Units are millimeters.

| File | Part | Load and supports | Reference |
|---|---|---|---|
| `cast_motor_mount.step` | sand cast A356-T6 ribbed motor mount | base underside fixed; 2 kN belt pull at a pulley 30 mm behind the upright, shared by four flange bolts | 124.3 MPa von Mises at the +X rib tip, 0.266 mm max deflection |
| `enclosure_shell.step` | molded ABS shell with bosses, snap tongue and USB opening | rim and boss ends fixed, snap tongue free; 250 N on a 30 mm circle near one corner | 93 MPa von Mises at the corner boss fillet, top deflects 1.93 mm |
| `drone_arm_truss.step` | MJF PA12 drone arm with Warren truss windows | hub back face fixed; 40 N thrust and 0.8 N m torque on a rigid motor | 11.5 MPa von Mises at the first top chord window corner, motor rises 4.09 mm |
| `shelf_bracket_sharp.step` | PLA L bracket, 0.5 mm inside corner | screw bonded in its hole, back face on a wall (contact); 60 N shelf load | 77.5 MPa max principal at the corner (about 78, plus or minus 5%) |
| `shelf_bracket_fillet.step` | the same bracket with a 6 mm fillet | same | 25.3 MPa max principal on the fillet |

`wobbl validate --only cast-motor-mount,enclosure-shell,drone-arm-truss,bracket-sharp,bracket-fillet`
runs them.
