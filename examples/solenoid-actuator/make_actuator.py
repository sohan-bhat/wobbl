"""Write actuator.step: a pull solenoid as separate named solids, the way a CAD tool exports an assembly.

A square steel frame wraps a copper coil on a bobbin, and a steel plunger sits 3 mm short of the pole piece at the
bottom of the bore. Sizes in mm.

    python make_actuator.py
"""
from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent


def build(occ):
    frame_out = occ.addBox(-20, -20, 0, 40, 40, 44)
    frame_in = occ.addBox(-16, -20, 4, 32, 40, 36)          # open on two sides, a C frame seen from the front
    frame, _ = occ.cut([(3, frame_out)], [(3, frame_in)])
    bore = occ.addCylinder(0, 0, 38, 0, 0, 8, 6.3)          # hole in the top plate for the plunger
    frame, _ = occ.cut(frame, [(3, bore)])
    pole = occ.addCylinder(0, 0, 4, 0, 0, 12, 6)            # fixed pole piece on the bottom plate
    frame, _ = occ.fuse(frame, [(3, pole)])
    c_out = occ.addCylinder(0, 0, 5, 0, 0, 34, 15)
    c_in = occ.addCylinder(0, 0, 5, 0, 0, 34, 7)
    coil, _ = occ.cut([(3, c_out)], [(3, c_in)])
    plunger = occ.addCylinder(0, 0, 19, 0, 0, 34, 6)         # 3 mm above the pole, through the top plate
    return [frame[0][1], coil[0][1], plunger]


if __name__ == "__main__":
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("actuator")
    tags = build(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    for t, name in zip(tags, ("Frame", "Coil", "Plunger")):
        gmsh.model.setEntityName(3, t, name)
    gmsh.write(str(HERE / "actuator.step"))
    gmsh.finalize()
