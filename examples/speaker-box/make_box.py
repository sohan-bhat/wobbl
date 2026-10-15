"""Write box.step: a closed loudspeaker cabinet of 18 mm board, 240 x 200 x 360 mm outside, with a cross brace
inside that has a 90 mm hole through it, the way braces are cut to stiffen the walls without splitting the air.

    python make_box.py
"""
from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box")
    occ = gmsh.model.occ
    outer = occ.addBox(0, 0, 0, 240, 200, 360)
    inner = occ.addBox(18, 18, 18, 204, 164, 324)
    shell, _ = occ.cut([(3, outer)], [(3, inner)])
    brace = occ.addBox(18, 18, 210, 204, 164, 18)
    hole = occ.addCylinder(120, 100, 205, 0, 0, 28, 45)
    brace, _ = occ.cut([(3, brace)], [(3, hole)])
    cabinet, _ = occ.fuse(shell, brace)
    occ.synchronize()
    gmsh.write(str(HERE / "box.step"))
    gmsh.finalize()
