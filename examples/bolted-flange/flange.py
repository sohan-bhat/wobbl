"""Two steel flanges bolted face to face, the kind that joins two lengths of pipe or a shaft to a hub.

Each flange is a disk with a bore through its middle and a ring of bolt holes; the lower one sits on z = 0 and the
upper one on top of it. They are two separate bodies, so they can press together and open apart. Coordinates in mm.
"""
import math

PARAMS = {
    "diameter": 120.0,
    "thickness": 14.0,
    "bore": 50.0,
    "pcd": 90.0,          # pitch circle of the bolt holes
    "holes": 4,
    "hole": 9.0,          # medium clearance for M8 (ISO 273)
}


def build(occ, diameter, thickness, bore, pcd, holes, hole):
    for k in range(2):
        z = k * thickness
        disk = occ.addCylinder(0, 0, z, 0, 0, thickness, diameter / 2)
        cut = [(3, occ.addCylinder(0, 0, z - 1, 0, 0, thickness + 2, bore / 2))]
        for j in range(int(holes)):
            a = math.pi / int(holes) + 2 * math.pi * j / int(holes)
            cut.append((3, occ.addCylinder(pcd / 2 * math.cos(a), pcd / 2 * math.sin(a), z - 1, 0, 0,
                                           thickness + 2, hole / 2)))
        occ.cut([(3, disk)], cut)
