"""A round aluminum tube strut, the kind that braces a frame or props a hatch open.

The tube stands along z from z = 0 to z = length, with both ends cut square. Coordinates in mm.
"""

PARAMS = {
    "length": 600.0,
    "od": 20.0,           # outside diameter
    "wall": 3.0,
}


def build(occ, length, od, wall):
    outer = occ.addCylinder(0, 0, 0, 0, 0, length, od / 2)
    inner = occ.addCylinder(0, 0, -1, 0, 0, length + 2, od / 2 - wall)
    occ.cut([(3, outer)], [(3, inner)])
