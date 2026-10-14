"""Carbon fiber arm of a heavy lift drone: a round tube, clamped in the hub at x = 0, motor at the far end.

The tube runs along x from the hub (x = 0) to the motor mount (x = length). Coordinates in mm.
"""

PARAMS = {
    "diameter": 25.0,     # outside diameter
    "wall": 1.0,          # the laminate, eight 0.125 mm plies
    "length": 400.0,
}


def build(occ, diameter, wall, length):
    outer = occ.addCylinder(0, 0, 0, length, 0, 0, diameter / 2)
    inner = occ.addCylinder(-1, 0, 0, length + 2, 0, 0, diameter / 2 - wall)
    occ.cut([(3, outer)], [(3, inner)])
