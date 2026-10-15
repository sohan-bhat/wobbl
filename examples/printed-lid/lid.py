"""A shallow box lid printed face down: a flat top with a lip around it.

The top lies on the bed at z = 0 and the lip rises from it. Coordinates in mm.
"""

PARAMS = {
    "length": 220.0,
    "width": 60.0,
    "top": 2.5,           # top thickness, on the bed
    "wall": 2.0,          # lip thickness
    "lip": 12.0,          # lip height above the top
}


def build(occ, length, width, top, wall, lip):
    outer = occ.addBox(0, 0, 0, length, width, top + lip)
    inner = occ.addBox(wall, wall, top, length - 2 * wall, width - 2 * wall, lip + 1)
    occ.cut([(3, outer)], [(3, inner)])
