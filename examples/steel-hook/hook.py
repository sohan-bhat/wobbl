"""A forged steel J hook that bolts to a beam through the eye in its shank.

The shank stands along z with its eye at the top; below it the hook bends through 200 degrees around a center at
x = radius, so a sling hangs in the bottom of the bend at (radius, 0, -(radius - depth / 2)). The section is a
rectangle, depth across the bend and width along y. Coordinates in mm.
"""
import math

PARAMS = {
    "radius": 22.0,       # centerline radius of the bend
    "depth": 12.0,        # section depth across the bend, the dimension that resists bending
    "width": 10.0,        # section width
    "shank": 70.0,        # shank length above the bend
    "eye": 8.5,           # eye hole for an M8 bolt
    "sweep": 200.0,       # degrees of bend
}


def build(occ, radius, depth, width, shank, eye, sweep):
    body = occ.addBox(-depth / 2, -width / 2, 0, depth, width, shank)
    # the bend: the shank's bottom face swept around an axis along y through (radius, 0, 0), downward first
    base = occ.addRectangle(-depth / 2, -width / 2, 0, depth, width)
    bend = occ.revolve([(2, base)], radius, 0, 0, 0, 1, 0, -math.radians(sweep))
    vols = [e for e in bend if e[0] == 3]
    hook, _ = occ.fuse([(3, body)], vols)
    hole = occ.addCylinder(0, -width, shank - 1.5 * eye, 0, 2 * width, 0, eye / 2)
    occ.cut(hook, [(3, hole)])
