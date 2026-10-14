"""A machined 7075-T6 bracket: a bolted base and an arm ending in a lug with a pin hole.

The base bolts down through two holes; the arm runs along +x to the lug, whose pin pulls up and out at 30
degrees to the arm. Coordinates in mm, the plate lies in x-y and is `thickness` thick in z.
"""

PARAMS = {
    "thickness": 5.0,
    "arm_width": 28.0,
    "lug_radius": 14.0,
    "pin_hole": 8.0,
    "fillet": 6.0,
}


def build(occ, thickness, arm_width, lug_radius, pin_hole, fillet):
    base = occ.addBox(0, -25, 0, 30, 50, thickness)
    arm = occ.addBox(0, -arm_width / 2, 0, 90, arm_width, thickness)
    lug = occ.addCylinder(90, 0, 0, 0, 0, thickness, lug_radius)
    body, _ = occ.fuse([(3, base)], [(3, arm), (3, lug)])
    occ.synchronize()
    # round the two inside corners where the arm leaves the base
    edges = []
    for y in (-arm_width / 2, arm_width / 2):
        edges += [t for _, t in occ.getEntitiesInBoundingBox(29.9, y - 0.1, -0.1, 30.1, y + 0.1, thickness + 0.1, 1)]
    vol = body[0][1]
    if edges and fillet > 0:
        body = occ.fillet([vol], edges, [fillet])
        vol = body[0][1]
    holes = [occ.addCylinder(12, y, -1, 0, 0, thickness + 2, 3.3) for y in (-15, 15)]
    holes.append(occ.addCylinder(90, 0, -1, 0, 0, thickness + 2, pin_hole / 2))
    occ.cut([(3, vol)], [(3, h) for h in holes])
