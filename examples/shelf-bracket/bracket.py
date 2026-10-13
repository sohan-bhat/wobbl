"""A 3D printed L shelf bracket, screwed to a wall through two holes in its upright.

Coordinates in mm: the wall is the plane x = 0, z points up, the bracket is
`width` wide along y. The shelf rests on the top of the horizontal arm.
Run `wobbl run bracket.py --param fillet=6` to try a rounded inside corner.
"""

PARAMS = {
    "thickness": 6.0,      # both legs
    "width": 25.0,
    "height": 80.0,        # upright
    "depth": 90.0,         # arm, measured from the wall
    "fillet": 0.5,         # inside corner radius; 0.5 is what a printer leaves on a "sharp" corner
    "hole": 4.5,           # screw clearance holes in the upright
}


def build(occ, thickness, width, height, depth, fillet, hole):
    t = thickness
    arm = occ.addBox(0, 0, 0, depth, width, t)
    upright = occ.addBox(0, 0, 0, t, width, height)
    body, _ = occ.fuse([(3, arm)], [(3, upright)])
    occ.synchronize()
    if fillet > 0:
        import gmsh

        # the inside corner edge runs along y at x = t, z = t
        edges = gmsh.model.getEntitiesInBoundingBox(t - 1e-3, -1e-3, t - 1e-3, t + 1e-3, width + 1e-3, t + 1e-3, 1)
        body = occ.fillet([v for _, v in body], [e for _, e in edges], [fillet])
    holes = [occ.addCylinder(-1, width / 2, z, t + 2, 0, 0, hole / 2) for z in (25.0, height - 15.0)]
    occ.cut(body, [(3, h) for h in holes])
