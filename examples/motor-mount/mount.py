"""A ribbed motor mount for a NEMA 17 stepper driving a timing belt, printed in PETG.

Coordinates in mm: the base sits on z = 0 and bolts down through four holes; the
upright is the plane y = 0..thickness; the motor bolts to its front (+y) face and
its shaft passes back through the pilot hole to a pulley behind the upright.
Two triangular ribs brace the upright on the motor side.
"""

PARAMS = {
    "thickness": 5.0,     # base and upright
    "rib": 4.0,           # rib thickness, 0 for no ribs
    "fillet": 3.0,        # base to upright inside corner
    "height": 55.0,       # upright
}


def build(occ, thickness, rib, fillet, height):
    import gmsh

    t = thickness
    base = occ.addBox(-35, 0, 0, 70, 50, t)
    upright = occ.addBox(-25, 0, 0, 50, t, height)
    body, _ = occ.fuse([(3, base)], [(3, upright)])
    occ.synchronize()
    if fillet > 0:
        edges = gmsh.model.getEntitiesInBoundingBox(-25.1, t - 1e-3, t - 1e-3, 25.1, t + 1e-3, t + 1e-3, 1)
        body = occ.fillet([v for _, v in body], [e for _, e in edges], [fillet])
    ribs = []
    for x0 in ((-25.0, 25.0 - rib) if rib > 0 else ()):
        pts = [occ.addPoint(x0, t, t), occ.addPoint(x0, 30, t), occ.addPoint(x0, t, 40)]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 3]) for i in range(3)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        ribs += [e for e in occ.extrude([(2, surf)], rib, 0, 0) if e[0] == 3]
    if ribs:
        body, _ = occ.fuse(body, ribs)
    cz = 31.0
    cutters = [occ.addCylinder(0, -1, cz, 0, t + 2, 0, 11.25)]
    cutters += [occ.addCylinder(sx * 15.5, -1, cz + sz * 15.5, 0, t + 2, 0, 1.7) for sx in (-1, 1) for sz in (-1, 1)]
    cutters += [occ.addCylinder(sx * 29, y, -1, 0, 0, t + 2, 2.75) for sx in (-1, 1) for y in (12, 42)]
    occ.cut(body, [(3, c) for c in cutters])
