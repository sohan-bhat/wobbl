"""Six parts built with gmsh OCC that nothing in crack growth was tuned for, each with its loads and supports and
the box (x, y, z ranges in mm) where a real stress hot spot of that part lies."""
from __future__ import annotations


def _fillet(o, vol, box, r):
    o.synchronize()
    edges = [t for _, t in o.getEntitiesInBoundingBox(*box, 1)]
    return o.fillet([vol], edges, [r])[0][1] if edges else vol


def plate_hole(o):
    b = o.addBox(0, 0, 0, 100, 40, 5)
    o.cut([(3, b)], [(3, o.addCylinder(50, 20, -1, 0, 0, 7, 5))])


def l_bracket(o):
    foot = o.addBox(0, 0, 0, 60, 40, 8)
    wall = o.addBox(0, 0, 0, 8, 40, 60)
    (body, _) = o.fuse([(3, foot)], [(3, wall)])
    _fillet(o, body[0][1], (7.9, -0.1, 7.9, 8.1, 40.1, 8.1), 4)


def shaft_shoulder(o):
    big = o.addCylinder(0, 0, 0, 40, 0, 0, 16)
    small = o.addCylinder(40, 0, 0, 60, 0, 0, 10)
    (body, _) = o.fuse([(3, big)], [(3, small)])
    _fillet(o, body[0][1], (39.9, -16.1, -16.1, 40.1, 16.1, 16.1), 2)


def lug(o):
    plate = o.addBox(0, -15, 0, 70, 30, 6)
    ear = o.addCylinder(70, 0, 0, 0, 0, 6, 15)
    (body, _) = o.fuse([(3, plate)], [(3, ear)])
    o.cut(body, [(3, o.addCylinder(70, 0, -1, 0, 0, 8, 7))])


def i_beam(o):
    f1 = o.addBox(0, -20, 0, 160, 40, 4)
    f2 = o.addBox(0, -20, 46, 160, 40, 4)
    w = o.addBox(0, -2, 4, 160, 4, 42)
    (body, _) = o.fuse([(3, f1)], [(3, f2), (3, w)])
    o.synchronize()
    edges = []
    for z in (4, 46):
        for y in (-2, 2):
            edges += [t for _, t in o.getEntitiesInBoundingBox(-0.1, y - 0.1, z - 0.1, 160.1, y + 0.1, z + 0.1, 1)]
    o.fillet([body[0][1]], edges, [3])


def tube(o):
    a = o.addCylinder(0, 0, 0, 120, 0, 0, 14)
    o.cut([(3, a)], [(3, o.addCylinder(-1, 0, 0, 122, 0, 0, 10))])


# name: (builder, load, fix, hot spot box, material)
PARTS = {
    "plate with hole": (plate_hole, "6 kN 1,0,0 @ x=max", "x=min", ((40, 60), (0, 40), (0, 5))),
    "L bracket fillet": (l_bracket, "800 N 1,0,0 @ z=max", "z=min", ((-1, 20), (-1, 41), (-1, 20))),
    "shaft shoulder": (shaft_shoulder, "1.2 kN 0,1,0 @ x=max", "x=min", ((30, 50), (-17, 17), (-17, 17))),
    "lug": (lug, "4 kN 1,0,0 bearing @ hole d=14", "x=min", ((40, 86), (-16, 16), (0, 6))),
    "I beam flange": (i_beam, "1.5 kN 0,0,-1 @ x=max", "x=min", ((-1, 60), (-21, 21), (-1, 51))),
    "tube": (tube, "1.2 kN 0,1,0 @ x=max", "x=min", ((-1, 40), (-15, 15), (-15, 15))),
}
