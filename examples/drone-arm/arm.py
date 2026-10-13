"""A quadcopter arm printed in PA12 (MJF), with lightening windows and a motor pad.

The root block bolts into the frame through two vertical holes; the arm runs
along +x; the motor sits on the round pad at the tip with its axis at
x = length. Coordinates in mm, z up, the arm's underside at z = 0.
"""

PARAMS = {
    "length": 150.0,      # root face to motor axis
    "width": 16.0,
    "root_height": 20.0,
    "tip_height": 12.0,
    "window": 3.0,        # material left around each lightening window
    "pad": 30.0,          # motor pad diameter
}


def build(occ, length, width, root_height, tip_height, window, pad):
    # tapered arm as an extruded side profile (in the x-z plane), extruded across y
    pts = [(-20, 0), (length, 0), (length, tip_height), (0, root_height), (-20, root_height)]
    tags = [occ.addPoint(x, -width / 2, z) for x, z in pts]
    lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    arm = [e for e in occ.extrude([(2, face)], 0, width, 0) if e[0] == 3]
    disk = occ.addCylinder(length, 0, 0, 0, 0, tip_height + 3, pad / 2)
    body, _ = occ.fuse(arm, [(3, disk)])
    cut = []
    # three slotted windows through the width, shrinking with the taper
    for i, x0 in enumerate((22.0, 58.0, 92.0)):
        h_here = root_height + (tip_height - root_height) * (x0 + 14) / length
        slot = occ.addBox(x0, -width, window, 26 - 4 * i, 2 * width, max(h_here - 2 * window, 2))
        cut.append((3, slot))
    for xb in (-14.0, -6.0):
        cut.append((3, occ.addCylinder(xb, 0, -1, 0, 0, root_height + 2, 1.6)))
    for a in range(4):
        import math
        cut.append((3, occ.addCylinder(length + 9.5 * math.cos(a * math.pi / 2 + math.pi / 4),
                                       9.5 * math.sin(a * math.pi / 2 + math.pi / 4), -1, 0, 0,
                                       tip_height + 5, 1.6)))
    cut.append((3, occ.addCylinder(length, 0, -1, 0, 0, tip_height + 5, 4.0)))
    occ.cut(body, cut)
