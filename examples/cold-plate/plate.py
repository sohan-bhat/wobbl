"""An aluminum cold plate: a block with a drilled U shaped channel whose two ports sit side by side on one end.

Two bores run from the x = 0 end most of the way along the plate and a cross bore joins them near the far end,
the way a plate is made with a drill and a plug. The bottom face, z = 0, carries the electronics. Coordinates in mm.
"""

PARAMS = {
    "length": 120.0,
    "width": 70.0,
    "thickness": 16.0,
    "bore": 8.0,          # channel diameter
    "pitch": 40.0,        # distance between the two bores
}


def build(occ, length, width, thickness, bore, pitch):
    block = occ.addBox(0, 0, 0, length, width, thickness)
    r, z = bore / 2, thickness / 2
    y1, y2 = (width - pitch) / 2, (width + pitch) / 2
    reach = length - 2 * bore
    a = occ.addCylinder(-1, y1, z, reach + 1, 0, 0, r)
    b = occ.addCylinder(-1, y2, z, reach + 1, 0, 0, r)
    c = occ.addCylinder(reach - r, y1, z, 0, pitch, 0, r)
    occ.cut([(3, block)], [(3, a), (3, b), (3, c)])
