"""A sand cast belt pulley: a rim, a thin web and a thick hub with a bore.

The pulley lies flat with its axis along z and the web's underside at z = 0. Coordinates in mm.
"""

PARAMS = {
    "rim_od": 120.0,
    "rim_wall": 8.0,      # radial thickness of the rim
    "rim_width": 24.0,
    "web": 6.0,           # web thickness
    "hub_od": 40.0,
    "hub_length": 36.0,
    "bore": 14.0,
}


def build(occ, rim_od, rim_wall, rim_width, web, hub_od, hub_length, bore):
    z0 = -(rim_width - web) / 2
    rim = occ.addCylinder(0, 0, z0, 0, 0, rim_width, rim_od / 2)
    occ.cut([(3, rim)], [(3, occ.addCylinder(0, 0, z0 - 1, 0, 0, rim_width + 2, rim_od / 2 - rim_wall))])
    disk = occ.addCylinder(0, 0, 0, 0, 0, web, rim_od / 2 - rim_wall / 2)
    hz = -(hub_length - web) / 2
    hub = occ.addCylinder(0, 0, hz, 0, 0, hub_length, hub_od / 2)
    body, _ = occ.fuse([(3, rim)], [(3, disk), (3, hub)])
    occ.cut(body, [(3, occ.addCylinder(0, 0, hz - 1, 0, 0, hub_length + 2, bore / 2))])
