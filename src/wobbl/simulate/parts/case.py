"""A sealed handheld device housing in polycarbonate, with a cradle for its battery.

A closed box with rounded vertical corners, 2.5 mm walls all round, and two ribs inside on the floor of the box
that hold the battery between them. Coordinates in mm, centered on x = 0, y = 0, the bottom face on z = 0.
"""

PARAMS = {
    "wall": 2.5,
    "length": 100.0,
    "width": 55.0,
    "height": 22.0,
    "corner": 5.0,        # outside radius of the vertical corners
    "rib": 2.0,           # battery cradle ribs: thickness
    "rib_height": 7.0,
    "cradle": 56.0,       # distance between the ribs
}


def build(occ, wall, length, width, height, corner, rib, rib_height, cradle):
    import gmsh

    def rounded_box(x0, y0, z0, lx, ly, lz, r):
        b = occ.addBox(x0, y0, z0, lx, ly, lz)
        occ.synchronize()
        eps = 1e-3
        edges = []
        for cx in (x0, x0 + lx):
            for cy in (y0, y0 + ly):
                edges += gmsh.model.getEntitiesInBoundingBox(cx - eps, cy - eps, z0 - eps, cx + eps, cy + eps,
                                                             z0 + lz + eps, 1)
        return occ.fillet([b], [e for _, e in edges], [r])

    outer = rounded_box(-length / 2, -width / 2, 0, length, width, height, corner)
    inner = rounded_box(-length / 2 + wall, -width / 2 + wall, wall, length - 2 * wall, width - 2 * wall,
                        height - 2 * wall, max(corner - wall, 0.5))
    shell, _ = occ.cut(outer, inner)
    ribs = []
    for sx in (-1, 1):
        x = sx * cradle / 2 - (rib if sx > 0 else 0)
        ribs.append((3, occ.addBox(x, -width / 2 + wall + 6, wall - 0.5, rib, width - 2 * wall - 12,
                                   rib_height + 0.5)))
    occ.fuse(shell, ribs)
