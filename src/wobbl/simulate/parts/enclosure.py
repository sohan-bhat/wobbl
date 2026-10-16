"""Top shell of a handheld electronics enclosure in polycarbonate.

An open bottom box: the rim sits on z = 0, the outside of the top is at
z = height, and four screw bosses run from the underside of the top down to the
rim. Coordinates in mm, centered on x = 0, y = 0.
"""

PARAMS = {
    "wall": 3.0,          # side walls and top
    "length": 90.0,
    "width": 60.0,
    "height": 28.0,
    "corner": 6.0,        # outside radius of the vertical corners
    "boss": 7.0,          # boss diameter
}


def build(occ, wall, length, width, height, corner, boss):
    def rounded_box(x0, y0, z0, lx, ly, lz, r):
        b = occ.addBox(x0, y0, z0, lx, ly, lz)
        if r <= 0:
            return [(3, b)]
        import gmsh
        occ.synchronize()
        eps = 1e-3
        edges = []
        for cx in (x0, x0 + lx):
            for cy in (y0, y0 + ly):
                edges += gmsh.model.getEntitiesInBoundingBox(cx - eps, cy - eps, z0 - eps, cx + eps, cy + eps,
                                                             z0 + lz + eps, 1)
        return occ.fillet([b], [e for _, e in edges], [r])

    outer = rounded_box(-length / 2, -width / 2, 0, length, width, height, corner)
    inner = rounded_box(-length / 2 + wall, -width / 2 + wall, -1, length - 2 * wall, width - 2 * wall,
                        height - wall + 1, max(corner - wall, 0.5))
    shell, _ = occ.cut(outer, inner)
    bosses = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (length / 2 - wall - boss / 2), sy * (width / 2 - wall - boss / 2)
            post = occ.addCylinder(x, y, 0, 0, 0, height - wall + 0.5, boss / 2)
            hole = occ.addCylinder(x, y, -1, 0, 0, height - wall, 1.25)
            p, _ = occ.cut([(3, post)], [(3, hole)])
            bosses += p
    occ.fuse(shell, bosses)
