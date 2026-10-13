"""Geometry for the validation cases, built with OpenCASCADE through gmsh and saved as STEP (mm)."""
from __future__ import annotations

import math
from pathlib import Path


def _session(path: Path, build):
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("v")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


def box(path, lx, ly, lz):
    return _session(Path(path), lambda o: o.addBox(0, 0, 0, lx, ly, lz))


def plate_with_hole(path, length, width, thick, d):
    def b(o):
        p = o.addBox(-length / 2, -width / 2, 0, length, width, thick)
        h = o.addCylinder(0, 0, -1, 0, 0, thick + 2, d / 2)
        o.cut([(3, p)], [(3, h)])
    return _session(Path(path), b)


def quarter_tube(path, ri, ro, length):
    def b(o):
        outer = o.addCylinder(0, 0, 0, 0, 0, length, ro, angle=math.pi / 2)
        inner = o.addCylinder(0, 0, -1, 0, 0, length + 2, ri, angle=math.pi / 2)
        o.cut([(3, outer)], [(3, inner)])
    return _session(Path(path), b)


def shaft(path, d, length):
    return _session(Path(path), lambda o: o.addCylinder(0, 0, 0, 0, 0, length, d / 2))


def _quarter_ellipse_ring(o, a_in, b_in, a_out, b_out, z0, dz):
    outer = o.addDisk(0, 0, z0, a_out, b_out)
    inner = o.addDisk(0, 0, z0, a_in, b_in)
    ring, _ = o.cut([(2, outer)], [(2, inner)])
    quad = o.addRectangle(0, 0, z0, a_out * 1.1, b_out * 1.1)
    q, _ = o.intersect(ring, [(2, quad)])
    ext = o.extrude(q, 0, 0, dz)
    return [e for e in ext if e[0] == 3]


def nafems_le1(path):
    """Elliptic membrane: inner ellipse 2000 x 1000 mm, outer 3250 x 2750 mm, 100 mm thick."""
    return _session(Path(path), lambda o: _quarter_ellipse_ring(o, 2000, 1000, 3250, 2750, 0, 100))


def nafems_le10(path):
    """Thick elliptic plate, 600 mm thick, with a 2 mm band at mid thickness so the outer edge line can be held."""
    def b(o):
        vols = []
        for z0, dz in ((-300, 299), (-1, 2), (1, 299)):
            vols += _quarter_ellipse_ring(o, 2000, 1000, 3250, 2750, z0, dz)
        o.fragment(vols[:1], vols[1:])
    return _session(Path(path), b)
