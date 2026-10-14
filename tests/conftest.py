"""Shared fixtures: small parts built with gmsh's OpenCASCADE kernel, and solver availability."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
# a test without a solver should skip or fail, never download one
os.environ.setdefault("WOBBL_AUTO_INSTALL", "0")


def _write(path: Path, build):
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("t")
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="session")
def parts(tmp_path_factory):
    d = tmp_path_factory.mktemp("parts")

    def plate_hole(occ):
        b = occ.addBox(0, 0, 0, 100, 40, 5)
        h = occ.addCylinder(50, 20, -1, 0, 0, 7, 4)
        occ.cut([(3, b)], [(3, h)])

    def cantilever(occ):
        occ.addBox(0, 0, 0, 200, 20, 10)

    def lbracket(occ):
        a = occ.addBox(0, 0, 0, 80, 6, 20)
        b = occ.addBox(0, 0, 0, 6, 70, 20)
        occ.fuse([(3, a)], [(3, b)])

    def mount(occ):
        base = occ.addBox(0, 0, 0, 60, 40, 5)
        holes = [occ.addCylinder(x, y, -1, 0, 0, 7, 2.25) for x in (10, 50) for y in (10, 30)]
        occ.cut([(3, base)], [(3, h) for h in holes])

    out = {}
    for name, fn in (("plate_hole", plate_hole), ("cantilever", cantilever), ("lbracket", lbracket),
                     ("mount", mount)):
        out[name] = str(_write(d / f"{name}.step", fn))
    return out


def _have(domain):
    from wobbl.solvers import base

    try:
        return base.for_domain(domain).locate() is not None
    except Exception:  # noqa: BLE001
        return False


needs_ccx = pytest.mark.skipif(not _have("structural"), reason="CalculiX not installed (run wobbl setup)")
needs_spice = pytest.mark.skipif(not _have("circuit"), reason="ngspice not installed (run wobbl setup)")
