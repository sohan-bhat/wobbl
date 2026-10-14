"""Casting pieces: the multi body mesh, Chvorinov's rule, feeding paths and the Niyama porosity curve."""
import math

import numpy as np
import pytest

from wobbl.process import casting as C
from wobbl.process.data import get_alloy, get_mold
from wobbl.process.regions import parse_chill, parse_riser, split_interfaces


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    from wobbl.geometry.cad import prepare
    from wobbl.process.regions import build
    from wobbl.validation import geometry as G

    d = tmp_path_factory.mktemp("slab")
    prep = prepare(G.box(d / "slab.step", 20, 20, 20), d / "geo")
    return build(prep, [], [], (0, 0, 40), 2.5, 6.0)


def test_riser_and_chill_text():
    r = parse_riser("riser d=40 h=60 at 50,0,30 axis=x")
    assert (r.diameter, r.height, r.at, r.axis) == (40.0, 60.0, (50.0, 0.0, 30.0), "x")
    assert r.modulus == pytest.approx(40 * 60 / (4 * 60 + 40))
    c = parse_chill("copper 30x30x15 at 50,0,-10")
    assert (c.material, c.size, c.center) == ("copper", (30.0, 30.0, 15.0), (50.0, 0.0, -10.0))
    with pytest.raises(ValueError, match="d=40"):
        parse_riser("a big riser")


def test_mold_block_and_split_interfaces(slab):
    assert slab.names == ["casting", "mold"]
    assert slab.volumes()[0] == pytest.approx(8000, rel=1e-6)
    assert slab.interfaces()[(0, 1)] == pytest.approx(800, rel=1e-6)
    metal = slab.region == 0
    X2, tets2, (a, b) = split_interfaces(slab.X, slab.tets, metal)
    assert len(a) == len(b) and len(X2) > len(slab.X)
    assert np.allclose(X2[a], X2[b])
    assert not set(np.unique(tets2[metal]).tolist()) & set(np.unique(tets2[~metal]).tolist())


def test_chvorinov_constant_by_hand():
    al, sand = get_alloy("pure-al"), get_mold("sand")
    dh = al.latent + al.specific_heat * (al.liquidus - 0.5 * (al.liquidus + al.solidus))
    tm = 0.5 * (al.liquidus + al.solidus)
    B = math.pi / 4 * (al.density * dh / (tm - 25.0)) ** 2 / (sand.conductivity * sand.density * sand.specific_heat)
    assert C.chvorinov_constant(al, sand, al.liquidus, 25.0) == pytest.approx(B * 1e-6, rel=1e-12)


def test_slab_freezes_in_chvorinov_time(slab):
    al, sand = get_alloy("pure-al"), get_mold("sand")
    st = C.CastingStudy("slab", film=0.0, contact=1e5)
    sv = C.solidify(st, slab, al, sand, al.liquidus, 1500.0)
    t = float(np.nanmax(sv.result.events["solidus"][sv.casting_nodes]))
    assert t == pytest.approx(C.chvorinov_constant(al, sand, al.liquidus, 25.0) * 10.0 ** 2, rel=0.04)
    assert sv.result.energy["error"] < 1e-6


def test_feeding_finds_a_region_cut_off_from_the_riser(slab):
    """A made up freezing field: a riser at the top freezes last, but a spot at the bottom center freezes after
    everything around it, so liquid there is cut off."""
    al, sand = get_alloy("pure-al"), get_mold("sand")
    st = C.CastingStudy("slab", film=0.0)
    sv = C.solidify(st, slab, al, sand, al.liquidus, 1.0)
    X = sv.rm.X
    z = X[:, 2]
    t = 100.0 + 2.0 * z                                     # later toward the top
    spot = np.linalg.norm(X - [10, 10, 3], axis=1) < 4.0
    t[spot] = 200.0 - np.linalg.norm(X[spot] - [10, 10, 3], axis=1)
    sv.result.events["solidus"][:] = np.nan
    sv.result.events["solidus"][sv.casting_nodes] = t[sv.casting_nodes]
    sv.riser_nodes = sv.casting_nodes[z[sv.casting_nodes] > 19.9]
    fd = C.feeding(sv)
    assert len(fd["pockets"]) == 1
    assert np.linalg.norm(np.asarray(fd["pockets"][0]["hot_xyz"]) - [10, 10, 3]) < 2.5


def test_niyama_porosity_follows_carlson_and_beckermann():
    a = get_alloy("a356")
    p = a.niyama
    # Table IV at 1 C/s: Ny = 1.26 (C s)^0.5/mm is Ny* = 211, the 0.01% threshold, and Table III gives 0.012% there
    ny, gp = C.niyama(a, np.array([1.26, 0.01, 1e-4]), np.array([1.0, 1.0, 1.0]))
    nys = p["c_lambda"] * 1260 * math.sqrt(p["pressure_drop"] / (p["viscosity"] * p["beta"] * p["freezing_range"]))
    assert nys == pytest.approx(211, rel=0.01)
    assert ny[0] == pytest.approx(1.26)
    assert gp[0] == pytest.approx(7.771 * 211 ** -1.206, rel=0.02)
    assert gp[1] > 1.0
    assert gp[2] == pytest.approx(100 * a.shrinkage)          # capped at the total shrinkage
    # the fitted pieces meet where they change over
    per = 1.26 / nys                                   # C/mm of gradient for each unit of Ny* at 1 C/s
    for edge in (1.43, 18.0):
        lo, hi = C.niyama(a, np.array([edge * (1 - 1e-6), edge * (1 + 1e-6)]) * per, np.ones(2))[1]
        assert lo == pytest.approx(hi, rel=2e-3)


def test_a_small_casting_end_to_end(tmp_path):
    """A block given no riser: Wobbl places one over its hot spot, the block then freezes fed with the riser last,
    and the outputs are all there."""

    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "block.step", 40, 30, 12)
    st = C.CastingStudy(str(part), auto_risers=True, quality="fast", mesh_size=3.0)
    o = C.run_casting(st, tmp_path / "out")
    assert o.domain == "casting" and o.verdict in ("ok", "marginal")
    assert o.values["risers_placed"] and len(o.values["risers"]) == 1
    assert not o.values["pockets"]
    assert o.values["t_riser_s"] > o.values["t_casting_s"]
    names = {c.name: c.status for c in o.checks}
    assert names["energy balance"] == "pass" and names["feeding"] == "pass"
    assert names["mesh convergence"] == "fail"            # fast quality solves one mesh and says so
    from wobbl.viz import load
    head, _ = load(tmp_path / "out" / "scene.wobbl")
    assert head["domain"] == "casting" and head["timeline"]["frames"] > 10
    assert not (tmp_path / "out" / "work").exists()
