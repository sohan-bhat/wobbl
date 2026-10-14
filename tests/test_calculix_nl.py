"""The nonlinear CalculiX deck and its readers, without a solver: steps that ramp and remove the loads, plastic
tables, contact pairs, springs and pretension sections, and the .sta and .dat files CalculiX writes."""
import numpy as np
import pytest

from wobbl.fe import NodeConstraint, StaticModel
from wobbl.solvers import calculix_nl as nl


@pytest.fixture(scope="module")
def bar(parts, tmp_path_factory):
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path_factory.mktemp("bar"))
    return build_mesh(prep, Sizing(h=8.0))


def _model(bar, steps, **kw):
    root = np.nonzero(np.abs(bar.X[:, 0]) < 1e-6)[0]
    tip = np.nonzero(np.abs(bar.X[:, 0] - 200) < 1e-6)[0]
    F = np.zeros((bar.n_nodes, 3))
    F[tip, 2] = -100.0 / len(tip)
    sm = StaticModel(bar, 200000.0, 0.3, 7.85e-9, [NodeConstraint("fix1", root)], forces=F)
    mat = nl.Material("steel", 200000.0, 0.3, hardening=[(250.0, 0.0), (400.0, 0.2)])
    return nl.NonlinearModel(sm, [mat], steps, **kw)


def test_steps_ramp_the_loads_up_and_back_down(bar, tmp_path):
    m = _model(bar, [nl.Step(1.5, nlgeom=True), nl.Step(0.0, nlgeom=True)])
    nl.write_nonlinear_inp(tmp_path / "a.inp", m)
    deck = (tmp_path / "a.inp").read_text()
    assert "*PLASTIC\n250,0\n400,0.2\n" in deck
    assert deck.count("*STEP, INC=300, NLGEOM\n") == 2
    first, second = deck.split("*END STEP\n")[:2]
    f1 = [float(ln.split(",")[2]) for ln in first.split("*CLOAD\n")[1].splitlines() if ln.count(",") == 2]
    f2 = [float(ln.split(",")[2]) for ln in second.split("*CLOAD\n")[1].splitlines() if ln.count(",") == 2]
    assert sum(f1) == pytest.approx(-150.0) and f2 and all(v == 0 for v in f2)
    # a step that saves only its end does it through a time point, and prints no totals
    assert "*NODE FILE, TIME POINTS=TEND\nU,RF\n" in first and "*NODE PRINT" not in first


def test_a_collapse_ramp_saves_every_increment_and_gives_up_early(bar, tmp_path):
    m = _model(bar, [nl.Step(3.0, may_stop=True, every_increment=True, smallest=5e-4)],
               monitor={"MON": np.array([5])}, fields=("U", "PEEQ"))
    nl.write_nonlinear_inp(tmp_path / "b.inp", m)
    deck = (tmp_path / "b.inp").read_text()
    assert "*CONTROLS, PARAMETERS=TIME INCREMENTATION\n4,8,9,10,10,4,,5\n" in deck
    assert "*NODE FILE\nU\n*EL FILE\nPEEQ\n" in deck
    assert "*NODE PRINT, NSET=NFIX1, TOTALS=ONLY\nRF\n" in deck and "*NODE PRINT, NSET=MON\nU\n" in deck


def test_contact_springs_and_pretension_are_written(bar, tmp_path):
    top = np.isclose(bar.X[bar.tris[:, :3], 2], 10).all(1)
    bottom = np.isclose(bar.X[bar.tris[:, :3], 2], 0).all(1)
    cp = nl.ContactPair("C1", top, bottom, 5e4, mu=0.2, stick=500.0, tension=0.01, clearance=0.0)
    cut = np.column_stack([bar.tri_elem[top][:3], bar.tri_face[top][:3]])
    pt = nl.Pretension("bolt1", cut, np.array([0, 0, 1.0]), 12000.0)
    m = _model(bar, [nl.Step(0.0, preload=True), nl.Step(1.0, lock=True)], contact=[cp],
               springs=[(np.array([3, 4]), 0.5)], pretension=[pt])
    info = nl.write_nonlinear_inp(tmp_path / "c.inp", m)
    deck = (tmp_path / "c.inp").read_text()
    assert "*CONTACT PAIR, INTERACTION=C1I, TYPE=SURFACE TO SURFACE\nC1S,C1M\n" in deck
    assert "*FRICTION\n0.2,500\n" in deck and "*CLEARANCE, MASTER=C1M, SLAVE=C1S, VALUE=0\n" in deck
    assert deck.count("TYPE=SPRING1") == 3 and "*SPRING, ELSET=ESPR1_3\n3\n0.5\n" in deck
    node = info["pretension"]["bolt1"] + 1
    assert f"*PRE-TENSION SECTION, SURFACE=PT1, NODE={node}\n0,0,1\n" in deck
    preload, lock = deck.split("*END STEP\n")[:2]
    assert f"{node},1,1.2000000000e+04\n" in preload
    assert f"*BOUNDARY, FIXED\n{node},1,1\n" in lock and f"{node},1,0.\n" in lock


def test_read_sta_counts_finished_increments_and_every_iteration(tmp_path):
    (tmp_path / "j.sta").write_text(
        "SUMMARY OF JOB INFORMATION\n  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME\n"
        "     1          1     1     3  0.100000E+00  0.100000E+00  0.100000E+00\n"
        "     1          2     1U   10  0.100000E+00  0.100000E+00  0.100000E+00\n"
        "     1          2     2     7  0.125000E+00  0.125000E+00  0.250000E-01\n"
        "     2          1     1     2  0.112500E+01  0.125000E+00  0.125000E+00\n")
    assert nl.read_sta(tmp_path / "j.sta") == (3, 22, 2)


def test_read_dat_reads_totals_and_nodal_rows(tmp_path):
    (tmp_path / "j.dat").write_text(
        "                        S T E P       1\n\n\n                                INCREMENT     1\n\n\n"
        " total force (fx,fy,fz) for set NFIX1 and time  0.2500000E+00\n\n"
        "        3.837763E-13 -1.865175E-14  1.500000E+03\n\n"
        " displacements (vx,vy,vz) for set MON and time  0.2500000E+00\n\n"
        "        12  1.000000E-01  0.000000E+00 -2.500000E-01\n"
        "        13  2.000000E-01  0.000000E+00 -3.500000E-01\n\n")
    d = nl.read_dat(tmp_path / "j.dat")
    t, rows = d[("total force", "NFIX1")][0]
    assert t == 0.25 and rows[0][2] == 1500.0
    t, rows = d[("displacements", "MON")][0]
    assert rows.shape == (2, 4) and rows[1, 3] == -0.35
