"""Flow correlations, fluid properties and the checks every flow answer runs, against published values. Nothing here
needs OpenFOAM."""
from __future__ import annotations

import math

import pytest

from wobbl.flow import correlations as C
from wobbl.flow import fluids as F
from wobbl.flow import verify as V


def test_friction_factors_match_the_moody_chart():
    assert C.friction_factor(1000)[0] == pytest.approx(0.064)
    # smooth pipe at Re 1e5 is 0.0180 on the Moody chart, and 0.0222 at a relative roughness of 0.001
    assert C.colebrook(1e5) == pytest.approx(0.01799, rel=2e-3)
    assert C.colebrook(1e5, 1e-3) == pytest.approx(0.0222, rel=5e-3)
    assert C.haaland(1e5) == pytest.approx(C.colebrook(1e5), rel=0.02)
    # Churchill's single equation reproduces both ends
    assert C.churchill(500) == pytest.approx(64 / 500, rel=1e-3)
    assert C.churchill(1e6) == pytest.approx(C.colebrook(1e6), rel=0.02)
    assert C.friction_factor(3000)[1].startswith("Churchill")
    assert C.rect_fRe(1.0) == pytest.approx(56.91, rel=1e-3) and C.rect_fRe(0.0) == 96.0


def test_drag_curves():
    # Clift, Grace and Weber's curve: 1.09 at Re 100, 0.47 at Re 1000, and the drag crisis past 3e5
    assert C.sphere_cd(100) == pytest.approx(1.087, rel=1e-3)
    assert C.sphere_cd(1000) == pytest.approx(0.471, rel=5e-3)
    assert C.sphere_cd(6e5) < 0.15 < C.sphere_cd(2e5)
    for re in (20, 260, 1500, 1.2e4, 4.4e4):     # the pieces meet within a few percent
        assert C.sphere_cd(re * 0.999) == pytest.approx(C.sphere_cd(re * 1.001), rel=0.04)
    assert 1.0 < C.cylinder_cd(1e4) < 1.25
    assert C.strouhal_williamson(100) == pytest.approx(0.1643, rel=1e-3)


def test_heat_correlations():
    # a long channel between plates tends to the fully developed 7.54
    assert C.nu_plates_developing(100, 0.7, 10.0, 0.01) == pytest.approx(7.55, rel=1e-3)
    assert C.nu_plates_developing(1000, 0.7, 0.05, 0.01) > 9
    assert C.nu_gnielinski(1e5, 0.7) == pytest.approx(188, rel=0.05)
    # Teertstra's model reaches the fully developed parallel plate value, Nu_b = Re*_b Pr / 2, for short gaps
    nu = C.nu_teertstra(0.05, 0.002, 0.1, 1.5e-5, 0.7)
    assert nu == pytest.approx(0.05 * 0.002 / 1.5e-5 * 0.002 / 0.1 * 0.7 / 2, rel=0.02)


def test_fluid_properties_match_handbooks():
    w = F.get_fluid("water", 20)
    assert w.rho == pytest.approx(998.2, abs=0.2)
    assert w.mu == pytest.approx(1.002e-3, rel=0.01)
    assert w.k == pytest.approx(0.598, rel=0.01)
    a = F.get_fluid("air", 20)
    assert a.rho == pytest.approx(1.204, rel=2e-3)
    assert a.mu == pytest.approx(1.81e-5, rel=0.01)
    assert a.Pr == pytest.approx(0.71, rel=0.02)
    assert F.get_fluid("seawater").nu == pytest.approx(1.1883e-6)
    c = F.get_fluid("custom:rho=870,mu=0.1")
    assert c.nu == pytest.approx(0.1 / 870)
    with pytest.raises(ValueError):
        F.get_fluid("honey")


def test_oils_and_glycol_follow_their_tables_and_grades():
    oil = F.get_fluid("engine oil", 300 - 273.15)          # a tabulated point comes back exactly
    assert (oil.rho, oil.mu, oil.k) == pytest.approx((884.1, 0.486, 0.145))
    mid = F.get_fluid("engine-oil", 305 - 273.15).mu        # Andrade's law sits between the points, log wise
    assert 0.253 < mid < 0.486 and mid == pytest.approx((0.253 * 0.486) ** 0.5, rel=0.03)
    assert F.get_fluid("glycol", 20).nu == pytest.approx(19.1e-6, rel=0.02)
    for grade, hot in F.VI100_AT_100C.items():              # ISO grades hold their definition at 40 C
        assert F.get_fluid(f"ISO VG {grade}", 40).nu == pytest.approx(grade * 1e-6, rel=1e-9)
        assert F.get_fluid(f"iso-vg-{grade}", 100).nu == pytest.approx(hot * 1e-6, rel=1e-9)
    with pytest.raises(ValueError):
        F.get_fluid("glycerin", 80)


def test_flow_quantities_read_in_si():
    assert F.qty("20 L/min", "flowrate") == pytest.approx(20e-3 / 60)
    assert F.qty("40 CFM", "flowrate") == pytest.approx(0.018878, rel=1e-4)
    assert F.qty("15 kn", "velocity") == pytest.approx(7.7167, rel=1e-4)
    assert F.qty("0.5 inH2O", "pressure") == pytest.approx(124.54, rel=1e-3)
    assert F.qty("6000 rpm", "rotation") == pytest.approx(100)
    assert F.qty("25 mm", "length") == pytest.approx(0.025)
    assert F.qty("1.2 kg", "mass") == pytest.approx(1.2)
    assert F.qty("15 W", "power") == pytest.approx(15)
    with pytest.raises(ValueError):
        F.qty("20 furlongs", "velocity")


def test_gci_reproduces_celik_2008():
    # Celik et al. (2008), Table 1: p 1.53, extrapolated 6.1685, fine grid index 2.17%
    g = V.gci([6.063, 5.972, 5.863], [1.0, 1.5, 2.0])
    assert g.kind == "monotone"
    assert g.order == pytest.approx(1.53, abs=0.01)
    assert g.extrapolated == pytest.approx(6.1685, abs=1e-3)
    assert g.gci_fine == pytest.approx(0.0217, abs=2e-4)
    assert V.gci_check(g, "drag", "N", tol=0.03).status == "pass"


def test_gci_flags_grids_that_do_not_converge():
    assert V.gci([1.0, 1.02, 0.99], [1, 1.5, 2.25]).kind == "oscillatory"
    g = V.gci([1.0, 0.9, 0.85], [1, 2, 4])
    assert g.kind == "diverging" and V.gci_check(g, "x", "").status == "fail"
    assert V.gci([2.0, 2.0, 2.0], [1, 2, 4]).gci_fine == 0.0
    assert V.gci_check(None, "x", "", values_only=[1.0]).status == "warn"


def test_residuals_mass_and_wall_checks():
    log = "\n".join(f"smoothSolver:  Solving for Ux, Initial residual = {r}, Final residual = 1e-9, No Iterations 2\n"
                    f"GAMG:  Solving for p, Initial residual = {10 * r}, Final residual = 1e-9, No Iterations 9"
                    for r in (1.0, 1e-3, 1e-6))
    assert V.residuals(log)["p"] == pytest.approx((10.0, 1e-5))
    assert V.residual_check(log).status == "pass"
    assert V.residual_check(log.replace("1e-06", "0.01")).status == "fail"
    assert V.residual_check("").status == "fail"
    assert V.mass_check(1.0, 0.999).status == "pass" and V.mass_check(1.0, 0.95).status == "fail"
    assert V.yplus_check({"wall": (0.1, 1.2, 0.6)}, "resolved").status == "pass"
    assert V.yplus_check({"wall": (5, 40, 20)}, "resolved").status == "fail"
    assert V.yplus_check({"wall": (30, 90, 60)}, "wall functions").status == "pass"
    assert V.model_reason("laminar", "pipe") is None
    level, why = V.model_reason("rans", "separated")
    assert level == "medium" and "25%" in why


def test_first_cell_height_scales_with_y_plus():
    h1 = C.first_cell(1.0, 10.0, 1.5e-5, 1.0)
    assert C.first_cell(30.0, 10.0, 1.5e-5, 1.0) == pytest.approx(30 * h1)
    assert 1e-6 < h1 < 1e-4
    assert math.isfinite(C.darcy_dp(0.02, 10, 0.025, 1000, 1.0, 2.0))


def test_tensor_fields_read_as_one_row_per_cell(tmp_path):
    from wobbl.flow import foam
    f = tmp_path / "UPrime2Mean"
    f.write_text("internalField   nonuniform List<symmTensor> 2\n(\n(1 2 3 4 5 6)\n(7 8 9 10 11 12)\n)\n;\n")
    assert foam.read_field(f).shape == (2, 6)
    g = tmp_path / "p"
    g.write_text("internalField   nonuniform List<scalar> 3\n(\n1\n2\n3\n)\n;\n")
    assert foam.read_field(g).shape == (3,)
