"""Flow studies through OpenFOAM, against exact answers and published data. Skipped where OpenFOAM is not
installed ('wobbl setup --only openfoam')."""
from __future__ import annotations

import pytest

from conftest import needs_foam

pytestmark = [pytest.mark.solver, needs_foam]


def test_laminar_pipe_slice_reproduces_64_over_re(tmp_path):
    from wobbl.flow.pipecfd import friction

    r = friction(500.0, 0.0, tmp_path)
    assert r["f"] == pytest.approx(64 / 500, rel=2e-3)
    assert {c.name: c.status for c in r["checks"]} == {"grid convergence": "pass", "iterative convergence": "pass",
                                                       "CFD against the chart": "pass"}
    assert not (tmp_path / "cfd").exists()            # case folders go once the numbers are read


def test_transitional_and_rough_pipes_are_not_checked_by_cfd(tmp_path):
    from wobbl.flow.pipecfd import friction

    assert "transitional" in friction(3000.0, 0.0, tmp_path)["skipped"]
    assert "rough" in friction(1e5, 1e-3, tmp_path)["skipped"]


def test_installed_openfoam_is_lean_and_passes_its_check():
    from wobbl.flow import install

    got = install.check(lambda m: None)
    assert got["works"], got
    info = install.installed()
    if info:      # an OpenFOAM of the user's own carries no marker
        assert info["kept_MB"] < 260 and info["full_MB"] > 3 * info["kept_MB"]


def test_heated_gap_between_fins_matches_stephan_and_shah(tmp_path):
    """Developing laminar flow and heat between two plates, against Stephan's heat transfer at Re 100 and Shah's
    apparent friction at Re 1000."""
    from wobbl.flow import correlations as C
    from wobbl.flow.heatsink import channel_run, shah_london_dp

    r = channel_run(100.0, 0.71, 3.55, (90, 27), tmp_path)
    assert r["nu"] == pytest.approx(C.nu_plates_developing(100.0, 0.71, 7.1, 2.0), rel=0.03)
    assert abs(r["q_in"] - r["q_out"]) < 1e-6 * r["q_in"]
    r = channel_run(1000.0, 0.71, 5.16, (90, 27), tmp_path)
    assert r["dp"] == pytest.approx(shah_london_dp(1000.0, 5.16), rel=0.05)
    assert not any(tmp_path.iterdir())


@pytest.mark.slow
def test_time_accurate_runs_reproduce_vortex_shedding_at_re_100(tmp_path):
    """The time accurate solver settings, against the cylinder's vortex street at Re 100: Williamson's (1996)
    Strouhal number 0.164, and the mean drag coefficient of 1.33 and lift amplitude of about 0.33 that Park, Kwon
    and Choi (1998) and Henderson (1995) computed. Measured here: St 0.161, Cd 1.333, lift amplitude 0.35."""
    from wobbl.flow import correlations as C
    from wobbl.flow.external import shedding

    r = shedding(100.0, tmp_path, "laminar", periods=10, st_guess=0.165, grid=(24, 60), settle_periods=15,
                 max_co=0.9)
    assert r["st"] == pytest.approx(C.strouhal_williamson(100.0), rel=0.03)
    assert r["cd"] == pytest.approx(1.33, rel=0.03)
    assert r["cl_amp"] == pytest.approx(0.33, rel=0.12)
    assert abs(r["st_halves"][0] - r["st_halves"][1]) < 0.01 * r["st"]       # the street had settled


@pytest.mark.slow
def test_iddes_sheds_at_the_cylinders_strouhal_number_at_re_3900(tmp_path):
    """Spalart Allmaras IDDES on a coarse 3D slab of the cylinder (18 thousand cells, 90 degrees of span), against
    the subcritical Strouhal number of about 0.21 (Williamson 1996 gives 0.215 at Re 3900). Measured here in about
    50 minutes: St 0.233 over eight cycles, the two halves of the run within 1%, and 99% of the near wake's energy
    resolved. The mean drag of 1.76 is far above the 0.9 to 1.1 of fine grids, which is why the check is on the
    frequency only and a coarse grid is reported with low confidence."""
    from wobbl.flow.external import shedding

    r = shedding(3900.0, tmp_path, "des", periods=8, st_guess=0.2, settle_periods=6)
    assert 0.18 < r["st"] < 0.25
    assert abs(r["st_halves"][0] - r["st_halves"][1]) < 0.03 * r["st"]
    assert r["resolved"]["resolved_fraction"] > 0.8
