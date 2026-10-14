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
