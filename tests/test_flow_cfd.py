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
