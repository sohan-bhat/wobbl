"""The crack growth command line and agent tool, checked without solving."""
import asyncio

import pytest
from click.testing import CliRunner

from wobbl.cli.crack import study_from_options
from wobbl.cli.main import cli
from wobbl.structural.spec import SpecError


def test_study_options_are_checked_before_any_solve():
    args = ("part.step", "7075-t6", ["4 kN +x @ x=max"], ["x=min"], "through 1 mm at hottest")
    st = study_from_options(*args, r_ratio=0.1, target_life=1e5, detectable="0.05 in")
    assert st.detectable == pytest.approx(1.27) and st.target_life == 1e5
    with pytest.raises(KeyError, match="no crack growth data"):
        study_from_options("part.step", "pla", *args[2:])
    assert study_from_options("part.step", "pla", *args[2:], growth="C=1e-7,n=3").growth
    with pytest.raises(SpecError):
        study_from_options("part.step", "7075-t6", [], ["x=min"], "through 1 mm")
    with pytest.raises(SpecError):
        study_from_options(*args, r_ratio=1.2)


def test_cli_and_mcp_offer_crack_growth():
    r = CliRunner().invoke(cli, ["crack", "--help"])
    assert r.exit_code == 0 and "--crack" in r.output and "--target-life" in r.output
    from wobbl.mcp.server import build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert "grow_crack" in names
