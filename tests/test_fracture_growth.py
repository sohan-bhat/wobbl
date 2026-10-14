"""Crack growth laws and load spectra."""
import math

import numpy as np
import pytest

from wobbl.fracture.growth import GrowthLaw, closure_f, law_for, parse_constants, parse_spectrum
from wobbl.materials import CRACK_GROWTH, LIBRARY, crack_growth


def test_every_growth_record_names_a_library_material_and_a_source():
    for key, g in CRACK_GROWTH.items():
        assert key in LIBRARY and g.source and g.C > 0 and 2 <= g.n <= 4.5 and g.dk_th > 0
    assert crack_growth("7075-t6*") is CRACK_GROWTH["7075-t6"]
    with pytest.raises(KeyError, match="--growth"):
        crack_growth("pla")


def test_closure_function():
    f = closure_f(np.array([-1.0, 0.0, 0.5, 0.9]))
    assert 0.2 < f[1] < 0.45            # about a third of the range is closed at R = 0
    assert f[0] < f[1] < f[2] < f[3]
    assert f[3] == pytest.approx(0.9, abs=0.02)   # at high R the crack never closes


def test_nasgro_matches_paris_at_r0_away_from_threshold_and_toughness():
    paris = GrowthLaw("paris", 2.71e-8, 3.7)
    nasgro = law_for("7075-t6")
    plain = GrowthLaw("nasgro", nasgro.C, 3.7)   # same C, no threshold or toughness
    assert plain.rate(8.0, 0.0) == pytest.approx(paris.rate(8.0, 0.0), rel=1e-9)
    assert nasgro.rate(1.0, 0.0) == 0.0                     # below the 1.8 MPa sqrt(m) threshold
    assert nasgro.rate(30.0, 0.0) == math.inf               # past K_Ic = 29
    assert nasgro.rate(5.0, 0.5) > nasgro.rate(5.0, 0.0)    # higher R, faster
    assert nasgro.threshold(0.5) < nasgro.threshold(0.0) == pytest.approx(1.8)
    assert law_for("7075-t6", "paris").rate(5.0, 0.7) == pytest.approx(paris.rate(5.0, 0.0))


def test_custom_constants_and_spectra(tmp_path):
    assert parse_constants("C=2.7e-8, n=3.7 dkth=1.8 kic=29") == {"C": 2.7e-8, "n": 3.7, "dkth": 1.8, "kic": 29.0}
    with pytest.raises(ValueError):
        parse_constants("n=3")
    law = law_for("", custom="C=1e-8,n=3,kic=40")
    assert law.kc == 40 and law.dk1 == 0
    sp = parse_spectrum(r_ratio=0.1)
    assert sp.constant and sp.r[0] == pytest.approx(0.1)
    f = tmp_path / "block.csv"
    f.write_text("max,min\n1.0,0.1\n0.5,0.05,9  # nine smaller cycles\n")
    sp = parse_spectrum(str(f))
    assert sp.cycles == 10 and sp.peak == 1.0 and not sp.constant
    with pytest.raises(ValueError):
        parse_spectrum(str(tmp_path / "none.csv"))



def test_load_blocks_and_counted_cycles():
    from wobbl.fracture.growth import parse_blocks

    sp = parse_spectrum("100% x 3650 R=0; 70% x 1e8 R=0.714")
    assert sp.cycles == pytest.approx(3650 + 1e8) and len(sp.smax) == 2
    assert sp.smin[1] == pytest.approx(0.7 * 0.714) and sp.expanded() is None
    sp = parse_blocks(["1,0.1 x 2", "0.5 x 3"], r_ratio=0.2)
    assert sp.smin.tolist() == pytest.approx([0.1, 0.1]) and sp.cycles == 5
    assert len(sp.expanded().smax) == 5
    with pytest.raises(ValueError):
        parse_blocks(["lots x 3"])


def test_metals_without_data_are_estimated_from_their_stiffness():
    from wobbl.materials import resolve_material

    cast = law_for(resolve_material("custom:E=70000,nu=0.33,yield=165,process=cast"), "paris")
    assert cast.estimated and cast.C == pytest.approx(1.65e-8 * 27, rel=1e-6)   # IIW's aluminium curve
    steel = law_for(resolve_material("custom:E=205000,nu=0.3,yield=350"), "paris")
    assert steel.C == pytest.approx(1.65e-8) and steel.dk1 == 0.0
    assert not law_for("7075-t6").estimated
    with pytest.raises(KeyError, match="only metals"):
        law_for(resolve_material("pla"))
