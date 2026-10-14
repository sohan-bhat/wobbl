"""Stress life fatigue against Shigley's worked numbers: the S-N line, Marin factors, mean stress and Miner's rule."""
import math

import pytest

from wobbl.life import fatigue as F
from wobbl.materials import FATIGUE, fatigue_data, get_material, resolve_material

KPSI = F.KPSI


def test_shigley_example_6_2():
    """1050 HR steel, Sut 90 kpsi: S'e = 45 kpsi, and with f = 0.86 read off Fig. 6-18 the line gives 64.6 kpsi at
    1e4 cycles and about 77 500 cycles at 55 kpsi fully reversed."""
    su = 90 * KPSI
    book = F.Curve(su, 49.5 * KPSI, 0.86 * su, 0.5 * su, 1e6, True)
    assert book.b == pytest.approx(-0.0785, abs=1e-4)
    assert book.strength(1e4) / KPSI == pytest.approx(64.6, abs=0.1)
    assert book.life(55 * KPSI) == pytest.approx(77500, rel=0.005)
    # Wobbl's fit of Fig. 6-18 gives f = 0.864 at 90 kpsi, which moves the life 2%
    assert F.shigley_f(su) == pytest.approx(0.864, abs=0.001)
    assert F.shigley_f(200 * KPSI) == pytest.approx(0.776, abs=0.002) and F.shigley_f(400.0) == 0.9


def test_marin_factors():
    ka, _ = F.surface_factor("machined", 520.0, True)
    assert ka == pytest.approx(0.860, abs=0.001)                     # Shigley Example 6-3
    assert F.surface_factor("ground", 520.0, True)[0] == pytest.approx(1.58 * 520 ** -0.085)
    assert F.surface_factor("machined", 50.0, False)[0] == 1.0       # plastics: the data are for parts as made
    assert F.size_factor(32.0, "bending", True)[0] == pytest.approx(0.856, abs=0.001)
    assert F.size_factor(32.0, "axial", True)[0] == 1.0
    assert F.size_factor(100.0, "bending", True)[0] == pytest.approx(1.51 * 100 ** -0.157)
    assert F.reliability_factor(0.99)[0] == pytest.approx(0.814, abs=0.001)    # Shigley Table 6-5
    assert F.reliability_factor(0.5)[0] == 1.0
    with pytest.raises(ValueError):
        F.surface_factor("sanded", 500.0, True)


def test_built_curve_uses_the_material_data():
    mat = get_material("6061-t6")
    c = F.build_curve(mat, FATIGUE["6061-t6"], "machined", 10.0, "bending", 0.9)
    k = c.factors["surface ka"][0] * c.factors["size kb"][0] * c.factors["reliability ke"][0]
    assert c.se == pytest.approx(96.5 * k) and c.ne == 5e8 and not c.knee
    assert c.s1000 == pytest.approx(0.9 * 310)
    # aluminum has no endurance limit: lower stress always means more, but finite, cycles
    assert math.isfinite(c.life(0.5 * c.se)) and c.life(0.5 * c.se) > c.life(c.se) == pytest.approx(5e8)
    steel = F.build_curve(get_material("4140"), FATIGUE["4140"])
    assert steel.knee and steel.life(0.99 * steel.se) == math.inf


def test_mean_stress_models():
    c = F.Curve(600.0, 400.0, 540.0, 200.0, 1e6, True)
    assert F.safety_factor(100, 100, 200, c, "goodman") == pytest.approx(1 / (0.5 + 100 / 600))
    assert F.safety_factor(100, 100, 200, c, "soderberg") == pytest.approx(1 / (0.5 + 0.25))
    n = F.safety_factor(100, 100, 200, c, "gerber")
    assert n * 100 / 200 + (n * 100 / 600) ** 2 == pytest.approx(1.0)
    assert F.safety_factor(100, 100, 200, c, "gerber") > F.safety_factor(100, 100, 200, c, "goodman")
    assert F.safety_factor(100, -50, 200, c) == pytest.approx(2.0)        # no credit for a compressive mean
    assert F.equivalent_amplitude(100, 300, c, "goodman") == pytest.approx(200.0)
    assert F.equivalent_amplitude(100, 700, c, "goodman") == math.inf
    assert F.first_cycle_yield(150, 100, 400) == pytest.approx(1.6)
    assert F.cycle_stresses(200.0, 0.0) == (100.0, 100.0) and F.cycle_stresses(200.0, -1.0) == (200.0, 0.0)


def test_miners_rule_and_the_factor_that_reaches_it():
    c = F.Curve(600.0, 400.0, 540.0, 200.0, 1e6, True)
    blocks = [F.Block(400.0, 0.0, 1e3), F.Block(300.0, 0.0, 2e4), F.Block(150.0, 0.0, 1e9)]
    D, lives = F.damage(blocks, c)
    assert lives[2] == math.inf
    assert D == pytest.approx(1e3 / c.life(400) + 2e4 / c.life(300))
    s = F.factor_to_failure(blocks, c, repeats=3)
    assert 3 * F.damage(blocks, c, scale=s)[0] == pytest.approx(1.0, rel=1e-6)
    # ten cycles of 1 MPa: the line still allows hundreds of cycles just under Sut, so the part breaks at Sut
    assert F.factor_to_failure([F.Block(1.0, 0.0, 10.0)], c) == pytest.approx(c.su, rel=1e-6)


def test_blocks_from_text_and_random_bands():
    b = F.parse_block("60% x 1e6", 50.0, 50.0)
    assert (b.amplitude, b.mean, b.cycles) == (30.0, 30.0, 1e6)
    r = F.parse_block("1.0 x 2e4 R=-1", 50.0, 50.0)
    assert (r.amplitude, r.mean) == (100.0, 0.0)
    with pytest.raises(ValueError):
        F.parse_block("lots", 1.0, 1.0)
    bands = F.steinberg_blocks(10.0, 100.0, 60.0)
    assert [x.amplitude for x in bands] == [10.0, 20.0, 30.0]
    assert sum(x.cycles for x in bands) == pytest.approx(0.9973 * 6000, rel=1e-4)


def test_notch_sensitivity():
    q, _ = F.notch_sensitivity(FATIGUE["4140"], 690.0, 1.0)
    assert q == pytest.approx(1 / (1 + 0.0254 * (2079 / 690) ** 1.8 / 1.0))
    qa, _ = F.notch_sensitivity(FATIGUE["6061-t6"], 310.0, 2.0)
    assert qa == pytest.approx(1 / (1 + 0.635 / 2.0))
    assert F.notch_sensitivity(FATIGUE["pla"], 50.0, 2.0)[0] == 1.0
    assert F.notch_sensitivity(FATIGUE["4140"], 690.0, None)[0] == 1.0


def test_fatigue_data_for_custom_and_modified_materials():
    steel = fatigue_data(resolve_material("custom:E=200000,nu=0.29,yield=340,uts=620"))
    assert steel.knee and steel.strength == pytest.approx(310) and steel.estimated
    assert fatigue_data(resolve_material("pla:yield=40")).strength == pytest.approx(12.5 * 40 / 50)
    assert fatigue_data(get_material("copper")) is None
    assert all(f.source for f in FATIGUE.values())


def test_rainflow_matches_the_astm_e1049_example():
    """ASTM E1049-85 Fig. 6 and Table 4: the history -2, 1, -3, 5, -1, 3, -4, 4, -2 counts to ranges 3 (half a
    cycle), 4 (one and a half), 6 (half), 8 (one) and 9 (half)."""
    from collections import Counter

    from wobbl.life.fatigue import rainflow

    c = Counter()
    for rng, _, n in rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2]):
        c[rng] += n
    assert dict(c) == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def test_history_blocks_keep_every_cycle_and_the_worst_range(tmp_path):
    import numpy as np

    from wobbl.life.fatigue import history_blocks, read_history, reversals

    x = 1.0 + np.sin(np.linspace(0, 20 * np.pi, 4001)) + 0.3 * np.sin(np.linspace(0, 140 * np.pi, 4001))
    blocks = history_blocks(x, 50.0)
    from wobbl.life.fatigue import rainflow
    cyc = rainflow(x)
    assert sum(b.cycles for b in blocks) == pytest.approx(sum(n for r, _, n in cyc if r > 0))
    assert max(b.amplitude for b in blocks) == pytest.approx(50.0 * max(r for r, _, _ in cyc) / 2)
    assert len(reversals([0, 1, 1, 2, 1, 1, 0])) == 3
    f = tmp_path / "loads.csv"
    f.write_text("time,load\n0,0\n0.1,1.2\n0.2,-0.4\n0.3,0.9\n")
    assert list(read_history(str(f))) == [0, 1.2, -0.4, 0.9]
    assert list(read_history("0, 1, -0.5 0.8")) == [0, 1, -0.5, 0.8]
    with pytest.raises(ValueError):
        read_history("nothing here")
