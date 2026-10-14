"""Bench drawings and schematics of circuit runs: drawn from the run's own numbers, small and well formed."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wobbl.circuits import bench, draw

COLOR = dict(zip(["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "gray", "white"], bench.BAND),
             gold=bench.GOLD, silver=bench.SILVER)


def well_formed(svg: str, limit: int = 40_000):
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") and root.get("role") == "img" and root.get("aria-label")
    assert len(svg.encode()) < limit
    assert "<image" not in svg and "base64" not in svg and "Inter" not in svg and "IBM Plex Mono" in svg
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    return root


@pytest.mark.parametrize("ohms,names", [
    (47, "yellow violet black gold"), (10e3, "brown black orange gold"), (4.7, "yellow violet gold gold"),
    (0.47, "yellow violet silver gold"), (150, "brown green brown gold"), (123, "brown red orange black brown"),
])
def test_resistor_color_code(ohms, names):
    assert bench.bands(ohms) == [COLOR[n] for n in names.split()]


def test_values_too_small_for_bands_and_si_prefixes():
    assert bench.bands(0) is None and bench.bands(0.001) is None
    assert bench.eng(1e-5, "F") == "10 µF" and bench.eng(5e5, "Hz") == "500 kHz" and bench.eng(0.1, "s") == "100 ms"
    assert bench.eng(47, "Ω") == "47 Ω" and bench.eng(2.2e-12, "F") == "2.2 pF"


LED = {"supply": 5.0, "resistor": 47.0, "led": "blue", "vf": 3.2, "imax": 0.03, "rating": 0.25}


def test_led_bench_follows_the_run():
    svg, label = draw.led_setup(LED, {"i_led_A": 0.036862, "v_led_V": 3.2675}, {}, "fails")
    root = well_formed(svg)
    assert root.get("data-live") and "36.9 mA" in svg and "0.037" in svg and "5.00" in svg and "FAILS" in svg
    assert all(COLOR[n] in svg for n in ("yellow", "violet")) and bench.LED_COLOR["blue"] in svg
    assert "stroke-dasharray=\"3 3\"" in svg and "data-drift" in svg  # it overheats
    assert "47 ohm" in label and "36.9 mA" in label
    safe, _ = draw.led_setup(dict(LED, resistor=150.0, led="red", vf=2.0), {"i_led_A": 0.0198, "v_led_V": 2.03},
                             {}, "ok")
    assert "OK" in safe and "stroke-dasharray=\"3 3\"" not in safe and bench.LED_COLOR["red"] in safe
    assert COLOR["green"] in safe  # 150 ohm is brown green brown
    fast, slow = (float(re.search(r'data-flow="([\d.]+)"', x).group(1)) for x in (svg, safe))
    assert fast / slow == pytest.approx(0.036862 / 0.0198, rel=0.01)  # the dots move with the current
    series = [float(v) for v in re.search(r'data-series="([^"]+)" data-fmt="1 mA"', svg).group(1).split()]
    assert series[0] == 0 and series[-1] == pytest.approx(36.862, rel=1e-3)


def test_rc_bench_traces_the_charge_and_picks_the_capacitor():
    t = np.linspace(0, 0.5, 400)
    data = {"time": t, "v(out)": 5 * (1 - np.exp(-t / 0.1))}
    svg, _ = draw.rc_setup({"r": 1e4, "c": 1e-5, "v": 5.0, "tau": 0.1}, {"v_at_tau": 3.16}, data, "ok")
    well_formed(svg)
    assert "4.97 V" in svg and "τ = RC = 100 ms" in svg and 'pathLength="1"' in svg and "data-k" in svg
    small, _ = draw.rc_setup({"r": 1e4, "c": 1e-7, "v": 5.0, "tau": 1e-3}, {"v_at_tau": 3.16},
                             {"time": t / 100, "v(out)": data["v(out)"]}, "ok")
    assert ">104<" in small  # a ceramic disc marked with its code


def _buck_data(f=5e5, cycles=300):
    t = np.linspace(0, cycles / f, cycles * 200 + 1)
    phase = (t * f) % 1
    il = 2 + 0.3 * np.where(phase < 0.437, phase / 0.437 * 2 - 1, 1 - (phase - 0.437) / 0.563 * 2)
    return {"time": t, "v(out)": 5.07 + 0.00428 * np.sin(2 * np.pi * f * t), "i(l1)": il}


def test_buck_bench_switches_and_scopes_the_ripple():
    p = {"vin": 12.0, "vout": 5.0, "current": 2.0, "frequency": 5e5, "duty": 0.437, "max_ripple": "0.05"}
    svg, label = draw.buck_setup(p, {"ripple_V": 0.00856, "vout_avg": 5.07}, _buck_data(), "ok")
    well_formed(svg)
    assert "8.56 mV" in svg and "50 mV limit" in svg and "12.00" in svg and "2.03" in svg
    assert 'data-pwm="0.437 1"' in svg and "0.437 1 inverted" in svg and 'data-sweep="3"' in svg
    assert "data-wave" in svg and "500 kHz" in label


def _mosfet(theta):
    t = np.linspace(0, 40 / 2e4, 4001)
    on = (t * 2e4) % 1 < 0.5
    p = {"current": 15.0, "rds_on": 0.008, "vbus": 12.0, "frequency": 2e4, "duty": 0.5, "theta_ja": theta,
         "ambient": "50"}
    res = {"tj_C": 110.4, "p_total_W": 1.51, "p_cond_W": 1.366, "p_sw_W": 0.144}
    return draw.mosfet_setup(p, res, {"time": t, "i(vs)": np.where(on, 15.0, 0), "v(d)": np.where(on, 0.12, 12)},
                             "ok")


def test_mosfet_bench_heats_up_to_the_runs_junction_temperature():
    svg, label = _mosfet("40")
    well_formed(svg)
    assert "110 °C" in svg and "150 °C limit" in svg and ">7.5" in svg and "20.00" in svg and "data-heat" in svg
    series = [float(v) for v in re.search(r'data-series="([^"]+)" data-fmt="0 °C"', svg).group(1).split()]
    assert series[0] == 50 and series[-1] == pytest.approx(110.4) and series == sorted(series)
    bare, bare_label = _mosfet("62")
    assert "40 C/W heatsink" in label and "no heatsink" in bare_label and svg.count("data-heat") > \
        bare.count("data-heat")
