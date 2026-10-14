"""Bench drawings and schematics of circuit runs: drawn from the run's own numbers, small and well formed."""
import xml.etree.ElementTree as ET

import pytest

from wobbl.circuits import bench

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
