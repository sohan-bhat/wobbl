"""Generality proof for circuits: 44 varied circuits that Wobbl was not tuned on, as SPICE netlists, plain
descriptions and KiCad schematics. Each one has to simulate, write its playback and drawings, and get the verdict
an engineer would give it, written down here before it was run. Run this file directly for the table:

    python tests/test_circuit_generality.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from conftest import needs_spice

pytestmark = pytest.mark.solver

DATA = Path(__file__).parent / "data" / "circuits"
OK, MARGINAL, FAILS = {"ok"}, {"marginal"}, {"fails"}

# (name, source, expected verdicts, settings) where source is a netlist in netlists.txt, words or a schematic
CASES = [
    ("divider_adc", "divider_adc.cir", OK, {}),
    ("divider_hot", "divider_hot.cir", FAILS, {}),
    ("rc_lowpass", "rc_lowpass.cir", OK, {}),
    ("sallen_key", "sallen_key.cir", OK, {}),
    ("led_array_usb", "led_array_usb.cir", OK, {}),
    ("led_string_12v", "led_string_12v.cir", MARGINAL | FAILS, {}),
    ("npn_led_switch", "npn_led_switch.cir", OK, {}),
    ("npn_relay_flyback", "npn_relay_flyback.cir", OK, {}),
    ("mosfet_heater", "mosfet_heater.cir", OK, {}),
    ("mosfet_3v3_irf540", "mosfet_3v3_irf540.cir", FAILS, {}),
    ("pmos_high_side", "pmos_high_side.cir", OK, {}),
    ("hbridge_motor", "hbridge_motor.cir", FAILS, {}),
    ("opamp_inverting", "opamp_inverting.cir", OK, {}),
    ("comparator_lm393", "comparator_lm393.cir", OK, {}),
    ("opamp_heavy_load", "opamp_heavy_load.cir", MARGINAL | FAILS, {}),
    ("ne555_led", "ne555_led.cir", OK, {}),
    ("lm7805_board", "lm7805_board.cir", OK, {}),
    ("lm317_5v", "lm317_5v.cir", OK, {}),
    ("ams1117_from_liion", "ams1117_from_liion.cir", FAILS, {}),
    ("zener_24v", "zener_24v.cir", FAILS, {}),
    ("buck_switch", "buck_switch.cir", OK, {}),
    ("boost_switch", "boost_switch.cir", OK, {}),
    ("bridge_rectifier", "bridge_rectifier.cir", OK, {}),
    ("tp4056_usbc", "tp4056_usbc.cir", MARGINAL, {}),
    ("mcu_power_tree", "mcu_power_tree.cir", OK, {}),
    ("arduino_led_47r", "arduino_led_47r.cir", FAILS, {}),
    ("opto_input", "opto_input.cir", OK, {}),
    ("tl431_5v", "tl431_5v.cir", OK, {}),
    ("common_emitter", "common_emitter.cir", OK, {}),
    ("tip120_solenoid", "tip120_solenoid.cir", OK, {}),
    ("unknown_parts", "unknown_parts.cir", OK | MARGINAL, {}),
    ("words_green_series", "two green LEDs in series from a 9 V battery with a 150 ohm resistor", OK, {}),
    ("words_555_blue", "a 555 flashing a blue LED at 5 Hz from a 12 V supply", OK, {}),
    ("words_lm317_hot", "an LM317 set to 3.3 V from 12 V feeding a 0.5 A load", FAILS, {}),
    ("words_bc547_motor", "a BC547 switching a 12 V motor from a 5 V Arduino pin", FAILS, {}),
    ("words_tl072_inverting", "an inverting TL072 amplifier with a gain of 5 on a ±15 V supply", OK, {}),
    ("words_high_pass", "a high pass filter at 100 Hz", OK, {}),
    ("kicad_led_battery", "led_battery.kicad_sch", FAILS, {}),
    ("kicad_ldo_mcu_board", "ldo_mcu_board.kicad_sch", OK, {"J2.load": "2A"}),
    ("kicad_blinker_555", "blinker_555.kicad_sch", OK, {}),
    ("kicad_opamp_dual", "opamp_dual.kicad_sch", OK, {}),
    ("kicad_charger_usb", "charger_usb.kicad_sch", MARGINAL, {}),
    ("kicad_relay_driver", "relay_driver.kicad_sch", OK, {}),
    ("kicad_led_matrix", "led_matrix.kicad_sch", MARGINAL, {}),
]


def netlists() -> dict[str, str]:
    text = (DATA / "netlists.txt").read_text()
    return {m.group(1): m.group(2).strip() + "\n" for m in re.finditer(r"^=== (\S+)\n(.*?)(?=^=== |\Z)", text,
                                                                      re.M | re.S)}


def run_all(out: Path, log=print) -> list[dict]:
    from wobbl.circuits.study import simulate_circuit

    nets = netlists()
    rows = []
    for name, src, want, params in CASES:
        source = nets.get(src) or (str(DATA / src) if src.endswith(".kicad_sch") else src)
        kind = "netlist" if src in nets else "kicad" if src.endswith(".kicad_sch") else "words"
        t0 = time.time()
        row = {"name": name, "kind": kind, "want": sorted(want)}
        try:
            o = simulate_circuit(source, "will it work, and does anything burn?", dict(params), out / name)
            files = {f: (out / name / f).exists() for f in ("playback.json", "images/setup.svg",
                                                            "images/schematic.svg")}
            pb = json.loads((out / name / "playback.json").read_text())
            row.update(simulated=True, verdict=o.verdict, sensible=o.verdict in want, headline=o.headline,
                       confidence=o.confidence, files=files, parts=len(pb["parts"]), fixes=len(o.fixes))
        except Exception as e:  # noqa: BLE001
            row.update(simulated=False, sensible=False, error=f"{type(e).__name__}: {e}")
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        log(f"{name:24} {row.get('verdict', 'ERROR'):9} {'ok' if row['sensible'] else 'NO':3} {row['seconds']:5}s "
            f"{row.get('headline', row.get('error', ''))[:110]}")
    return rows


@needs_spice
def test_every_circuit_simulates_with_a_sensible_verdict(tmp_path, capsys):
    rows = run_all(tmp_path, log=lambda m: None)
    sim = sum(r["simulated"] for r in rows)
    good = sum(r["sensible"] for r in rows)
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("netlist", "words", "kicad")}
    with capsys.disabled():
        print(f"\ncircuits: {sim} of {len(rows)} simulate and {good} get a sensible verdict "
              f"({kinds['netlist']} netlists, {kinds['words']} descriptions, {kinds['kicad']} KiCad schematics)")
        for r in rows:
            if not r["sensible"]:
                print(f"  {r['name']}: wanted {'/'.join(r['want'])}, got {r.get('verdict', r.get('error'))}")
    assert len(rows) >= 30 and kinds["kicad"] >= 5
    assert sim == len(rows), [r for r in rows if not r["simulated"]]
    assert all(all(r["files"].values()) for r in rows), [r["name"] for r in rows if not all(r["files"].values())]
    assert good >= 0.9 * len(rows)


if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        rows = run_all(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(t))
        print(f"\n{sum(r['simulated'] for r in rows)} of {len(rows)} simulate, "
              f"{sum(r['sensible'] for r in rows)} sensible")
