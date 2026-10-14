"""KiCad schematics and plain descriptions become circuits with the right parts on the right nets. Nothing is
solved here."""
from pathlib import Path

import pytest

from wobbl.circuits.describe import build
from wobbl.circuits.inputs import kind_of
from wobbl.circuits.kicad import parse_netlist_export, parse_schematic, transform

DATA = Path(__file__).parent / "data" / "circuits"


def _sch(name, params=None):
    f = DATA / f"{name}.kicad_sch"
    return parse_schematic(f.read_text(), f, params or {})


def test_symbol_transform_follows_eeschema():
    # a resistor's pin 1 sits at (0, 3.81) in its library, above the body
    pin = (0, 3.81)

    def at(angle, mirror=None):
        t = transform(angle, mirror)
        return round(t[0] * pin[0] + t[1] * pin[1], 3), round(t[2] * pin[0] + t[3] * pin[1], 3)

    assert at(0) == (0, -3.81)            # sheet y points down, so above is negative
    assert at(90) == (-3.81, 0)           # rotated counterclockwise, the top pin goes left
    assert at(180) == (0, 3.81) and at(270) == (3.81, 0)
    assert at(0, "x") == (0, 3.81) and at(90, "y") == (3.81, 0)


def test_battery_led_board_traces_wires_labels_and_power_symbols():
    c = _sch("led_battery")
    bt, r1, d1 = (c.element(x) for x in ("BT1", "R1", "D1"))
    assert bt.nodes[0] == r1.nodes[0] and bt.nodes[1] == "0"      # joined by a wire
    assert r1.nodes[1] == d1.nodes[0] == "led_a" and d1.nodes[1] == "0"
    assert d1.rest == ["LED_RED_0805"] and d1.meta["footprint"].startswith("LED_SMD")


def test_power_tree_board_maps_regulator_mcu_and_mosfet_by_pin_name():
    c = _sch("ldo_mcu_board", {"J2.load": "2A"})
    u1, u2, q1 = c.element("U1"), c.element("U2"), c.element("Q1")
    assert u1.pins == ["in", "gnd", "out"] and u1.nodes == ["p12v", "0", "p5v"]
    assert dict(zip(u2.pins, u2.nodes))["VCC"] == "p5v" and dict(zip(u2.pins, u2.nodes))["PB0"] == "led"
    assert q1.letter == "m" and q1.nodes == ["stripm", "gate", "0"]
    assert c.element("J2 load").nodes == ["p12v", "stripm"] and c.names["p12v"] == "+12V"


def test_dual_op_amp_units_share_their_power_pins():
    c = _sch("opamp_dual")
    a, b = c.element("U1A"), c.element("U1B")
    assert a.nodes == ["sense", "fb", "p12v", "0", "amp_out"] and b.nodes == ["ref", "ref_buf", "p12v", "0", "ref_buf"]
    assert any("+12V" in x for x in c.assumptions)                   # nothing drives the rail on the sheet


def test_charger_maps_its_pins_and_battery():
    c = _sch("charger_usb")
    u1 = c.element("U1")
    assert u1.pins == ["vin", "gnd", "bat", "prog", "stat"] and u1.nodes == ["p5v", "0", "vbat", "prog", "stat"]
    assert c.element("BT1").meta["as"]["kind"] == "battery"


def test_netlist_export_reads_components_and_nets():
    text = """(export (version "E")
      (components
        (comp (ref "R1") (value "1k") (footprint "Resistor_SMD:R_0603_1608Metric") (libsource (lib "Device") (part "R")))
        (comp (ref "D1") (value "Blue") (libsource (lib "Device") (part "LED"))))
      (libparts (libpart (lib "Device") (part "LED") (pins (pin (num "1") (name "K")) (pin (num "2") (name "A")))))
      (nets
        (net (code "1") (name "+5V") (node (ref "R1") (pin "1")))
        (net (code "2") (name "Net-(D1-A)") (node (ref "R1") (pin "2")) (node (ref "D1") (pin "2")))
        (net (code "3") (name "GND") (node (ref "D1") (pin "1")))))"""
    c = parse_netlist_export(text)
    assert c.element("R1").nodes == ["p5v", "netm_d1ma"] and c.element("D1").nodes == ["netm_d1ma", "0"]
    assert c.element("D1").rest == ["LED_BLUE"] and any(e.meta.get("ref") == "+5V supply" for e in c.elements)


@pytest.mark.parametrize("text,kind", [("V1 a 0 5\nR1 a 0 1k\n", "netlist"), ("(kicad_sch (version 1))", "kicad"),
                                       ("(export (version D))", "kicad-netlist"), ("a red LED on 5 V", "description")])
def test_input_kinds(text, kind):
    assert kind_of(text) == kind


def test_descriptions_build_netlists_and_say_their_guesses():
    said = []
    c = build("three red LEDs in parallel from a 9 V battery", {}, lambda t, o="": said.append((t, o)))
    assert sum(e.letter == "d" for e in c.elements) == 3 and sum(e.letter == "r" for e in c.elements) == 3
    assert c.element("BT1").meta["battery"].upper() == "9V"
    assert said and said[0][0].startswith("LED resistor not given") and said[0][1] == "--set resistor=470"
    c = build("a 555 blinking a red LED at 2 Hz from 9 V", {})
    assert any(e.rest == ["NE555"] for e in c.elements) and any(e.letter == "d" for e in c.elements)
    c = build("an AMS1117-3.3 from USB powering an ESP32 with an LED on GPIO2", {})
    u = next(e for e in c.elements if e.rest == ["ESP32"])
    assert u.nodes[0] == "vreg" and len(u.nodes) == 3


def test_words_that_are_not_a_circuit_say_how_to_give_one():
    with pytest.raises(ValueError, match="could not build") as e:
        build("hello there", {})
    assert "netlist" in e.value.hint and "KiCad" in e.value.hint
