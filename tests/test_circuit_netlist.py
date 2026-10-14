"""Reading circuits and binding their parts: SPICE text, the parts library and instrumenting, without a solver."""
import pytest

from wobbl.circuits.netlist import join_parasitics, parse_spice, safe_node
from wobbl.circuits.parts import e_series, lookup

NET = """LED driver
V1 vcc 0 DC 9
R1 vcc a 330 ; 0805
D1 a 0 1N4148
Q1 c b 0 2N3904
M1 d g 0 IRLZ44N
XU1 inp inn vcc 0 out LM358
+ params: x=1
.model foo D(IS=1n)
.subckt LM358 1 2 3 4 5
R1 1 2 1meg
.ends
.tran 1u 1m
.control
run
.endc
.end
"""


def test_spice_lines_become_parts_with_pins():
    c = parse_spice(NET)
    assert c.title == "LED driver" and [e.name for e in c.elements] == ["V1", "R1", "D1", "Q1", "M1", "XU1"]
    r1, d1, q1, m1, xu1 = c.elements[1:]
    assert r1.value == 330 and r1.meta["note"] == "0805" and d1.pins == ["a", "k"] and d1.model == "1N4148"
    assert q1.nodes == ["c", "b", "0"] and m1.nodes == ["d", "g", "0"]     # a library MOSFET has three terminals
    assert xu1.ref == "U1" and xu1.model == "LM358" and xu1.pins == ["1", "2", "3", "4", "5"]
    assert "foo" in c.models and "lm358" in c.subckts and c.analyses == [".tran 1u 1m"]


def test_a_first_line_that_is_a_part_is_not_taken_as_the_title():
    c = parse_spice("V1 in 0 5\nR1 in 0 1k\n")
    assert [e.name for e in c.elements] == ["V1", "R1"] and c.title == ""


def test_instrumenting_puts_an_ammeter_on_every_pin():
    c = parse_spice(NET)
    net = c.spice(instrument=True)
    assert "vwb_r1_0 vcc wb_r1_0 0" in net and "R1 wb_r1_0 a 330" in net
    assert "vwb_q1_2 0 wb_q1_2 0" in net             # three pin parts get one on each pin
    assert ".save" in net and "i(vwb_xu1_4)" in net and net.rstrip().endswith(".end")
    assert len([p for p in c.probes() if p[0].name == "XU1"]) == 5


def test_board_parasitics_join_the_netlist():
    c = parse_spice("* buck\nV1 sw 0 5\nL1 sw out 10u\nC1 out 0 10u\n")
    said = join_parasitics(c, {"traces": [{"net": "sw", "to": "L1.1", "r": 0.01, "l": 5e-9}],
                               "nets": {"sw": {"c": 3e-10}}})
    assert len(said) == 2 and c.element("L1").nodes[0] == "sw_pe0"
    assert c.element("Lpar0").nodes == ["sw_pt0", "sw_pe0"] and c.element("Cpar0").kind == "parasitic"


@pytest.mark.parametrize("name,key,kind", [
    ("1N4007", "1N4007", "diode"), ("1N4148W", "1N4148W", "diode"), ("BZX79C5V1", "BZX79C5V1", "zener"),
    ("IRLZ44NPBF", "IRLZ44N", "nmos"), ("AO3401A", "AO3401A", "pmos"), ("LM7805CT", "LM7805CT", "regulator"),
    ("AMS1117-3.3", "AMS1117-3.3", "regulator"), ("LM358DR", "LM358", "opamp"), ("NE555P", "NE555", "timer"),
    ("TP4056", "TP4056", "charger"), ("ESP32-WROOM-32E", "ESP32-WROOM-32E", "mcu"), ("TIP120", "TIP120", "npn"),
])
def test_library_finds_parts_by_number(name, key, kind):
    p = lookup(name)
    assert p is not None and p.key == key and p.kind == kind and p.source


def test_unknown_names_and_net_names():
    assert lookup("vcc") is None and lookup("XYZ123") is None
    assert safe_node("GND") == "0" and safe_node("+5V") == "p5v" and safe_node("3V3") == "n3v3"
    assert e_series(330.5) == 360 and e_series(1234) == 1300 and e_series(98, up=False) == 91


def test_a_makers_model_saved_in_the_parts_folder_is_used(tmp_path, monkeypatch):
    from wobbl.circuits import bind as b

    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "vendor.lib").write_text("* maker model\n.model D1N4148X D(IS=2.5n N=1.75 RS=0.6)\n")
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path))
    monkeypatch.setattr(b, "_USER", None)
    c = parse_spice("* t\nV1 a 0 5\nR1 a k 1k\nD1 k 0 D1N4148X\n")
    b.bind(c)
    assert "d1n4148x" in c.models and "~/.wobbl/parts" in c.element("D1").part["source"]
    monkeypatch.setattr(b, "_USER", None)
