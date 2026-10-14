"""Reading real board files (KiCad 5 to 8 and Gerber X2), nets as graphs, and planes under a trace."""
from __future__ import annotations

import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from wobbl.fields.pcb import (
    default_stackup,
    net_graph,
    parse_sexpr,
    read_board,
    read_gerbers,
    read_kicad_pcb,
    reference_planes,
)
from wobbl.fields.pcb.gerber import _eval, macro_shapes, parse_excellon

DATA = Path(__file__).parent / "data" / "pcb"
MM = 1e-3


def mm(*xy):
    return tuple(v * MM for v in xy)


@pytest.fixture(scope="module")
def k8():
    return read_kicad_pcb(DATA / "two_layer_k8.kicad_pcb")


@pytest.fixture(scope="module")
def k5():
    return read_kicad_pcb(DATA / "two_layer_k5.kicad_pcb")


@pytest.fixture(scope="module")
def four():
    return read_board(DATA / "four_layer_no_stackup.kicad_pcb")


@pytest.fixture(scope="module")
def gbr():
    return read_board(DATA / "gerber_two_layer")


def pad(board, name):
    return next(p for p in board.pads if p.name == name)


def test_sexpr_tokenizer_handles_quotes_escapes_and_nesting():
    tree = parse_sexpr('(a "b \\"c\\" (d)" (e 1.5 (f)) "" g\\h)')
    assert tree == [["a", 'b "c" (d)', ["e", "1.5", ["f"]], "", "g\\h"]]
    with pytest.raises(ValueError, match="open bracket"):
        parse_sexpr("(a (b)")


def test_kicad8_stackup(k8):
    st = k8.stackup
    assert st.source == "kicad stackup"
    assert k8.copper_layers() == ["F.Cu", "B.Cu"]
    assert st.board_thickness == pytest.approx(1.58 * MM)
    assert st.height_between("F.Cu", "B.Cu") == pytest.approx(1.51 * MM)
    assert st.er_between("F.Cu", "B.Cu") == pytest.approx(4.5)
    assert st.layer("F.Cu").z_bottom == pytest.approx(1.545 * MM)
    assert st.layer("B.Cu").z_bottom == 0.0
    assert st.dielectrics[0].loss_tangent == pytest.approx(0.02)


def test_kicad8_pads_follow_the_footprint_rotation(k8):
    r1 = pad(k8, "R1.1")
    assert r1.at == pytest.approx(mm(110, -100))           # footprint at (110, 99.175) turned 90 degrees, y flipped
    assert pad(k8, "R1.2").at == pytest.approx(mm(110, -98.35))
    assert r1.angle_deg == pytest.approx(90)
    # the pad is 0.8 by 0.95 before turning, so 0.95 wide and 0.8 tall on the board
    assert r1.contains([mm(110.45, -100)])[0]
    assert not r1.contains([mm(110, -100.45)])[0]
    j2 = pad(k8, "J1.2")
    assert j2.at == pytest.approx(mm(97.46, -100))       # J1 is turned 270 degrees, so pin 2 lands left of pin 1
    assert j2.angle_deg == pytest.approx(270)
    assert j2.layers == ["F.Cu", "B.Cu"] and j2.drill == pytest.approx(1.0 * MM) and j2.net == "GND"
    assert pad(k8, "C1.2").at == pytest.approx(mm(120.775, -98.35))
    assert k8.nets == ["GND", "SIG", "VCC", "TEST"]


def test_kicad8_tracks_and_the_arc(k8):
    sig = k8.net_items("SIG").tracks
    total = sum(t.length for t in sig)
    straight = 4 + 3
    arc = math.pi * 1.5
    assert total == pytest.approx((straight + arc) * MM, rel=0.005)
    arc_pieces = [t for t in sig if 104 * MM - 1e-9 <= min(t.start[0], t.end[0]) and max(t.start[0], t.end[0])
                  <= 107 * MM + 1e-9]
    assert len(arc_pieces) > 6
    c, r = np.array(mm(105.5, -100)), 1.5 * MM
    for t in arc_pieces:
        mid = (np.array(t.start) + np.array(t.end)) / 2
        assert np.hypot(*(mid - c)) > 0.99 * r            # chords stay within 1% of the radius
        assert np.hypot(*(np.array(t.start) - c)) == pytest.approx(r, rel=1e-6)
    assert all(t.width == pytest.approx(0.25 * MM) for t in sig)


def test_kicad8_zone_outline_and_teardrops(k8):
    assert len(k8.zones) == 1                            # the teardrop fillet is left out
    z = k8.zones[0]
    assert (z.net, z.layer) == ("GND", "B.Cu")
    assert z.area() == pytest.approx(278.5 * MM * MM)    # 30 x 10 mm minus a 3 x 3 and a 5 x 2.5 mm cut out
    assert z.contains([mm(105, -103), mm(100, -100), mm(114.5, -98.25)]).tolist() == [True, False, False]
    assert len(k8.outline) == 1
    assert k8.bbox() == pytest.approx((95 * MM, -105 * MM, 125 * MM, -95 * MM))
    assert any("teardrop" in n for n in k8.notes)


def test_net_graph_merges_a_routed_net_into_one_edge(k8):
    g = net_graph(k8, "SIG")
    assert g.connected and sorted(g.pads) == ["J1.1", "R1.1"]
    assert len(g.edges) == 1
    e = g.edges[0]
    assert {e.a, e.b} == {"J1.1", "R1.1"} and e.kind == "track" and e.layer == "F.Cu"
    assert e.length == pytest.approx(sum(t.length for t in k8.net_items("SIG").tracks))
    assert e.width == pytest.approx(0.25 * MM)
    assert e.polyline.shape[1] == 2 and len(e.polyline) == len(k8.net_items("SIG").tracks) + 1


def test_net_graph_vias_join_layers(k8):
    g = net_graph(k8, "VCC")
    assert g.connected
    vias = [e for e in g.edges if e.kind == "via"]
    assert len(vias) == 2
    for e in vias:
        assert e.layers == ("F.Cu", "B.Cu")
        assert e.length == pytest.approx(k8.stackup.distance("F.Cu", "B.Cu"))
        assert e.length == pytest.approx(1.545 * MM)
        assert e.drill == pytest.approx(0.4 * MM) and e.diameter == pytest.approx(0.8 * MM)
    assert g.length("track") == pytest.approx((3 + 3 + 3.225) * MM)
    assert {e.layer for e in g.edges if e.kind == "track"} == {"F.Cu", "B.Cu"}


def test_net_graph_joins_through_a_zone(k8):
    g = net_graph(k8, "GND")
    assert g.connected and sorted(g.pads) == ["C1.2", "J1.2"]
    zone_edges = [e for e in g.edges if e.kind == "zone"]
    assert len(zone_edges) == 2 and all(e.length == 0 for e in zone_edges)
    tied = {e.b for e in zone_edges}
    assert "J1.2" in tied and any(n.endswith(":B.Cu") for n in tied)
    assert net_graph(k8, "TEST").pads == ["TP1.1"]


def test_net_graph_reports_unrouted_pads(k8):
    from wobbl.fields.pcb import Board, Track

    b = Board("cut", "kicad", k8.stackup, [t for t in k8.tracks if t.net != "SIG" or t.start[0] > 105 * MM],
              k8.vias, k8.pads, k8.zones, k8.nets)
    g = net_graph(b, "SIG")
    assert not g.connected and len(g.unconnected) == 1
    assert isinstance(b.tracks[0], Track)
    with pytest.raises(ValueError, match="no net named"):
        net_graph(k8, "NOPE")


def test_reference_planes_find_the_pour_under_a_trace(k8):
    planes = reference_planes(k8, "F.Cu", mm(102.5, -100))
    assert planes == [("B.Cu", "GND", pytest.approx(1.51 * MM))]
    assert reference_planes(k8, "F.Cu", mm(100.5, -100)) == []     # over the clearance cut out around J1.1
    assert reference_planes(k8, "B.Cu", mm(102.5, -100)) == []


def test_kicad5_board_reads_the_same(k8, k5):
    assert k5.copper_layers() == ["F.Cu", "B.Cu"]       # the file renamed them Top and Bottom
    assert k5.nets == ["GND", "SIG", "VCC"]
    for p5 in k5.pads:
        p8 = pad(k8, p5.name)
        assert p5.at == pytest.approx(p8.at) and p5.angle_deg == pytest.approx(p8.angle_deg)
        assert p5.layers == p8.layers and p5.net == p8.net
    assert "default" in k5.stackup.source and any("No usable stackup" in n for n in k5.notes)
    assert k5.stackup.board_thickness == pytest.approx(1.6 * MM)
    z = k5.zones[0]
    assert z.area() == pytest.approx(278.5 * MM * MM) and z.stroke == pytest.approx(0.254 * MM)
    assert len(k5.outline) == 1 and k5.bbox() == pytest.approx(k8.bbox())   # the old style arc closes the edge
    g = net_graph(k5, "SIG")
    assert g.connected and len(g.edges) == 1
    assert g.edges[0].length == pytest.approx((4 + 2 * math.hypot(1.5, 1.5) + 3) * MM)
    assert all(net_graph(k5, n).connected for n in k5.nets)


def test_four_layer_board_gets_a_default_stackup(four):
    st = four.stackup
    assert four.copper_layers() == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    assert "JLCPCB" in st.source and any("No usable stackup" in n for n in four.notes)
    assert st.board_thickness == pytest.approx(1.57 * MM)
    assert st.height_between("F.Cu", "In1.Cu") == pytest.approx(0.2 * MM)
    assert st.height_between("In1.Cu", "In2.Cu") == pytest.approx(1.065 * MM)
    assert st.layer("In1.Cu").thickness == pytest.approx(17.5e-6)
    assert st.layer("B.Cu").thickness == pytest.approx(35e-6)
    assert st.er_between("F.Cu", "B.Cu") == pytest.approx((0.4 * 4.4 + 1.065 * 4.6) / 1.465)
    u1 = pad(four, "U1.1")
    assert u1.at == pytest.approx(mm(52.475, -51.905)) and u1.angle_deg == pytest.approx(180)
    assert pad(four, "R5.1").at == pytest.approx(mm(59.2, -50)) and pad(four, "R5.1").layers == ["B.Cu"]
    blind = next(v for v in four.vias if v.layers == ("In1.Cu", "B.Cu"))
    assert four.via_layers(blind) == ["In1.Cu", "In2.Cu", "B.Cu"]


def test_four_layer_graph_and_planes(four):
    g = net_graph(four, "CLK")
    assert g.connected
    through = [e for e in g.edges if e.kind == "via" and e.a.startswith("V1:")]
    assert sum(e.length for e in through) == pytest.approx(four.stackup.distance("F.Cu", "B.Cu"))
    assert "V1:In1.Cu" in g.nodes                       # the inner layer track lands on the through via
    assert g.length("track") == pytest.approx((3.525 + 3.2 + 1.905) * MM)
    assert net_graph(four, "GND").connected
    planes = reference_planes(four, "F.Cu", mm(54, -51.905))
    assert planes == [("In2.Cu", "GND", pytest.approx((0.2 + 0.0175 + 1.065) * MM))]


def test_gerber_set_reads_like_the_kicad_board(gbr, k8):
    assert gbr.source == "gerber" and gbr.copper_layers() == ["F.Cu", "B.Cu"]
    assert gbr.stackup.source == ".gbrjob"
    assert gbr.stackup.height_between("F.Cu", "B.Cu") == pytest.approx(1.51 * MM)
    assert gbr.nets == ["GND", "SIG", "TEST", "VCC"]
    for name in ("J1.1", "J1.2", "R1.1", "R1.2", "C1.1", "C1.2", "TP1.1"):
        pg, pk = pad(gbr, name), pad(k8, name)
        assert pg.at == pytest.approx(pk.at, abs=1e-9) and pg.net == pk.net and pg.layers == pk.layers
        probe = np.array(pk.at) + np.array([[0.4, 0], [0, 0.4], [0.47, 0], [0, 0.47]]) * MM
        assert pg.contains(probe).tolist() == pk.contains(probe).tolist()
    assert pad(gbr, "R1.1").shape == "roundrect"
    assert sorted(pad(gbr, "TP1.1").size) == pytest.approx([1.0 * MM, 2.0 * MM])     # macro with primitives 21 and 5
    assert pad(gbr, "J1.1").drill == pytest.approx(1.0 * MM)
    assert len(gbr.vias) == 3 and all(v.drill == pytest.approx(0.4 * MM) for v in gbr.vias)
    assert all(v.layers == ("F.Cu", "B.Cu") for v in gbr.vias)
    sig = sum(t.length for t in gbr.net_items("SIG").tracks)
    assert sig == pytest.approx(sum(t.length for t in k8.net_items("SIG").tracks), rel=1e-3)
    assert len(gbr.outline) == 1 and gbr.bbox() == pytest.approx(k8.bbox(), abs=1e-6)
    assert any("not conductors" in n for n in gbr.notes)


def test_gerber_region_with_a_clear_flash(gbr):
    z = gbr.zones[0]
    assert (z.net, z.layer) == ("GND", "B.Cu")
    hole = 8 * math.sin(2 * math.pi / 16) * 1.0          # 16 sided circle of 1 mm radius cleared out
    assert z.area() == pytest.approx((278.5 - hole) * MM * MM)
    assert z.contains([mm(122, -103), mm(122, -101.5), mm(100, -100)]).tolist() == [False, True, False]


def test_gerber_net_graphs(gbr):
    for net in gbr.nets:
        assert net_graph(gbr, net).connected, net
    g = net_graph(gbr, "SIG")
    assert len(g.edges) == 1 and g.edges[0].length == pytest.approx(11.699 * MM, rel=1e-3)
    assert reference_planes(gbr, "F.Cu", mm(102.5, -100)) == [("B.Cu", "GND", pytest.approx(1.51 * MM))]


def test_gerber_zip_and_single_file(tmp_path, gbr):
    z = tmp_path / "board.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (DATA / "gerber_two_layer").iterdir():
            zf.write(f, f"gerbers/{f.name}")
    from_zip = read_board(z)
    one = read_board(DATA / "gerber_two_layer" / "board-F_Cu.gbr")
    listed = read_gerbers(sorted((DATA / "gerber_two_layer").iterdir()))
    for b in (from_zip, one, listed):
        assert b.summary()["pads"] == 7 and len(b.vias) == 3 and len(b.zones) == 1 and b.nets == gbr.nets


def test_gerbers_without_attributes_get_traced_nets(tmp_path):
    rename = {"board-F_Cu.gbr": "board.gtl", "board-B_Cu.gbr": "board.gbl", "board-Edge_Cuts.gbr": "board.gko",
              "board-PTH.drl": "board.drl"}
    for src, dst in rename.items():
        text = (DATA / "gerber_two_layer" / src).read_text()
        lines = [ln for ln in text.splitlines() if not ln.startswith(("%TF", "%TA", "%TO", "%TD"))]
        (tmp_path / dst).write_text("\n".join(lines) + "\n")
    b = read_board(tmp_path)
    assert b.copper_layers() == ["F.Cu", "B.Cu"] and len(b.outline) == 1
    assert len(b.nets) == 4 and any("traced" in n for n in b.notes)
    assert len(b.vias) == 3                               # small round flashes on both layers become vias

    def net_at(x, y):
        return next(p.net for p in b.pads if np.allclose(p.at, mm(x, y)))

    assert net_at(100, -100) == net_at(110, -100)        # J1.1 and R1.1 share the routed net
    assert net_at(110, -98.35) == net_at(119.225, -98.35)
    assert net_at(97.46, -100) == net_at(120.775, -98.35)    # joined only through the pour
    assert len({net_at(100, -100), net_at(110, -98.35), net_at(97.46, -100), net_at(115, -103)}) == 4
    assert all(net_graph(b, n).connected for n in b.nets)


def test_macro_expressions_and_primitives():
    assert _eval("$1+$1", {1: 0.25}) == pytest.approx(0.5)
    assert _eval("($1-0.1)x2/4", {1: 0.5}) == pytest.approx(0.2)
    assert _eval("-$2X3", {2: 1.0}) == pytest.approx(-3.0)
    rings, unknown = macro_shapes(["21,1,2,1,0,0,90", "1,1,0.5,3,0", "99,1,2"], [])
    assert unknown == {"99"} and len(rings) == 2
    assert np.ptp(rings[0][:, 0]) == pytest.approx(1.0) and np.ptp(rings[0][:, 1]) == pytest.approx(2.0)


def test_excellon_formats():
    hits = parse_excellon("M48\n;FILE_FORMAT=4:3\nMETRIC,LZ\nT1F00S00C0.500\n%\nT01\nX0037Y00045\nY0015\nM30\n")
    assert hits == [(pytest.approx(37 * MM), pytest.approx(4.5 * MM), pytest.approx(0.5 * MM)),
                    (pytest.approx(37 * MM), pytest.approx(15 * MM), pytest.approx(0.5 * MM))]


def test_default_stackups():
    two = default_stackup(["F.Cu", "B.Cu"])
    assert two.board_thickness == pytest.approx(1.6 * MM) and two.er_between("F.Cu", "B.Cu") == 4.5
    six = default_stackup([f"L{i}" for i in range(6)])
    assert six.board_thickness == pytest.approx(1.6 * MM) and len(six.dielectrics) == 5
    thin = default_stackup(["F.Cu", "B.Cu"], 0.8 * MM)
    assert thin.board_thickness == pytest.approx(0.8 * MM) and "scaled" in thin.source


def test_board_helpers(k8):
    on_b = k8.items_on("B.Cu")
    assert len(on_b.zones) == 1 and len(on_b.vias) == 3 and {p.name for p in on_b.pads} == {"J1.1", "J1.2"}
    assert len(on_b.tracks) == 1
    gnd = k8.net_items("GND")
    assert len(gnd.pads) == 2 and len(gnd.vias) == 1 and len(gnd.zones) == 1
    with pytest.raises(ValueError, match="no copper layer"):
        k8.items_on("In1.Cu")


def test_unreadable_files_raise_plain_errors(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match="neither a KiCad board nor a Gerber"):
        read_board(bad)
    cut = tmp_path / "cut.kicad_pcb"
    cut.write_text((DATA / "two_layer_k8.kicad_pcb").read_text()[:2000])
    with pytest.raises(ValueError, match="cut short"):
        read_board(cut)
    with pytest.raises(ValueError, match="does not exist"):
        read_board(tmp_path / "missing.kicad_pcb")
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(DATA / "gerber_two_layer" / "board.gbrjob", empty)
    with pytest.raises(ValueError, match="no Gerber"):
        read_board(empty)
