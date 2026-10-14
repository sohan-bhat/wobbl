"""Any circuit on a breadboard: every pin lands on its own net, the wires carry the run's currents, and the
bench drawing shows them."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import needs_spice
from wobbl.circuits import board, draw
from wobbl.circuits.netlist import parse_spice

NETLISTS = {
    "divider": "* divider\nV1 in 0 9\nR1 in out 10k\nR2 out 0 4.7k\n",
    "bridge": "* bridge\nV1 a b SIN(0 12 50)\nD1 a pos DR\nD2 b pos DR\nD3 0 a DR\nD4 0 b DR\nC1 pos 0 1000u\n"
              "R1 pos 0 470\n.model DR D(IS=1e-9)\n",
    "darlington": "* darlington\nV1 vcc 0 12\nV2 in 0 3\nRB in b1 10k\nQ1 vcc b1 b2 QN\nQ2 c2 b2 0 QN\n"
                  "RL vcc c2 100\n.model QN NPN(BF=100)\n",
    "opamp": "* inverting\nV1 vp 0 12\nV2 vn 0 -12\nVIN in 0 SIN(0 0.5 1k)\nR1 in inv 10k\nR2 inv out 47k\n"
             "XU1 0 inv vp vn out OPAMP\nRL out 0 10k\n.subckt OPAMP inp inn vcc vee out\nE1 x 0 inp inn 1e5\n"
             "R1 x out 100\n.ends\n",
    "cmos": "* cmos\nVDD vdd 0 5\nVIN in 0 PULSE(0 5 0 1n 1n 1u 2u)\nMP out in vdd vdd PF\nMN out in 0 0 NF\n"
            "CL out 0 10p\n.model PF PMOS(VTO=-1)\n.model NF NMOS(VTO=1)\n",
    "meter": "* meter\nV1 vcc 0 6\nVM vcc a 0\nR1 a b 100\nD1 b 0 DS\n.model DS D(IS=1e-14)\n",
    "ladder": "* ladder\nV1 in 0 5\n" + "".join(f"R{k} n{k - 1} n{k} 1k\nC{k} n{k} 0 1u\n" for k in range(1, 7))
              .replace("n0", "in"),
    "floating": "* floating\nV1 vcc 0 9\nR1 vcc a 1k\nV2 a b 1.5\nR2 b 0 2k\n",
    "sallen-key": "* sk\nVIN in 0 1\nR1 in a 10k\nR2 a b 10k\nC1 a out 10n\nC2 b 0 10n\nE1 out 0 b out 1e5\n",
}


def conductors(pl):
    """Which holes the board and its wires join: a strip of five, a whole rail, or the two ends of a wire."""
    up: dict = {}

    def find(x):
        while up.setdefault(x, x) != x:
            x = up[x]
        return x

    def key(h):
        col, row = h
        return ("rail", row) if row in ("t+", "t-", "b-", "b+") else ("strip", col, row in "abcde")

    for w in pl["wires"]:
        up[find(key(w["ends"][0]))] = find(key(w["ends"][1]))
    return lambda h: find(key(h))


@pytest.mark.parametrize("name", NETLISTS)
def test_every_pin_lands_on_its_own_net(name):
    c = parse_spice(NETLISTS[name])
    pl = board.plan(c)
    where = conductors(pl)
    on = {}
    for p in pl["parts"]:
        for pin in p["pins"]:
            on.setdefault(pin["net"], set()).add(where(pin["hole"]))
    for ins in pl["instruments"]:
        for ld in ins["leads"]:
            on.setdefault(ld["net"], set()).add(where(ld["hole"]))
    for net, places in on.items():
        assert len(places) == 1, f"{name}: net {net} is split over {len(places)} conductors"
    owners = [next(iter(v)) for v in on.values()]
    assert len(owners) == len(set(owners)), f"{name}: two nets share a conductor"
    holes = [tuple(pin["hole"]) for p in pl["parts"] for pin in p["pins"]]
    assert len(holes) == len(set(holes)), f"{name}: two pins share a hole"
    placed = {p["name"] for p in pl["parts"]} | {i["name"] for i in pl["instruments"]}
    assert placed == {e.name for e in c.elements if e.nodes}


def test_parts_get_their_bodies_and_sources_become_instruments():
    pl = board.plan(parse_spice(NETLISTS["cmos"]))
    bodies = {p["name"]: (p["body"], len(p["pins"])) for p in pl["parts"]}
    assert bodies["MP"] == ("to92", 3) and bodies["CL"] == ("radial-capacitor", 2)  # body tied to source
    kinds = {i["name"]: i["kind"] for i in pl["instruments"]}
    assert kinds == {"VDD": "supply", "VIN": "generator"}
    assert [r[1] for r in next(i for i in pl["instruments"] if i["name"] == "VIN")["readings"]] == ["kHz", "V"]
    pl = board.plan(parse_spice(NETLISTS["meter"]))
    assert {i["name"]: i["kind"] for i in pl["instruments"]} == {"V1": "supply", "VM": "meter"}
    assert {p["name"]: p["body"] for p in pl["parts"]} == {"R1": "axial-resistor", "D1": "axial-diode"}


def test_wires_carry_the_runs_currents_by_kirchhoff():
    c = parse_spice(NETLISTS["divider"])
    i = 9 / 14.7e3
    data = {"v(in)": [9.0], "v(out)": [9 * 4.7 / 14.7], "i(v1)": [-i]}
    pl = board.plan(c, data)
    assert {p["name"]: round(p["current"] * 1e3, 4) for p in pl["parts"]} == {"R1": round(i * 1e3, 4),
                                                                             "R2": round(i * 1e3, 4)}
    rail = next(w for w in pl["wires"] if w["kind"] == "rail" and w["net"] == "in")
    assert abs(rail["current"]) == pytest.approx(i)  # all of the supply's current comes down the rail jumper
    assert next(i_ for i_ in pl["instruments"] if i_["name"] == "V1")["current"] == pytest.approx(i)
    r2 = next(p for p in pl["parts"] if p["name"] == "R2")
    assert r2["power"] == pytest.approx(i * i * 4.7e3) and r2["heat"] == pytest.approx(i * i * 4.7e3 / 0.25, abs=1e-3)


def test_bench_drawing_moves_current_at_its_size():
    c = parse_spice("* two loads\nV1 vcc 0 9\nR1 vcc 0 90\nR2 vcc 0 900\n")
    data = {"v(vcc)": [9.0], "i(v1)": [-0.11]}
    svg = draw.board_setup(board.plan(c, data), "warm")
    root = ET.fromstring(svg)
    assert root.get("data-live") and "WARM" in svg and "110 mA" in svg and "SUPPLY CURRENT" in svg
    speeds = sorted(float(v) for v in re.findall(r'data-flow="([\d.]+)"', svg))
    assert speeds[0] == pytest.approx(90 * 10 / 110, rel=0.01) and speeds[-1] == pytest.approx(90)  # R2 and the lead
    assert len(svg.encode()) < 40_000 and "base64" not in svg
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    assert "data-drift" in svg  # R1 burns 0.9 W in a 0.25 W part


def test_a_changing_current_plays_its_shape():
    c = parse_spice("* rc\nV1 in 0 5\nR1 in out 1k\nC1 out 0 1u\n")
    t = np.linspace(0, 5e-3, 200)
    vout = 5 * (1 - np.exp(-t / 1e-3))
    pl = board.plan(c, {"time": t, "v(in)": np.full_like(t, 5.0), "v(out)": vout, "i(v1)": -(5 - vout) / 1e3})
    r1 = next(p for p in pl["parts"] if p["name"] == "R1")
    assert pl["span"] == pytest.approx(5e-3) and r1["k"][0] == pytest.approx(1) and r1["k"][-1] < 0.02
    svg = draw.board_setup(pl)
    assert 'data-live="3"' in svg and "data-k" in svg


@pytest.mark.solver
@needs_spice
def test_any_netlist_runs_instrumented_and_draws(tmp_path):
    from wobbl.solvers import base

    c = parse_spice(NETLISTS["darlington"] + ".op\n")
    vecs = [f"v({n})" for n in c.nodes() if n != "0"] + [f"i({v})" for _, _, v in c.probes()]
    data = base.for_domain("circuit").run(c.spice(instrument=True, analyses=c.analyses, save=False), tmp_path,
                                          vecs, "run")[0]
    out = draw.write_circuit(tmp_path / "images", c, data)
    svg = (tmp_path / "images" / "setup.svg").read_text()
    assert set(out) == {"setup", "schematic"} and svg.count("data-flow") >= 4
    rl = next(p for p in board.plan(c, data)["parts"] if p["name"] == "RL")
    assert rl["current"] > 0.05 and rl["heat"] > 1  # the load runs hot in a quarter watt resistor
