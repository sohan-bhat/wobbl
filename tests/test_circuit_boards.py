"""Any circuit on a breadboard: every pin lands on its own net and the wires carry the run's currents."""
import pytest

from wobbl.circuits import board
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
