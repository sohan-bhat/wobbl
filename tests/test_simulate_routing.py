"""Generality proof for wobbl simulate: a varied set of things and questions from every domain goes through the
router, which has to pick the right domain, the right kind of thing and the right physics without any of them
naming a command. Nothing is solved here.

A case whose physics belong to a domain that has not registered them yet still checks the domain and the kind,
and checks the physics as soon as they register. Run this file directly for the accuracy table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from wobbl.simulate import registry
from wobbl.simulate.router import RouteError, recognize, route

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "src" / "wobbl" / "validation" / "parts"
EX = ROOT / "examples"


@dataclass
class Case:
    thing: str
    question: str
    domain: str
    kind: str | None = None
    physics: tuple[str, ...] | None = None     # the exact analyses, in any order
    given: tuple[str, ...] = ()                # --physics given explicitly

    @property
    def label(self) -> str:
        t = Path(self.thing).name if "/" in self.thing or "{" in self.thing else self.thing
        return f"{t} | {self.question}"[:90]


CASES = [
    # parts: CAD files and parametric parts
    Case(str(PARTS / "shelf_bracket_fillet.step"), "will it hold 200 N?", "part", "step", ("stress",)),
    Case(str(PARTS / "enclosure_shell.step"), "will it stay under 60 C with 8 W inside a sealed box?", "part", "step",
         ("heat",)),
    Case(str(PARTS / "drone_arm_truss.step"), "does it resonate between 4000 and 9000 rpm?", "part", "step",
         ("vibration",)),
    Case(str(EX / "strut-buckling" / "strut.py"), "will it buckle under 2 kN?", "part", "parametric", ("buckling",)),
    Case(str(EX / "motor-mount" / "mount.py"), "will it last a million cycles at 150 N?", "part", "parametric",
         ("stress", "fatigue")),
    Case(str(EX / "shelf-bracket" / "bracket.py"), "will 3 kN bend it for good?", "part", "parametric", ("plastic",)),
    Case(str(PARTS / "enclosure_shell.step"), "will it hold 50 N and stay under 60 C at 5 W?", "part", "step",
         ("stress", "heat")),
    Case(str(PARTS / "cast_motor_mount.step"), "", "part", "step", ("vibration", "heat"), ("vibration,heat",)),
    Case(str(EX / "heat-sink" / "heatsink.py"), "how hot does it get at 20 W with a small fan?", "part",
         "parametric", ("heat",)),
    Case("{tmp}/lid.stl", "what is the safety factor under 40 lbf?", "part", "stl", ("stress",)),
    Case(str(EX / "shelf-bracket" / "output"), "how many cycles until it cracks at R=-1?", "part", "result",
         ("fatigue",)),
    Case(str(EX / "drone-arm" / "arm.py"), "random vibration at gevs-qual on the shaker", "part", "parametric",
         ("vibration",)),
    Case(str(PARTS / "shelf_bracket_sharp.step"), "how fast does a crack grow from the hole?", "part", "step",
         ("stress", "crack")),
    Case(str(EX / "bracket-crack" / "bracket.py"), "damage tolerance check with a 1 mm flaw at the pin hole",
         "part", "parametric", ("stress", "crack")),
    Case(str(PARTS / "cast_motor_mount.step"), "how often should we inspect it for cracks?", "part", "step",
         ("stress", "crack")),
    Case(str(EX / "motor-mount-fatigue" / "output"), "how long until a crack from the fillet grows critical?",
         "part", "result", ("crack",)),
    Case(str(EX / "battery-pack" / "pack.py"), "", "part", "parametric", ("stress",)),
    Case(str(EX / "drone-arm" / "arm.py"), "what does the motor unbalance do when it is swept from 10 to 500 Hz?",
         "part", "parametric", ("vibration",)),
    Case(str(PARTS / "cast_motor_mount.step"), "how does it respond to a 2 g sine sweep from 20 to 2000 Hz?", "part",
         "step", ("vibration",)),
    Case(str(EX / "heat-sink" / "heatsink.py"), "what are the natural frequencies of the fins?", "part",
         "parametric", ("vibration",)),
    Case(str(EX / "shelf-bracket" / "output"), "how many repeats of this load history before it cracks?", "part",
         "result", ("fatigue",)),
    # several asks in one question: each load case gets its own physics, and a drop replaces a strength run only
    # for the clause it was asked in
    Case(str(EX / "shelf-bracket" / "bracket.py"), "will it survive a 1 m drop and hold a 10 kg shelf?", "part",
         "parametric", ("drop", "stress")),
    Case(str(PARTS / "enclosure_shell.step"), "will it survive a drop and stay under 60 C at 5 W?", "part", "step",
         ("drop", "heat")),
    Case(str(PARTS / "drone_arm_truss.step"), "will it hold 200 N and not resonate at 3000 rpm?", "part", "step",
         ("stress", "vibration")),
    Case(str(PARTS / "enclosure_shell.step"), "will it survive a 1.2 m drop onto tile?", "part", "step", ("drop",)),
    Case(str(EX / "drone-arm" / "arm.py"), "it is bolted at the base, will it hold 20 N and survive a 1 m drop?",
         "part", "parametric", ("stress", "drop")),
    # circuits: netlists and circuits in words
    Case("", "will a blue LED on 5V with a 47 ohm resistor fry?", "circuit", "led", ("ratings",)),
    Case("buck converter from 12 V to 3.3 V at 2 A", "how big is the ripple?", "circuit", "buck", ("transient",)),
    Case("", "will a MOSFET carrying 15 A with 8 mOhm Rds on overheat?", "circuit", "mosfet", ("ratings",)),
    Case("{tmp}/board.cir", "what is v(out)?", "circuit", "netlist", ("operating-point",)),
    Case("", "what is the RC time constant of 10k and 100nF?", "circuit", "rc", ("transient",)),
    Case("{tmp}/board.kicad_sch", "will anything overheat at 12 V?", "circuit", "kicad", ("ratings",)),
    Case("{tmp}/export.net", "what is the output voltage?", "circuit", "kicad", ("operating-point",)),
    Case("{tmp}/board.kicad_sch", "", "circuit", "kicad", ("ratings",)),
    Case("a 555 astable blinking a red LED from a 9 V battery", "how fast does it blink?", "circuit", "led",
         ("transient",)),
    Case("", "will an LM358 non inverting amplifier with a gain of 11 be stable?", "circuit", "words", ("ac",)),
    Case("{tmp}/board.cir", "what happens across part tolerances?", "circuit", "netlist", ("tolerance",)),
    # mechanisms: specs, URDF, MJCF and words
    Case("", "can two mg996r servos lift 500 g with a robot arm?", "mechanism", "arm"),
    Case("four bar linkage with a 120 mm ground, 30 mm crank, 125 mm coupler and 45 mm rocker", "where does it bind?",
         "mechanism", "fourbar"),
    Case("", "a gear train of 12:36 and 10:40 on an fa-130 motor, what output torque?", "mechanism", "gears"),
    Case("", "will the lamp tip over with 500 g at 120 mm reach?", "mechanism", "tip"),
    Case("{tmp}/robot.urdf", "can the shoulder hold the arm out level?", "mechanism", "urdf"),
    Case("{tmp}/arm.xml", "", "mechanism", "mjcf"),
    Case("{tmp}/gripper.toml", "does the gripper close in time?", "mechanism", "spec"),
    Case("a pendulum 1 m long released at 30 degrees", "what is its period?", "mechanism", "pendulum"),
    Case("", "what peak force when a 200 g part is dropped onto steel from 500 mm?", "mechanism", "impact"),
    # flights: bodies, rockets and motors
    Case(str(EX / "rocket-launch" / "rocket.toml"), "what is the apogee?", "flight", "rocket", ("trajectory",)),
    Case(str(EX / "knsb-motor" / "motor.toml"), "", "flight", "motor", ("motor",)),
    Case("a 9 cm rock dropped from 300 m", "how fast does it land?", "flight", "body", ("trajectory",)),
    Case("", "how far does a baseball thrown at 40 m/s at 35 degrees go?", "flight", "body", ("trajectory",)),
    Case("", "a golf ball launched at 70 m/s, how far does it fly?", "flight", "body", ("trajectory",)),
    Case("a steel ball falls from 10 m", "", "flight", "body", ("trajectory",)),
    # molecules, elements, materials, fuels, propellants and cells
    Case("H2O", "", "chemistry", "molecule"),
    Case("CCO", "", "chemistry", "molecule"),
    Case("c1ccccc1", "what is its boiling point?", "chemistry", "molecule"),
    Case("caffeine", "", "chemistry", "molecule"),
    Case("Fe", "", "chemistry", "element"),
    Case("methane", "what is the adiabatic flame temperature in air?", "chemistry", "fuel"),
    Case("18650", "can it run away at 60 C?", "chemistry", "cell"),
    Case("knsb", "what Isp at 5 MPa with an area ratio of 8?", "chemistry", "propellant"),
    Case("ap:70,al:16,htpb:14", "", "chemistry", "propellant"),
    Case("6061-t6", "", "chemistry", "material"),
    # reactions
    Case("CH4 + 2 O2 -> CO2 + 2 H2O", "how hot is the flame?", "chemistry", "reaction"),
    Case("2 H2 + O2 -> 2 H2O", "", "chemistry", "reaction"),
    Case("", "what are the products of C3H8 + 5 O2 -> 3 CO2 + 4 H2O at 10 bar?", "chemistry", "reaction"),
    Case("N2 + 3 H2 <=> 2 NH3", "how much ammonia at equilibrium?", "chemistry", "reaction"),
    # fluids
    Case("", "40 m of 25 mm pipe carrying 30 L/min of water, what is the pressure drop?", "fluid", "pipe"),
    Case("", "what is the drag on a 22 cm sphere at 20 m/s?", "fluid", "drag"),
    Case("apcsf_10x4.7", "can a 1.2 kg quad with 4 rotors hover?", "fluid", "prop"),
    Case("", "a 12 m boat hull at 8 knots, how much power does it need?", "fluid", "hull"),
    Case("{tmp}/cavity", "", "fluid", "cfd"),
    Case("", "air through a 100 mm duct at 40 CFM, how much pressure is lost?", "fluid", "pipe"),
    # fields
    Case("", "a solenoid with 800 turns at 0.6 A and a 6 mm stroke, does it pull 8 N?", "field", "solenoid"),
    Case("", "what impedance is a 0.3 mm microstrip on 0.17 mm fr4?", "field", "microstrip"),
    Case("", "where does a 2.4 GHz dipole 58 mm long resonate?", "field", "dipole"),
    Case("", "what are the room modes of a 5 x 4 x 2.7 m room?", "field", "room"),
    Case("", "two plates 100 mm across with a 2 mm gap at 3 kV, will it arc?", "field", "plates"),
]


def _files(tmp: Path) -> None:
    (tmp / "lid.stl").write_text("solid lid\nendsolid lid\n")
    (tmp / "board.cir").write_text("* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n")
    (tmp / "board.kicad_sch").write_text("(kicad_sch (version 20231120) (generator eeschema))\n")
    (tmp / "export.net").write_text("(export (version D) (design (source board.kicad_sch)))\n")
    (tmp / "robot.urdf").write_text('<?xml version="1.0"?>\n<robot name="arm"><link name="base"/></robot>\n')
    (tmp / "arm.xml").write_text("<mujoco model='arm'><worldbody/></mujoco>\n")
    (tmp / "gripper.toml").write_text('[[bodies]]\nname = "finger"\n\n[[joints]]\nname = "hinge"\n')
    (tmp / "cavity" / "system").mkdir(parents=True, exist_ok=True)
    (tmp / "cavity" / "system" / "controlDict").write_text("application icoFoam;\n")


@pytest.fixture(scope="module")
def tmp_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("things")
    _files(d)
    return d


def check(case: Case, tmp: Path) -> dict:
    """Route one case and say what it got: the domain, the kind and the physics, and whether each is right.
    physics_ok is None while the domain has not registered the physics the case expects."""
    thing = case.thing.replace("{tmp}", str(tmp))
    found = recognize(thing, case.question)
    got = {"domain": found[0][1].name if found else "", "kind": found[0][2] if found else "", "physics": None}
    registered = {a.name for a in registry.analyses(got["domain"])}
    if registered:
        try:
            plan = route(thing, case.question, list(case.given))
            got.update(domain=plan.domain, kind=plan.kind, physics=tuple(plan.names))
        except RouteError as e:
            got["error"] = str(e)
    got["domain_ok"] = got["domain"] == case.domain and (case.kind is None or got["kind"] == case.kind)
    want = case.physics
    if want and set(want) <= {a.name for a in registry.analyses(case.domain)}:
        got["physics_ok"] = got["physics"] is not None and sorted(got["physics"]) == sorted(want)
    else:
        got["physics_ok"] = None
    return got


def test_there_are_enough_varied_cases():
    domains = {c.domain for c in CASES}
    assert len(CASES) >= 40
    assert domains >= {"part", "circuit", "mechanism", "flight", "chemistry", "fluid", "field"}
    assert sum(c.kind == "reaction" for c in CASES) >= 4 and sum(c.kind == "molecule" for c in CASES) >= 4


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_routes(case, tmp_files):
    got = check(case, tmp_files)
    assert got["domain_ok"], (case, got)
    if got["physics_ok"] is None:
        pytest.skip(f"{case.domain} has not registered {', '.join(case.physics or ('its physics',))} yet")
    assert got["physics_ok"], (case, got)


def accuracy(tmp: Path) -> tuple[float, float, int, list]:
    rows = [(c, check(c, tmp)) for c in CASES]
    dom = sum(g["domain_ok"] for _, g in rows) / len(rows)
    checked = [g for _, g in rows if g["physics_ok"] is not None]
    phys = sum(g["physics_ok"] for g in checked) / len(checked) if checked else 0.0
    return dom, phys, len(checked), rows


def test_routing_accuracy(tmp_files, capsys):
    dom, phys, n, rows = accuracy(tmp_files)
    with capsys.disabled():
        print(f"\nrouting: domain and kind right on {dom:.0%} of {len(rows)} cases, physics right on {phys:.0%} of "
              f"the {n} whose physics are registered")
    assert dom == 1.0 and phys == 1.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        _files(Path(t))
        dom, phys, n, rows = accuracy(Path(t))
        for c, g in rows:
            mark = "ok" if g["domain_ok"] and g["physics_ok"] is not False else "WRONG"
            print(f"{mark:5} {g['domain']:9} {g['kind']:10} {','.join(g['physics'] or ['-']):16} {c.label}")
        print(f"\ndomain and kind {dom:.0%} of {len(rows)}, physics {phys:.0%} of {n} checkable")
