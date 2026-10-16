"""Typical things Wobbl designs from a few words when the user gives no spec: a hobby motor for a static fire and a
model rocket for a flight. Each is recorded as a guess with the exact options or file that replace it."""
from __future__ import annotations

from wobbl.chem.spec import read_spec as read_motor, spec_from_flags
from wobbl.chem.typical import EXAMPLE, class_of, design as motor
from wobbl.motion.spec import load_toml, read_spec
from wobbl.motion.typical import design as rocket, toml
from wobbl.simulate.router import route
from wobbl.simulate.runner import preview


def _spec(flags):
    return read_motor(spec_from_flags(flags["propellant"], [flags["grain"]], flags["throat"], flags["exit"],
                                      flags["casing"], float(flags["efficiency"])))


def test_with_nothing_said_the_motor_is_the_54_mm_example():
    flags, said = motor("a rocket thrust test")
    assert flags == EXAMPLE and "54 mm KNSB" in said and "J class" in said
    assert len(_spec(flags).motor.grains) == 5


def test_the_words_size_the_motor():
    for words, letter, dia in (("put my H motor on the test stand", "H", "38 mm"), ("an I class kndx motor", "I", "38"),
                               ("a J350 motor static fire", "J", "54 mm"), ("a 75 mm sugar motor", None, "75 mm")):
        flags, said = motor(words)
        assert dia in said, said
        if letter:
            assert f"{letter} class" in said, said
        _spec(flags)   # every design reads as a valid spec
    assert motor("an I class kndx motor")[0]["propellant"] == "kndx"
    assert class_of(300) == "H" and class_of(1238) == "J"


def test_a_typical_rocket_flies_the_motor_named():
    spec, said = rocket("how high does a C6 rocket go")
    st = read_spec(spec)
    assert st.motor.name == "C6" and "25 mm rocket" in said
    assert load_toml(toml(spec)) == spec
    spec, said = rocket("what is the apogee of a rocket on an H128")
    assert read_spec(spec).motor.name == "H128" and "an H128" in said
    assert "C6-5, the most flown" in rocket("how high does a rocket go")[1]


def test_the_plan_says_what_it_designed_and_how_to_give_the_real_one():
    for words, needle, override in (("a rocket thrust test", "no motor spec given", "--grain 'bates d=48"),
                                    ("how high does a C6 rocket go", "no rocket spec given", "your own rocket.toml")):
        steps = preview(route(words, ""), {})
        said = [(t, o) for s in steps for t, o in s.job.assumptions]
        assert any(needle in t and override in o for t, o in said), said
        assert not any(s.error for s in steps), [s.error for s in steps]


def test_without_a_resistor_is_taken_at_its_word():
    steps = preview(route("will a 9V battery and a red LED without a resistor fry", ""), {})
    said = [t for s in steps for t, _ in s.job.assumptions]
    assert any("no resistor, as the words say" in t for t in said) and not any("470 ohm" in t for t in said)


def test_a_typical_motor_is_named_for_its_propellant_and_casing():
    steps = preview(route("a rocket thrust test", ""), {})
    assert steps[0].job.options["casing"] == ["6061-t6 od=54 wall=1.6 L=400 sf=2"]


def test_a_radio_and_a_propeller_get_usual_values_and_a_heat_question_asks_for_its_power():
    def said(words):
        steps = preview(route(words, ""), {})
        return [(t, o) for s in steps for t, o in s.job.assumptions], [s.error for s in steps if s.error]

    assert not said("how far does a dipole antenna reach")[1]
    guesses, errors = said("how much thrust does an apcsf_10x4.7 prop make")
    assert any(o == "--rpm 6000" for _, o in guesses) and not errors
    _, errors = said("will this enclosure overheat")
    assert errors and errors[0].startswith("how much heat does it make?") and "--power '12 W'" in errors[0]
