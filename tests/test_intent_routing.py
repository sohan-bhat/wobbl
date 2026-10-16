"""Free-form requests in everyday words reach the right domain, kind and physics. The phrases come from one shared
table (wobbl.simulate.intent.SYNONYMS) that each analysis takes into its keywords, and a phrase outweighs a single
word, so "a rocket thrust test" is a static fire and not a flight. Nothing is solved here."""
from __future__ import annotations

import pytest

from wobbl.simulate.intent import SYNONYMS, best, phrases, weight
from wobbl.simulate.router import route

# (words, domain, kind or None for any, physics)
CASES = [
    # solid motors on the ground
    ("a rocket thrust test", "flight", "motor", ("motor",)),
    ("static fire a 54 mm knsb motor", "flight", "motor", ("motor",)),
    ("put my H motor on the test stand", "flight", "motor", ("motor",)),
    ("what does the thrust curve of a 38 mm KNSB motor look like", "flight", "motor", ("motor",)),
    ("how long does a knsb motor burn", "flight", "motor", ("motor",)),
    ("what chamber pressure does my sugar motor reach", "flight", "motor", ("motor",)),
    ("test fire my rocket motor and show the burn", "flight", "motor", ("motor",)),
    # flights
    ("how high does a C6 rocket go", "flight", "rocket", ("trajectory",)),
    ("what is the apogee of a rocket on an H128", "flight", "rocket", ("trajectory",)),
    ("how far does a baseball thrown at 40 m/s go", "flight", "body", ("trajectory",)),
    ("a 9 cm rock dropped from 300 m", "flight", "body", ("trajectory",)),
    ("how far does a golf ball launched at 70 m/s fly", "flight", "body", ("trajectory",)),
    # tests on a part named in words
    ("drop my phone case from 1.5 m", "part", None, ("drop",)),
    ("drop test a bracket from 2 m onto tile", "part", None, ("drop",)),
    ("will a 3d printed lid crack if it falls off a desk", "part", None, ("drop",)),
    ("shake test my drone arm", "part", None, ("vibration",)),
    ("bend test a shelf bracket with 300 N", "part", None, ("stress",)),
    ("pull test a hook with 500 N", "part", None, ("stress",)),
    ("will a bracket hold 200 N", "part", None, ("stress",)),
    ("heat soak a 15 W heat sink", "part", None, ("heat",)),
    ("how hot does a 10 W LED get on this heat sink", "part", None, ("heat",)),
    ("will this enclosure overheat with 8 W inside", "part", None, ("heat",)),
    # mechanisms
    ("will my robot arm tip over with 2 kg", "mechanism", "tip", ("stability",)),
    ("can two mg996r servos lift 500 g with a robot arm", "mechanism", "arm", ("torque",)),
    # circuits
    ("will a 9V battery and a red LED without a resistor fry", "circuit", "led", ("ratings",)),
    ("does a blue LED on 5V with a 47 ohm resistor burn out", "circuit", "led", ("ratings",)),
    # chemistry
    ("how hot is a methane flame", "chemistry", "fuel", ("combustion",)),
    ("caffeine IR spectrum", "chemistry", "molecule", ("spectrum",)),
    ("can an 18650 run away at 60 C", "chemistry", "cell", ("runaway",)),
    # flow and fields
    ("drone with apcsf_10x4.7 props, 1.2 kg, 4 rotors, can it hover", "fluid", "prop", None),
    ("pressure drop through 20 m of 25 mm pipe at 30 L/min", "fluid", "pipe", None),
    ("solenoid with 800 turns at 0.6 A, does it pull 8 N", "field", "solenoid", None),
]


def test_there_are_enough_varied_phrasings():
    assert len(CASES) >= 30
    assert len({c[1] for c in CASES}) >= 7


@pytest.mark.parametrize("words,domain,kind,physics", CASES, ids=[c[0][:50] for c in CASES])
def test_free_form_words_route(words, domain, kind, physics):
    p = route(words, "")
    assert p.domain == domain, p.reasons
    if kind:
        assert p.kind == kind, p.reasons
    if physics:
        assert set(p.names) == set(physics), p.reasons


def test_a_phrase_outweighs_a_word():
    assert weight("thrust test") > weight("rocket")
    assert weight("how far up") > weight("how high") > weight("apogee")
    w, said, _ = best(phrases("static fire"), "a rocket thrust test")
    assert said == "thrust test" and w == weight("thrust test")


def test_every_synonym_compiles_and_names_something():
    import re
    for name in SYNONYMS:
        rxs = phrases(name)
        assert rxs, name
        for rx in rxs:
            re.compile(rx)


TYPICAL = [("drop my phone case from 1.5 m", "case.py", {"material": ["pc"], "mass": ["150 g @ 0,0,6 r=26"]}),
           ("will a bracket hold 200 N", "bracket.py", {"load": ["200 N down @ top and within 12 of 65,12.5,6"]}),
           ("how hot does a 10 W LED get on this heat sink", "heatsink.py", {"material": ["6063-t5"]}),
           ("will a steel hook hold 3 kN", "hook.py", {"load": ["3 kN -z @ within 5 of 22,0,-16"]})]


@pytest.mark.parametrize("words,part,brings", TYPICAL, ids=[t[0][:40] for t in TYPICAL])
def test_a_part_named_in_words_runs_on_a_typical_one(words, part, brings):
    p = route(words, "")
    assert p.thing.endswith(part) and p.kind == "parametric" and p.question == words
    said = " ".join(t + " " + o for t, o in p.assumed)
    assert "no part file given" in said and "your own STEP, STL or Python part first" in said
    got = {n: v for n, v, _ in p.defaults}
    for k, v in brings.items():
        assert got.get(k) == v, got


def test_a_material_in_the_words_beats_the_typical_one():
    p = route("will a steel hook hold 3 kN", "")
    assert "material" not in {n for n, _, _ in p.defaults}
