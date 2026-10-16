"""What the user asks for in the result and where it goes, read from the words and from --export and --out. The
words about the result are taken out before routing, so "on my desktop" never steers the physics."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.simulate.intent import read_output

OWNER = "in downloads give me a 3d gif of a sim of a rocket thrust test"

WORDS = [
    (OWNER, "a rocket thrust test", "gif", "downloads"),
    ("show me a video of this bracket bending under 200 N", "this bracket bending under 200 N", "mp4", ""),
    ("will a 9V battery and a red LED without a resistor fry, give me a png on my desktop",
     "will a 9V battery and a red LED without a resistor fry", "png", "desktop"),
    ("make an mp4 of a bracket holding 200 N", "a bracket holding 200 N", "mp4", ""),
    ("drop my phone case from 1.5 m and save it to ~/clips/drop.mp4", "drop my phone case from 1.5 m", "mp4",
     "~/clips/drop.mp4"),
    ("give me a report of a hook holding 3 kN in my documents", "a hook holding 3 kN", "report", "documents"),
    ("simulate a rocket static fire as an animation", "a rocket static fire", "gif", ""),
    ("how hot does a 10 W LED get on this heat sink, save a picture to /tmp/sink",
     "how hot does a 10 W LED get on this heat sink", "png", "/tmp/sink"),
    ("a 9 cm rock dropped from 300 m, video please", "a 9 cm rock dropped from 300 m", "mp4", ""),
]
UNTOUCHED = ["will a cable clip hold 20 N", "is it still safe at 60 C", "how high does a C6 rocket go",
             "will it get hot at 10 W", "drop my phone case from 1.5 m"]


@pytest.mark.parametrize("words,left,fmt,out", WORDS, ids=[w[0][:40] for w in WORDS])
def test_the_result_words_are_read_and_taken_out(words, left, fmt, out):
    o = read_output(words)
    assert (o.text, o.format, o.out) == (left, fmt, out)


@pytest.mark.parametrize("words", UNTOUCHED)
def test_words_that_ask_for_nothing_stay_as_they_are(words):
    o = read_output(words)
    assert (o.text, o.format, o.out) == (words, "", "")


def _plan(*args, home=None, monkeypatch=None):
    if home is not None:
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "cfg"))
    r = CliRunner().invoke(cli, [*args, "--plan", "--json"])
    assert r.exit_code == 0, r.output
    return json.loads(r.output)


def test_the_owners_sentence_plans_a_static_fire_gif_in_downloads(tmp_path, monkeypatch):
    d = _plan(OWNER, home=tmp_path, monkeypatch=monkeypatch)
    assert d["domain"] == "flight" and d["kind"] == "motor" and d["analyses"] == ["motor"]
    assert d["thing"] == "a rocket thrust test"
    assert d["export"]["format"] == "gif"
    assert d["export"]["to"] == str(tmp_path / "Downloads" / "wobbl-rocket-thrust-test.gif")
    assert "--export gif --out" in d["explicit"] and any("no motor spec given" in a["assumed"]
                                                       for a in d["assumptions"])


def test_the_flags_ask_for_the_same_as_the_words(tmp_path):
    d = _plan("a rocket thrust test", "--export", "png", "--out", str(tmp_path / "fire.png"))
    assert d["export"]["format"] == "png" and d["export"]["to"] == str(tmp_path / "fire.png")
    d = _plan("a rocket thrust test", "--out", str(tmp_path / "fire.gif"))     # the suffix says the format
    assert d["export"]["format"] == "gif"
    d = _plan("a rocket thrust test")
    assert "export" not in d


def test_output_words_never_steer_the_physics():
    d = _plan("will a 9V battery and a red LED without a resistor fry, give me a png on my desktop")
    assert d["domain"] == "circuit" and d["export"]["format"] == "png"
