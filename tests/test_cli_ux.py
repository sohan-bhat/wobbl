"""The command line people meet first: one entry through simulate, friendly answers, a doctor that never hangs and
results that read well on a terminal and stay plain in a pipe."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from rich.console import Console

from wobbl.cli import admin, ui
from wobbl.cli.main import cli
from wobbl.setup import probe

ROOT = Path(__file__).resolve().parents[1]
PART = str(ROOT / "examples" / "shelf-bracket" / "bracket.py")
ROCK = "a 9 cm rock dropped from 300 m"
ANSI = re.compile(r"\x1b\[")


def run(*args):
    return CliRunner().invoke(cli, list(args))


def test_the_bare_form_is_exactly_simulate():
    for words in ([ROCK], [PART, "will it hold 200 N?"], ["will a blue LED on 5V with a 47 ohm resistor fry?"]):
        bare = run(*words, "--plan", "--json")
        full = run("simulate", *words, "--plan", "--json")
        assert bare.exit_code == full.exit_code == 0, bare.output
        assert json.loads(bare.output) == json.loads(full.output)
    first = run("--plan", "--json", ROCK)              # an option first goes to simulate too
    assert first.exit_code == 0 and json.loads(first.output)["domain"] == "flight"


def test_a_greeting_gets_a_friendly_answer_not_a_click_error():
    for words in ("hello", "hey what do yiu do", "what do you do"):
        for args in ([words], ["simulate", words]):
            r = run(*args)
            assert r.exit_code == 0, r.output
            assert "No such command" not in r.output and "Usage:" not in r.output and "error" not in r.output
            assert "simulates things before you build them" in r.output
            assert len(re.findall(r"^  wobbl simulate .*\"$", r.output, re.M)) == 3
    r = run("simulate", "hello", "--json")
    assert json.loads(r.output)["simulated"] is False


def test_nonsense_says_so_kindly_and_exits_nonzero():
    r = run("asdf qwer zxcv")
    assert r.exit_code == 2
    assert 'found nothing to simulate in "asdf qwer zxcv"' in r.output and "wobbl simulate --list" in r.output


def test_a_part_question_without_its_file_says_what_to_pass():
    for args in (["will my bracket crack under 200 N"], ["simulate", "will my bracket crack under 200 N"]):
        r = run(*args)
        assert r.exit_code == 2
        assert "needs its file" in r.output
        assert 'wobbl simulate bracket.step "will my bracket crack under 200 N"' in " ".join(r.output.split())
        assert ".step" in r.output


def test_a_typo_of_a_command_asks_what_was_meant():
    r = run("doctr")
    assert r.exit_code == 2 and "Did you mean wobbl doctor?" in " ".join(r.output.split())
    r = run("missing.step", "will it hold?")
    assert r.exit_code == 2 and "file not found: missing.step" in r.output


def test_a_hanging_probe_is_stopped_and_reported():
    hang = probe.Probe("hang", "os:system", ("sleep 30",), timeout=1)
    t0 = time.time()
    r = probe.run(hang)
    assert time.time() - t0 < 10
    assert r.ok is False and r.timed_out and "no answer within 1 s" in r.detail
    fine = probe.run(probe.Probe("gmsh", "wobbl.setup.probe:gmsh_version", timeout=60))
    assert fine.ok is True and fine.detail


def test_doctor_never_hangs(monkeypatch):
    monkeypatch.setattr(admin, "doctor_probes", lambda: [
        probe.Probe("stuck solver", "os:system", ("sleep 30",), timeout=1),
        probe.Probe("broken", "wobbl.setup.probe:no_such_check", timeout=20),
        probe.Probe("gmsh", "wobbl.setup.probe:gmsh_version", timeout=60)])
    t0 = time.time()
    r = run("doctor")
    assert time.time() - t0 < 30
    assert r.exit_code == 0, r.output
    assert any(ln.strip().startswith("timed out") and "stuck solver" in ln for ln in r.output.splitlines())
    assert any(ln.strip().startswith("missing") and "broken" in ln for ln in r.output.splitlines())
    assert any(ln.strip().startswith("ok") and "gmsh" in ln for ln in r.output.splitlines())
    assert "gave no answer in time" in " ".join(r.output.split())
    j = json.loads(run("doctor", "--json").output)
    assert j["stuck solver"]["ok"] is False and j["gmsh"]["ok"] is True and "wobbl" in j


LONG = ("the peak stress converged to within 2 percent over three meshes, and the reactions balance the "
        "applied loads, so the number is reliable enough to design with " * 2)


def outcome():
    return SimpleNamespace(
        verdict="fails", headline="The LED fries at 36.9 mA, above its 30 mA rating.",
        metrics=[("LED current", "36.9 mA"), ("rating", "30 mA"), ("resistor power", "63.9 mW")],
        confidence="high", confidence_reasons=[LONG],
        checks=[{"status": "pass", "name": "hand calculation", "summary": LONG}],
        fixes=[{"title": "Use a 91 ohm resistor", "summary": LONG}], assumptions=[LONG, "Room temperature."],
        images={"schematic": "/tmp/wobbl-out/a/very/long/path/" + "x" * 90 + "/schematic.svg"},
        report="/tmp/wobbl-out/report.html")


def test_results_are_plain_text_in_a_pipe(capsys):
    ui.outcome(outcome())
    out = capsys.readouterr().out
    assert not ANSI.search(out) and "╭" not in out and "━" not in out
    lines = out.splitlines()
    assert any(ln.startswith("FAILS  The LED fries") for ln in lines)
    assert all(ln == ln.rstrip() for ln in lines)                    # no padding at line ends
    assert all(len(ln) <= ui.MAX_WIDTH for ln in lines if "/tmp/" not in ln)
    assert any(ln.strip().startswith("schematic") and ln.endswith("/schematic.svg") for ln in lines)  # one line
    nums = [ln for ln in lines if ln.strip().startswith(("LED current", "rating", "resistor power"))]
    assert len({len(ln) for ln in nums}) == 1                         # the numbers line up on the right


def test_no_color_is_plain_even_on_a_terminal(monkeypatch):
    tty = Console(force_terminal=True, width=200)
    assert not ui.plain(tty)
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.plain(tty)


def test_the_result_panel_keeps_to_a_readable_width(monkeypatch):
    wide = Console(force_terminal=True, width=200, highlight=False, color_system="truecolor")
    monkeypatch.setattr(ui, "console", wide)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    with wide.capture() as cap:
        ui.outcome(outcome())
    text = re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
    panel = [ln for ln in text.splitlines() if ln.startswith(("╭", "│", "╰"))]
    assert panel and all(len(ln) == ui.MAX_WIDTH for ln in panel)
    assert all(len(ln.rstrip()) <= ui.MAX_WIDTH for ln in text.splitlines() if "/tmp/" not in ln)
    assert not any(re.search(r"\b\w+-\n", ln) for ln in text.splitlines())
    narrow = Console(force_terminal=True, width=60, highlight=False)
    monkeypatch.setattr(ui, "console", narrow)
    with narrow.capture() as cap:
        ui.outcome(outcome())
    assert all(len(ln) <= 60 for ln in re.sub(r"\x1b\[[0-9;]*m", "", cap.get()).splitlines() if "/tmp/" not in ln)


def test_a_run_leaves_a_short_summary_of_its_stages(monkeypatch):
    tty = Console(force_terminal=True, width=100, highlight=False)
    monkeypatch.setattr(ui, "console", tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    with tty.capture() as cap:
        with ui.progress(eta=30) as cb:
            cb("mesh", "level 0")
            time.sleep(0.15)
            cb("solve", "CalculiX")
            time.sleep(0.15)
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", cap.get())
    assert re.search(r"done in \d+\.\d s\s+mesh \d\.\d s, solve \d\.\d s", text)
