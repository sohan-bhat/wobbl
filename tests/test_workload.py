"""Any Wobbl command line as a workload: parsed by the CLI without running, priced by its domain's hook, by the
part it meshes, by its own history here or by its bundled example, and recorded when it finishes."""
from __future__ import annotations

import json
import sys

import pytest

from wobbl import estimate as E
from wobbl.jobs import workload as WL

M = E.Machine("test box", 6, 4, 16e9, 500e9)


@pytest.fixture(autouse=True)
def idle(monkeypatch):
    """Numbers on this machine are compared as if it were idle, whatever else is running."""
    monkeypatch.setattr(E, "load", lambda: 0.0)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    return tmp_path


def test_parse_finds_the_command_its_inputs_and_its_result_folder(tmp_path):
    spec = tmp_path / "rocket.toml"
    spec.write_text("[rocket]\n")
    w = WL.parse(["motion", str(spec), "-o", str(tmp_path / "res")])
    assert w.command == "motion" and w.inputs == [spec] and w.out == str(tmp_path / "res")
    assert WL.parse(["flow", "pipe", "--flow", "2 L/min"]).command == "flow pipe"
    assert WL.parse(["jobs", "run-plan", str(spec)]).out is None          # its default folder is not a choice
    with pytest.raises(WL.WorkloadError, match="not a Wobbl command"):
        WL.parse(["nosuch", "x"])
    with pytest.raises(WL.WorkloadError):
        WL.parse(["flow"])


def test_the_signature_follows_inputs_and_size_settings_but_not_the_result_folder(tmp_path):
    spec = tmp_path / "rocket.toml"
    spec.write_text("[rocket]\n")
    a = WL.parse(["motion", str(spec), "-o", "a"]).signature()
    assert a == WL.parse(["motion", str(spec), "-o", "b"]).signature()
    spec.write_text("[rocket]\nmass = 2\n")
    assert a != WL.parse(["motion", str(spec), "-o", "a"]).signature()


def test_a_domain_plugs_in_through_one_registry(home):
    def cost(w):
        return E.Cost("x", "flow", 120.0, 0.0, 2e8, 1e6, {}, ["From the flow domain's own model."])

    old = WL.REGISTRY.get("flow pipe")
    try:
        WL.register("flow pipe", domain="flow", cost=cost)
        e = E.estimate_command(["flow", "pipe"], M)
        assert e.verdict == "fits" and e.seconds == pytest.approx(120.0) and e.cost.solver == "wobbl flow pipe"
        assert "flow domain's own model" in e.basis[0]
        with pytest.raises(TypeError):
            WL.register("flow pipe", nonsense=1)
    finally:
        WL.REGISTRY.pop("flow pipe", None)
        if old:
            WL.REGISTRY["flow pipe"] = old


def test_unmeasured_commands_say_so_and_offer_a_job(home):
    e = E.estimate_command(["magnetic", "solenoid"], E.this_machine(home))
    assert e.verdict == "unknown" and not e.known and e.to_dict()["time_s"] == 0
    assert "not known" in e.basis[0] and e.options[0]["kind"] == "background"
    p = E.estimate_command(["flame", "propane"], M)
    assert p.verdict == "fits" and p.seconds == pytest.approx(28.1) and "bundled flame example" in p.basis[0]


def test_history_predicts_a_command_and_its_own_inputs_best(home):
    for t in (40.0, 50.0, 60.0):
        E.record("wobbl magnetic", 0.0, t, 0.0, 3e8, {"signature": "other", "disk": 2e6}, "run")
    e = E.estimate_command(["magnetic", "solenoid"], M)
    assert e.seconds == pytest.approx(50.0, rel=0.02) and "3 earlier wobbl magnetic runs" in e.basis[0]
    assert e.peak_bytes == pytest.approx(3e8)
    sig = WL.parse(["magnetic", "solenoid"]).signature()
    E.record("wobbl magnetic", 0.0, 12.0, 0.0, 1e8, {"signature": sig, "disk": 1e6}, "run")
    again = E.estimate_command(["magnetic", "solenoid"], M)
    assert again.seconds == pytest.approx(12.0) and "very magnetic study" in again.basis[0]


def test_a_finished_run_is_recorded_once_and_paired_with_its_estimate(home, tmp_path, monkeypatch):
    out = tmp_path / "res"
    out.mkdir()
    (out / "report.html").write_text("x" * 1000)
    argv = ["flame", "propane", "-o", str(out)]
    E.estimate_command(argv, E.this_machine(home))
    # under pytest nothing is recorded
    WL.record_outcome({"domain": "combustion", "t_total": 30.0}, out)
    assert E.load_calibration().get("runs", []) == []
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/wobbl", *argv])
    WL.record_outcome({"domain": "combustion", "t_total": 30.0}, out)
    WL.record_outcome({"domain": "combustion", "t_total": 30.0}, out)       # saved again after a new report
    runs = E.load_calibration()["runs"]
    assert len(runs) == 1 and runs[0]["solver"] == "wobbl flame" and runs[0]["measured_time"] == 30.0
    assert runs[0]["predicted_time"] == pytest.approx(28.1) and runs[0]["size"]["disk"] >= 1000
    assert runs[0]["size"]["domain"] == "combustion"
    assert E.factor("wobbl flame") == (pytest.approx(30.0 / 28.1), 1)


def test_the_outcome_writers_feed_the_history(home, tmp_path, monkeypatch):
    from wobbl.outcome_generic import GenericOutcome

    monkeypatch.setattr(sys, "argv", ["/x/wobbl", "motion", "-o", str(tmp_path)])
    o = GenericOutcome("motion", "", "how far", "ok", "lands", [], "high", [], [], [], t_total=4.5)
    o.save(tmp_path / "outcome.json")
    assert json.loads((tmp_path / "outcome.json").read_text())["t_total"] == 4.5
    assert [r["solver"] for r in E.load_calibration()["runs"]] == ["wobbl motion"]
