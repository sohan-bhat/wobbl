"""Fatigue answers from stresses given directly, from a finished strength result and from a vibration result."""
import json
import math
import shutil
from pathlib import Path

import pytest

from wobbl.life.fatigue import KPSI
from wobbl.life.fatigue_study import FatigueStudy, count, run_fatigue

ROOT = Path(__file__).resolve().parents[1]
STEEL_1050 = f"custom:E=207000,nu=0.29,yield={49.5 * KPSI:.1f},uts={90 * KPSI:.1f},process=wrought"


def test_shigley_example_from_the_command_inputs(tmp_path):
    """Shigley Example 6-2 through the whole study: 1050 HR, a polished rotating beam at 55 kpsi fully reversed.
    The book reads f = 0.86 off its figure and gets about 77 500 cycles; Wobbl's fit of the figure gives 0.864."""
    st = FatigueStudy(material=STEEL_1050, amplitude=f"{55 * KPSI} MPa", surface="polished", reliability=0.5,
                      cycles="1e4")
    o = run_fatigue(st, tmp_path / "out")
    assert o.values["life"] == pytest.approx(77500, rel=0.03)
    assert o.verdict in ("holds", "marginal", "fails") and (tmp_path / "out" / "report.html").exists()
    assert o.values["curve"]["se_MPa"] == pytest.approx(45 * KPSI, rel=1e-4)    # Sut written to 0.1 MPa above
    # 55 kpsi is past the 49.5 kpsi yield, so the notch yields a little each cycle while the life stays on the line
    assert any(c.name == "first cycle yield" and c.status == "warn" for c in o.checks)
    # S'e comes from Shigley's rule of thumb, so the answer says so and is not high confidence
    assert o.values["data_estimated"] and o.confidence != "high"


def test_mean_stress_and_a_duty_cycle(tmp_path):
    st = FatigueStudy(material="6061-t6", amplitude="40 MPa", mean="40 MPa", blocks=["100% x 1e5", "50% x 1e7"],
                      repeats=2, kt=2.0, radius="2 mm")
    o = run_fatigue(st, tmp_path / "out")
    v = o.values
    # Peterson for aluminum, a = 0.635 mm: q = 0.759, Kf = 1.76 on nominal stresses
    assert v["q"] == pytest.approx(1 / (1 + 0.635 / 2)) and v["kf"] == pytest.approx(1 + v["q"])
    assert v["blocks"][0]["amplitude_MPa"] == pytest.approx(40 * v["kf"])
    assert v["damage"] == pytest.approx(2 * sum(b["cycles"] / b["life"] for b in v["blocks"]))
    spread = v["mean_model_spread"]
    assert spread["gerber"] > spread["goodman"] > spread["soderberg"]
    assert any(c.name == "mean stress model" for c in o.checks)


def test_from_a_strength_result(tmp_path):
    d = tmp_path / "mount"
    d.mkdir()
    for f in ("outcome.json", "study.json"):
        shutil.copy(ROOT / "examples" / "cast-motor-mount" / "output" / f, d / f)
    o = run_fatigue(FatigueStudy(source=str(d), cycles="1e6"), tmp_path / "out")
    peak = json.loads((d / "outcome.json").read_text())["governing_MPa"]
    v = o.values
    assert v["source"] == "strength result" and v["material"].startswith("Custom material")
    # on and off loading: amplitude and mean are each half the peak; a casting gets an as cast surface
    assert v["blocks"][0]["amplitude_MPa"] == pytest.approx(peak / 2) == v["blocks"][0]["mean_MPa"]
    assert "as-cast" in v["factors"]["surface ka"]["why"]
    names = {c.name: c.status for c in o.checks}
    assert names["strength result"] == "warn"           # the source was medium confidence, and says why
    assert o.verdict == "fails"                         # 62 MPa amplitude on a 165 MPa casting at a million cycles


def test_from_a_vibration_result(tmp_path):
    d = tmp_path / "vib"
    d.mkdir()
    (d / "outcome.json").write_text(json.dumps({
        "domain": "vibration", "part": "", "confidence": "high", "confidence_reasons": ["passed"],
        "values": {"duration_s": 120.0, "random": {"sigma_rms_MPa": 20.0, "rate_Hz": 150.0, "xyz": [1, 2, 3]}}}))
    (d / "study.json").write_text(json.dumps({"base": {"material": "6061-t6"}}))
    o = run_fatigue(FatigueStudy(source=str(d)), tmp_path / "out")
    bands = o.values["blocks"]
    assert [b["amplitude_MPa"] for b in bands] == pytest.approx([20.0, 40.0, 60.0])
    assert sum(b["cycles"] for b in bands) == pytest.approx(0.9973 * 150 * 120, rel=1e-4)
    assert "test" in o.headline and math.isfinite(o.values["sf"])


def test_errors_and_cycle_counts(tmp_path):
    with pytest.raises(ValueError):
        run_fatigue(FatigueStudy(), tmp_path / "a")
    with pytest.raises(ValueError):
        run_fatigue(FatigueStudy(material="copper", amplitude="10 MPa"), tmp_path / "b")
    with pytest.raises(ValueError):
        run_fatigue(FatigueStudy(source=str(tmp_path)), tmp_path / "c")
    assert count("2,000,000") == 2e6 and count("5 million") == 5e6 and count("1e7") == 1e7


def test_a_load_history_from_a_strength_result_and_in_mpa(tmp_path):
    d = tmp_path / "mount"
    d.mkdir()
    for f in ("outcome.json", "study.json"):
        shutil.copy(ROOT / "examples" / "cast-motor-mount" / "output" / f, d / f)
    peak = json.loads((d / "outcome.json").read_text())["governing_MPa"]
    # three on and off cycles to the studied load and one to 1.2 times it: rainflow finds the big one first
    o = run_fatigue(FatigueStudy(source=str(d), history="0, 1, 0, 1, 0, 1.2, 0, 1, 0", repeats=1000),
                    tmp_path / "out")
    v = o.values
    big = max(v["blocks"], key=lambda b: b["amplitude_MPa"])
    assert big["amplitude_MPa"] == pytest.approx(0.6 * peak, rel=1e-6) and big["mean_MPa"] == pytest.approx(0.6 * peak)
    assert sum(b["cycles"] for b in v["blocks"]) == pytest.approx(4.0)
    assert "load history" in o.headline and any("rainflow" in a for a in o.assumptions)
    # the same history in MPa against a material, with no result at all
    o2 = run_fatigue(FatigueStudy(material="6061-t6", history="0, 80, 0, 80, -80, 80"), tmp_path / "out2")
    assert max(b["amplitude_MPa"] for b in o2.values["blocks"]) == pytest.approx(80.0)
    with pytest.raises(ValueError):
        run_fatigue(FatigueStudy(source=str(d), history="0, 1", blocks=["100% x 10"]), tmp_path / "out3")
