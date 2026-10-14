"""What a motion run writes, which the website reads: outcome.json, trajectory.csv, summary.txt, report.html and
light chart images, through run_motion and the wobbl motion command."""
import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wobbl.cli.main import cli
from wobbl.motion.spec import read_spec
from wobbl.motion.study import COLUMNS, MotionOutcome, run_motion

ROOT = Path(__file__).resolve().parents[1]
PHASES = {"rail", "boost", "coast", "drogue", "main", "fall", "landed"}
ROCK = ["--name", "rock", "--diameter", "9 cm", "--density", "2700 kg/m3", "--cd", "0.6", "--height", "300 m"]


def _rows(folder: Path) -> list[dict]:
    with open(folder / "trajectory.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _check_contract(folder: Path, phases: set[str]):
    d = json.loads((folder / "outcome.json").read_text())
    assert d["domain"] == "motion" and d["verdict"] in ("ok", "marginal", "fails")
    for key in ("headline", "metrics", "confidence", "confidence_reasons", "checks", "fixes", "events", "runtime_s",
                "model"):
        assert key in d, key
    assert all(set(c) >= {"name", "status", "summary"} for c in d["checks"])
    assert all(list(e) == ["name", "t_s", "alt_m", "x_m", "speed_mps"] for e in d["events"])
    assert d["runtime_s"] > 0 and isinstance(d["model"]["states"], int)
    assert all(e == e.lower() for e in d["model"]["effects"])
    with open(folder / "trajectory.csv", newline="") as fh:
        assert next(csv.reader(fh)) == COLUMNS
    rows = _rows(folder)
    assert len(rows) <= 500 and {r["phase"] for r in rows} == phases
    times = [float(r["t_s"]) for r in rows]
    assert times == sorted(times)
    for e in d["events"]:
        assert any(abs(float(r["t_s"]) - e["t_s"]) < 1e-9 and abs(float(r["alt_m"]) - e["alt_m"]) < 0.01
                   for r in rows), e["name"]
    last = rows[-1]
    assert last["phase"] == "landed" and float(last["speed_mps"]) == 0 and float(last["alt_m"]) == 0
    assert (folder / "summary.txt").read_text().startswith(d["verdict"].upper())
    assert "Motion report" in (folder / "report.html").read_text()
    for img in d["images"].values():
        from PIL import Image
        assert (folder / img).stat().st_size < 100_000 and Image.open(folder / img).mode == "P"
    return d, rows


def test_the_rock_from_the_command_line(tmp_path):
    r = CliRunner().invoke(cli, ["motion", *ROCK, "--question", "How fast does it hit?", "-o", str(tmp_path / "o"),
                                 "--json"])
    assert r.exit_code == 0, r.output
    d, rows = _check_contract(tmp_path / "o", {"fall", "landed"})
    assert [e["name"] for e in d["events"]] == ["release", "impact"]
    assert d["model"] == {"states": 4, "effects": ["gravity falling off with altitude", "1976 standard atmosphere",
                                                   "air drag"]}
    assert d["question"] == "How fast does it hit?" and d["part"] == "rock"
    assert float(rows[-2]["speed_mps"]) == pytest.approx(d["events"][-1]["speed_mps"], abs=1e-3)


def test_the_rocket_writes_every_phase(tmp_path):
    o = run_motion(read_spec(ROOT / "examples" / "rocket-launch" / "rocket.toml"), tmp_path / "o")
    d, rows = _check_contract(tmp_path / "o", PHASES - {"fall"})
    assert d["model"]["states"] == 6 and "weathercocking" in d["model"]["effects"]
    assert set(o.images) == {"flight", "path"}
    loaded = MotionOutcome.load(tmp_path / "o" / "outcome.json")
    assert loaded.events == o.events and loaded.model == o.model


def test_check_and_report_read_a_motion_result(tmp_path):
    out = tmp_path / "o"
    assert CliRunner().invoke(cli, ["motion", *ROCK, "-o", str(out), "--json"]).exit_code == 0
    r = CliRunner().invoke(cli, ["check", str(out), "--json"])
    assert r.exit_code == 0 and json.loads(r.output)["confidence"] == "high"
    assert CliRunner().invoke(cli, ["report", str(out)]).exit_code == 0


@pytest.mark.parametrize("args,words", [
    (["--name", "rock", "--height", "300 m"], "diameter"),
    (["--diameter", "9 cm", "--mass", "1 kg", "--density", "2700 kg/m3"], "not both"),
    (["--diameter", "9 cm", "--mass", "1 kg", "--wind", "5 bananas"], "--wind"),
    ([str(ROOT / "examples" / "rocket-launch" / "rocket.toml"), "--mass", "2 kg"], "cannot be combined"),
])
def test_bad_input_says_what_to_change(args, words, tmp_path):
    r = CliRunner().invoke(cli, ["motion", *args, "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and words in r.output


def test_a_spec_takes_todays_wind(tmp_path):
    from wobbl.motion.tools import study_from

    st = study_from(spec=str(ROOT / "examples" / "rocket-launch" / "rocket.toml"), wind="20 mph")
    assert st.wind == pytest.approx(8.9408)
