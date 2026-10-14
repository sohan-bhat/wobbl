"""Mechanism studies end to end: the outcome, the trajectory table, the report and the checks. Skipped when
MuJoCo is not installed."""
import csv
import json

import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")

WEAK_ARM = """
name = "weak arm"
question = "Can an SG90 hold this arm out level?"
hold = "1 s"
floor = true

[[body]]
name = "base"
shape = "box"
size = [60, 60, 20]
center = [0, 0, 10]
mass = 200
support = true

[[body]]
name = "link"
parent = "base"
pos = [0, 0, 60]
to = [150, 0, 0]
mass = 60
joint = { name = "shoulder", axis = "-y" }

[[mass]]
name = "load"
body = "link"
mass = 300
pos = [150, 0, 0]

[[drive]]
joint = "shoulder"
servo = "sg90"

[[move]]
t = 0
shoulder = 20
"""


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_a_pendulum_writes_every_output_and_passes_its_checks(tmp_path):
    from wobbl.mechanism.study import MechanismOutcome, run_mechanism

    o = run_mechanism("pendulum", {"periods": 1, "angle": "90 deg"}, tmp_path, "How long is a 90 degree swing?")
    assert o.verdict == "ok" and o.confidence == "high", o.confidence_reasons
    assert {c.name for c in o.checks} >= {"hand calculation", "time step convergence", "energy conservation"}
    for f in ("outcome.json", "summary.txt", "report.html", "trajectory.csv", "model.xml"):
        assert (tmp_path / f).exists(), f
    d = json.loads((tmp_path / "outcome.json").read_text())
    assert d["domain"] == "mechanism" and d["events"][0]["name"] == "release" and d["model"]["degrees_of_freedom"] == 1
    assert d["values"]["trajectory"] == "trajectory.csv" and d["images"]["swing"] == "images/swing.png"
    rows = _rows(tmp_path / "trajectory.csv")
    assert 100 < len(rows) <= 500 and set(rows[0]) >= {"t_s", "pivot_deg", "pivot_rpm", "pivot_Nm", "bob_z_m"}
    pb = json.loads((tmp_path / "playback.json").read_text())
    assert pb["manifest"]["format"] == "wobbl-playback" and pb["manifest"]["frames"] == len(pb["t"]) <= 150
    bob = next(b for b in pb["bodies"] if b["name"] == "bob")
    assert bob["shape"] == "sphere" and not bob["static"] and len(bob["pos"]) == len(pb["t"])
    w, x, y, z = bob["quat"][0]
    assert w * w + x * x + y * y + z * z == pytest.approx(1, abs=1e-3)
    assert bob["pos"][0][2] == pytest.approx(-0.0, abs=1e-6)      # released level with the pivot at 90 degrees
    assert len(pb["joints"]["pivot"]["angle"]["values"]) == len(pb["t"])
    assert (tmp_path / "playback.json").stat().st_size < 300_000
    back = MechanismOutcome.load(tmp_path / "outcome.json")
    assert back.events == o.events and back.runtime_s > 0
    assert "Events" in (tmp_path / "report.html").read_text()


def test_a_spec_with_a_weak_servo_fails_and_names_a_stronger_one(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    spec = tmp_path / "arm.toml"
    spec.write_text(WEAK_ARM)
    o = run_mechanism(str(spec), {}, tmp_path / "out")
    assert o.verdict == "fails" and "shoulder drive stalls" in o.headline
    assert o.values["drives"]["shoulder"]["peak_usage"] > 1
    assert o.fixes and o.fixes[0]["kind"] == "motor"
    assert any(e["name"] == "shoulder stalls" for e in o.events)
    assert o.question.startswith("Can an SG90")


def test_a_model_file_runs_as_written(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    xml = tmp_path / "drop.xml"
    xml.write_text("<mujoco><worldbody><geom type='plane' size='1 1 0.1'/><body pos='0 0 0.2'><freejoint/>"
                   "<geom type='box' size='0.02 0.02 0.02' mass='0.1'/></body></worldbody></mujoco>")
    o = run_mechanism(str(xml), {"duration": "0.5 s"}, tmp_path / "out")
    assert o.model["degrees_of_freedom"] == 6 and o.values["contacts"]       # the box lands on the plane
    assert not (tmp_path / "out" / "model.xml").exists()
    with pytest.raises(ValueError, match="duration and timestep"):
        run_mechanism(str(xml), {"servo": "sg90"}, tmp_path / "bad")


URDF = """<robot name="two_link">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="upper"><inertial><origin xyz="0.075 0 0"/><mass value="0.2"/>
    <inertia ixx="1e-5" iyy="4e-4" izz="4e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.075 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="0.01" length="0.15"/></geometry>
    </collision></link>
  <link name="fore"><inertial><origin xyz="0.06 0 0"/><mass value="0.3"/>
    <inertia ixx="1e-5" iyy="4e-4" izz="4e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/><origin xyz="0 0 0.1"/>
    <axis xyz="0 -1 0"/><limit lower="-3" upper="3" effort="0.6" velocity="4"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="fore"/><origin xyz="0.15 0 0"/>
    <axis xyz="0 -1 0"/><limit lower="-3" upper="3" effort="2.0" velocity="4"/></joint>
</robot>
"""


def test_a_spec_over_a_urdf_takes_the_file_s_effort_limits(tmp_path):
    from wobbl.mechanism.study import run_mechanism

    (tmp_path / "arm.urdf").write_text(URDF)
    (tmp_path / "spec.toml").write_text('model = "arm.urdf"\nhold = "0.3 s"\n[[drive]]\njoint = "shoulder"\n'
                                        '[[drive]]\njoint = "elbow"\n[[move]]\nt = 0\nshoulder = 0\nelbow = 0\n'
                                        '[[move]]\nt = 0.6\nshoulder = 20\nelbow = -20\n')
    o = run_mechanism(str(tmp_path / "spec.toml"), {}, tmp_path / "out")
    sh = o.values["drives"]["shoulder"]
    assert sh["motor"] == "the limits arm.urdf gives shoulder" and sh["peak_usage"] > 1     # 0.6 N m cannot hold it
    # holding level: 0.2 kg at 75 mm and 0.3 kg at 210 mm from the shoulder
    assert abs(sh["torque_Nm"]) > 9.80665 * (0.2 * 0.075 + 0.3 * 0.21) * 0.99
    assert o.values["drives"]["elbow"]["peak_usage"] < 0.5 and o.verdict == "fails"
