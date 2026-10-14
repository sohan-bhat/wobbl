"""Mechanism studies of real public robot models, MJCF and URDF, each asked a plain question.

None of these models was tuned for: they are fetched as their authors publish them, pinned to a commit, with only
the files their collision shapes need (about 1.2 MB in all), and deleted when the session ends. Each must load,
run the analyses its question asks for and pass its own checks. The set takes about ten minutes, so it runs when
WOBBL_ZOO=1 is set, and is skipped without MuJoCo or a network.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from wobbl.mechanism import install

MENAGERIE = "google-deepmind/mujoco_menagerie/affef0836947b64cc06c4ab1cbf0152835693374/"
MUJOCO = "google-deepmind/mujoco/28009f9105cd92784b7b0b30c0605a5e29107a77/model/"
DMC = "google-deepmind/dm_control/c22ec28159c541c8c6b6546cd56d41dcaf7fb33b/dm_control/suite/"
GYM = "Farama-Foundation/Gymnasium/9938990977eb5e85ea4523ecf5cb2678c1708ba4/gymnasium/envs/mujoco/assets/"
BULLET = "bulletphysics/bullet3/d3afe76132fc76b0f9b2359cbaef47a3766e7fbb/examples/pybullet/gym/pybullet_data/"
TUTORIAL = "ros/urdf_tutorial/050f1e47cfdb2c5f3eb0746bc15c57e6a870faef/urdf/"
DRAKE = "RobotLocomotion/drake/d2ef91eeffc298cc2cbf676cf237bdc213d19f23/examples/"
DMC_COMMON = ["common/materials.xml", "common/skybox.xml", "common/visual.xml"]
KUKA = ["kuka_iiwa/meshes/link_%d.stl" % i for i in range(8)]
XARM = ["xarm/xarm_description/meshes/xarm6/collision/%s_vhacd.obj" % n for n in
        ("base", "link1", "link2", "link3", "link4", "link5", "link6")]

# name, repository prefix, model file, other files it needs, the question asked of it
ZOO = [
    ("so101 arm", MENAGERIE, "robotstudio_so101/so101.xml", [],
     "Can the SO-101's servos hold its arm out with a 200 g load?"),
    ("dynamixel 2r", MENAGERIE, "dynamixel_2r/dynamixel_2r.xml", [],
     "What torque does each Dynamixel need to hold the arm in any pose?"),
    ("unitree go1", MENAGERIE, "unitree_go1/go1.xml", [], "What does each foot carry when the Go1 stands?"),
    ("ur5e", MENAGERIE, "universal_robots_ur5e/ur5e.xml", [],
     "Can the UR5e hold a 5 kg payload anywhere in its range?"),
    ("franka panda", MENAGERIE, "franka_emika_panda/panda.xml", [], "How far can the Panda reach?"),
    ("car", MUJOCO, "car/car.xml", [], "What does each wheel of the car carry on the floor?"),
    ("slider crank", MUJOCO, "slider_crank/slider_crank.xml", [],
     "How far does the slider travel, and do any parts collide?"),
    ("tendon arm", MUJOCO, "tendon_arm/arm26.xml", [],
     "What torque must the shoulder and elbow give to hold the arm out?"),
    ("acrobot", DMC, "acrobot.xml", DMC_COMMON, "Can the elbow hold the acrobot out level?"),
    ("finger", DMC, "finger.xml", DMC_COMMON, "Does the finger hit anything as it moves through its range?"),
    ("manipulator", DMC, "manipulator.xml", DMC_COMMON, "Can the manipulator's motors hold the arm out straight?"),
    ("walker", DMC, "walker.xml", DMC_COMMON, "Will the walker stand up or tip over?"),
    ("quadruped", DMC, "quadruped.xml", DMC_COMMON, "What force does each foot carry when the quadruped stands?"),
    ("ant", GYM, "ant.xml", [], "Will the ant stand on its legs, and what does each foot carry?"),
    ("humanoid", GYM, "humanoid.xml", [], "Can the humanoid's motors hold it standing?"),
    ("pusher", GYM, "pusher.xml", [], "What is the pusher arm's reach, and can its motors hold it?"),
    ("double pendulum", GYM, "inverted_double_pendulum.xml", [], "Does the pendulum fall over if nothing holds it?"),
    ("half cheetah", GYM, "half_cheetah.xml", [], "Does the cheetah tip over standing?"),
    ("kuka iiwa", BULLET, "kuka_iiwa/model.urdf", KUKA, "Can the iiwa hold a 5 kg payload at full reach?"),
    ("xarm6", BULLET, "xarm/xarm6_robot.urdf", XARM,
     "Does the xArm collide with itself anywhere in its joint range?"),
    ("r2d2", BULLET, "r2d2.urdf", ["l_finger.stl", "l_finger_tip.stl"], "Will R2-D2 tip over standing on its wheels?"),
    ("husky", BULLET, "husky/husky.urdf", [], "What does each of the Husky's wheels carry?"),
    ("two joint robot", BULLET, "TwoJointRobot_w_fixedJoints.urdf", [], "Can the joints hold the arm level?"),
    ("cartpole urdf", BULLET, "cartpole.urdf", [], "How far can the cart slide, and how much force holds the pole?"),
    ("flexible tutorial", TUTORIAL, "06-flexible.urdf", [],
     "How far does the gripper reach, and does anything collide?"),
    ("physics tutorial", TUTORIAL, "07-physics.urdf", [], "Can the gripper joints hold the gripper out?"),
    ("drake acrobot", DRAKE, "acrobot/Acrobot.urdf", [],
     "What torque does the elbow need to hold the acrobot horizontal?"),
]

needs_mujoco = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed")
asked_for = pytest.mark.skipif(not os.environ.get("WOBBL_ZOO"), reason="set WOBBL_ZOO=1 to fetch and run the "
                                                                     f"{len(ZOO)} public models (about ten minutes)")


def fetch(entry, root: Path) -> Path:
    """Download one model and the files it needs into its own folder, laid out as its repository has them."""
    from wobbl.setup.installer import download

    name, repo, main, extra, _ = entry
    folder = root / name.replace(" ", "_")
    base = main.rsplit("/", 1)[0] + "/" if "/" in main else ""
    for rel in [main] + [f if f.startswith(base) or not base else base + f for f in extra]:
        dest = folder / rel
        if not dest.exists():
            download(f"https://raw.githubusercontent.com/{repo}{rel}", dest, log=lambda m: None)
    return folder / main


@pytest.fixture(scope="session")
def zoo(tmp_path_factory):
    root = tmp_path_factory.mktemp("zoo")
    try:
        fetch(ZOO[0], root)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(root, ignore_errors=True)
        pytest.skip(f"no network to fetch the models ({e})")
    yield root
    shutil.rmtree(root, ignore_errors=True)


@needs_mujoco
@asked_for
@pytest.mark.slow
@pytest.mark.parametrize("entry", ZOO, ids=[z[0] for z in ZOO])
def test_a_real_model_runs_its_question_and_passes_its_checks(entry, zoo):
    from wobbl.mechanism.study import run_mechanism

    path = fetch(entry, zoo)
    out = zoo / "out" / entry[0].replace(" ", "_")
    o = run_mechanism(str(path), {}, out, entry[4])
    failed = [f"{c.name}: {c.summary}" for c in o.checks if c.status == "fail"]
    assert (out / "outcome.json").exists() and (out / "playback.json").exists()
    assert o.values.get("analyses"), "no analysis ran"
    assert not failed, failed
