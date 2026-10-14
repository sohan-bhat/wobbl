"""Loading model files the way people share them: package paths, missing and unreadable meshes, links with no
mass, ROS leftovers and MJCF files without their assets. Skipped when MuJoCo is not installed."""
import math
import struct

import numpy as np
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")


def _stl(path, half=0.02):
    """A closed box as a binary STL, half sizes ``half`` m."""
    h = half
    v = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(faces)))
        for a, b, c in faces:
            f.write(struct.pack("<3f", 0, 0, 0))
            for i in (a, b, c):
                f.write(struct.pack("<3f", *v[i]))
            f.write(b"\0\0")


ROBOT = """<?xml version="1.0"?>
<robot name="test_arm" xmlns:drake="http://drake.mit.edu">
  <link name="base"><collision><geometry><box size="0.2 0.2 0.05"/></geometry></collision>
    <inertial><mass value="2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="upper">
    <visual><geometry><mesh filename="package://my_robot/meshes/upper.stl"/></geometry></visual>
    <collision><geometry><mesh filename="package://my_robot/meshes/upper.stl"/></geometry></collision>
    <inertial><origin xyz="0.1 0 0"/><mass value="1"/>
      <inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="fore">
    <collision><geometry><mesh filename="package://my_robot/meshes/fore.dae"/></geometry></collision>
    <inertial><origin xyz="0.1 0 0"/><mass value="0.5"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="tool"><visual><origin xyz="0.02 0 0"/><geometry><sphere radius="0.03"/></geometry></visual></link>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="12" velocity="3"/></joint>
  <joint name="weld" type="fixed"><parent link="world"/><child link="base"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="fore"/>
    <origin xyz="0.2 0 0" rpy="$(optenv ELBOW_RPY 0 0 0)"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="5" velocity="4"/></joint>
  <joint name="wrist" type="revolute"><parent link="fore"/><child link="tool"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="4"/></joint>
  <transmission name="t"><joint name="elbow"/></transmission>
  <drake:collision_filter_group name="g"/>
</robot>
"""


def test_a_urdf_as_ros_shares_it_loads_with_what_was_assumed_written_down(tmp_path):
    from wobbl.mechanism.model import describe, load

    pkg = tmp_path / "ws" / "src" / "my_robot"
    _stl(pkg / "meshes" / "upper.stl")
    (pkg / "urdf").mkdir(parents=True)
    path = pkg / "urdf" / "arm.urdf"
    path.write_text(ROBOT)
    ld = load(path)
    m = ld.compile()
    info = describe(m)
    assert [j["name"] for j in info.joints] == ["shoulder", "elbow", "wrist"]
    assert ld.limits == {"shoulder": (12.0, 3.0), "elbow": (5.0, 4.0), "wrist": (1.0, 4.0)}
    text = " ".join(ld.notes)
    assert "fore.dae" in text and "cannot read" in text            # COLLADA is dropped, its mass kept
    assert "tool had only a visual shape" in text and "tool gave no mass" in text
    assert "base" not in text.split("gave no mass")[0].split(".")[-1]
    names = info.names["body"]
    assert m.body_mass[names.index("fore")] == pytest.approx(0.5)
    upper = names.index("upper")
    assert m.body_geomnum[upper] == 1 and m.geom_type[m.body_geomadr[upper]] == 7     # the found STL
    # the tool's mass came from its sphere at 1000 kg/m3
    assert m.body_mass[names.index("tool")] == pytest.approx(1000 * 4 / 3 * math.pi * 0.03 ** 3, rel=1e-3)
    assert not info.floating and info.leaves[0] == names.index("tool")


def test_an_mjcf_without_its_meshes_and_textures_still_runs_on_the_masses_it_states(tmp_path):
    from wobbl.mechanism.model import describe, load

    (tmp_path / "arm.xml").write_text("""<mujoco>
  <compiler meshdir="assets" texturedir="assets"/>
  <asset><mesh file="link.stl"/><mesh name="cover" file="cover.stl"/>
    <texture name="skin" type="2d" file="skin.png"/><material name="m" texture="skin"/></asset>
  <worldbody>
    <body name="link" pos="0 0 0.5"><joint name="j" type="hinge" axis="0 1 0"/>
      <inertial pos="0.1 0 0" mass="0.7" diaginertia="1e-3 1e-3 1e-3"/>
      <geom type="mesh" mesh="link" material="m"/><geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.01"
        mass="0"/>
      <body name="cover"><geom type="mesh" mesh="cover"/></body>
    </body>
  </worldbody>
</mujoco>""")
    _stl(tmp_path / "assets" / "cover.stl")
    ld = load(tmp_path / "arm.xml")
    m = ld.compile()
    assert m.nmesh == 1 and m.body_mass[1] == pytest.approx(0.7)
    assert any("link.stl" in n for n in ld.notes) and any("texture" in n for n in ld.notes)
    assert describe(m).joints[0]["name"] == "j"


def test_edits_add_a_payload_a_free_base_and_a_floor(tmp_path):
    from wobbl.mechanism.model import add_floor, add_payload, describe, end_effector, free_base, load, lowest_point

    pkg = tmp_path / "my_robot"
    _stl(pkg / "meshes" / "upper.stl")
    (pkg / "arm.urdf").write_text(ROBOT)
    ld = load(pkg / "arm.urdf")
    m0 = ld.compile()
    info = describe(m0)
    body, at = end_effector(m0, info)
    add_payload(ld, 2.0, info.names["body"][body], at)
    free_base(ld)
    m1 = ld.compile()
    add_floor(ld, lowest_point(m1))
    m = ld.compile()
    assert m.body_mass[1:].sum() == pytest.approx(m0.body_mass[1:].sum() + 2.0)
    assert describe(m).floating and describe(m).floor
    assert any("stands loose" in n for n in ld.notes)
    assert np.isclose(lowest_point(m), m.geom_pos[[g for g in range(m.ngeom) if m.geom_type[g] == 0][0], 2])
