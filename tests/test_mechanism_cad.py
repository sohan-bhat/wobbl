"""A mechanism assembled from a CAD file: each part a solid of one STEP assembly, its mass from its volume and
material, joined at points given in the assembly's own coordinates. Skipped when MuJoCo is not installed."""
import pytest

from wobbl.mechanism import install

pytestmark = pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed (wobbl setup --only mujoco)")
G = 9.81
PLA = 1240.0

SPEC = """name = "printed arm"
cad = "arm.step"

[[part]]
name = "base"
solid = 1
material = "pla"

[[part]]
name = "arm"
solid = 2
parent = "base"
material = "pla"
joint = { name = "hinge", axis = "y", at = [20, 20, 25], range = [-90, 90] }
"""


def _assembly(path):
    from conftest import _write

    def build(occ):
        occ.addBox(0, 0, 0, 40, 40, 20)            # the base
        occ.addBox(20, 15, 20, 200, 10, 10)        # an arm 200 x 10 x 10 mm, hinged 20 mm in from its end

    _write(path, build)


def test_a_step_assembly_runs_with_masses_from_its_solids(tmp_path):
    from wobbl.mechanism.model import describe, loaded
    from wobbl.mechanism.spec import read_spec
    from wobbl.mechanism.study import run_mechanism

    _assembly(tmp_path / "arm.step")
    (tmp_path / "arm.toml").write_text(SPEC)
    mech = read_spec(str(tmp_path / "arm.toml"))
    model = loaded(mech).compile()
    names = describe(model).names["body"]
    arm_mass = 200 * 10 * 10 * 1e-9 * PLA
    assert model.body_mass[names.index("arm")] == pytest.approx(arm_mass, rel=0.01)
    o = run_mechanism(str(tmp_path / "arm.toml"), {}, tmp_path / "out", "what torque does the hinge need to hold "
                                                                          "the arm out?")
    # level, the arm's middle sits 100 mm past the hinge
    assert abs(o.values["joints"]["hinge"]["needed"]) == pytest.approx(arm_mass * G * 0.1, rel=0.01)
    assert {c.name: c.status for c in o.checks}["hand statics"] == "pass"


def test_wobbl_simulate_reads_a_part_spec_as_a_mechanism(tmp_path):
    from wobbl.simulate import route

    (tmp_path / "arm.toml").write_text(SPEC)
    plan = route(str(tmp_path / "arm.toml"), "can the hinge hold the arm level?")
    assert plan.domain == "mechanism" and plan.kind == "spec" and plan.names == ["torque"]
