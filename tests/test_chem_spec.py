"""Motor designs from TOML and from flags, and the thrust curve handed to the motion domain."""
from pathlib import Path

import pytest

from wobbl.chem.curves import motion_motor
from wobbl.chem.spec import SpecError, parse_casing, parse_grain, read_spec, spec_from_flags

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "knsb-motor" / "motor.toml"

DESIGN = """
name = "test"
propellant = "knsb"
[[grain]]
shape = "bates"
diameter = "48 mm"
core = "20 mm"
length = "75 mm"
count = 4
[nozzle]
throat = "12 mm"
exit = "30 mm"
[casing]
material = "6061-t6"
outer_diameter = "54 mm"
wall = "1.6 mm"
length = "320 mm"
"""


def test_flags_build_the_same_motor_as_the_spec():
    d = spec_from_flags("knsb", ["bates d=48 core=16 L=75 x5"], "12.7", "32", "6061-t6 od=54 wall=1.6 L=400 sf=2",
                        efficiency=0.85)
    a, b = read_spec(d).motor, read_spec(EXAMPLE).motor
    assert len(a.grains) == len(b.grains) == 5
    assert a.kn(0.0) == pytest.approx(b.kn(0.0)) and a.nozzle.area_ratio == pytest.approx(b.nozzle.area_ratio)
    assert parse_grain("finocyl d=98 core=25 fins=6 fin_length=12 fin_width=5 x2")["count"] == 2
    assert parse_casing("ss304 od=38 id=34")["inner_diameter"] == "34 mm"
    custom = read_spec(DESIGN.replace('propellant = "knsb"', '[propellant]\nname = "my mix"\ndensity = "1700 kg/m3"\n'
                                      'a = 5.0\nn = 0.35\nT = 1600\nM = 40\nk = 1.14'))
    assert custom.motor.propellant.burn_rate.rate(1e6) == pytest.approx(0.005)


@pytest.mark.parametrize("bad,match", [
    (DESIGN.replace('core = "20 mm"', 'core = "20 mm"\ncolour = "red"'), "unknown key"),
    (DESIGN.replace('diameter = "48 mm"', 'diameter = "52 mm"'), "does not fit"),
    (DESIGN.replace('propellant = "knsb"', 'propellant = "unobtainium"'), "unknown propellant"),
    (DESIGN.replace('throat = "12 mm"', 'throat = "40 mm"'), "exit at least as wide"),
    (DESIGN.replace('length = "320 mm"', 'length = "200 mm"'), "longer than the casing"),
])
def test_bad_specs_say_what_is_wrong(bad, match):
    with pytest.raises(SpecError, match=match):
        read_spec(bad)


def test_the_motion_domain_takes_a_simulated_motor(tmp_path):
    pytest.importorskip("wobbl.motion.spec")
    from wobbl.motion.spec import read_spec as read_motion

    mo = motion_motor(EXAMPLE)
    assert mo.propellant == pytest.approx(1.0556, rel=0.01) and mo.loaded_mass > mo.propellant
    assert 1100 < mo.impulse < 1400 and mo.length == pytest.approx(0.4)
    assert mo.curve[-1][1] == 0.0
    rocket = (ROOT / "examples" / "rocket-launch" / "rocket.toml").read_text()
    start, end = rocket.index("[motor]"), rocket.index("[launch]")
    spec = rocket[:start] + f'[motor]\ndesign = "{EXAMPLE.as_posix()}"\nloaded_mass = "2.1 kg"\n\n' + rocket[end:]
    st = read_motion(spec)
    assert st.motor.impulse == pytest.approx(mo.impulse) and st.motor.loaded_mass == pytest.approx(2.1)
    with pytest.raises(Exception, match="loaded_mass"):
        read_motion(spec.replace('loaded_mass = "2.1 kg"', 'loaded_mass = "0.5 kg"'))
