"""Field studies: closed forms, problem files, template values and GetDP's on demand install, then a few solves
that skip when GetDP is not installed."""
from __future__ import annotations

import hashlib
import io
import math
import tarfile

import pytest

from wobbl.fields import antenna, board, electric, forms, magnetic, sound
from wobbl.fields.params import ParamError, value
from wobbl.materials import get_dielectric, get_magnetic


def _have_getdp() -> bool:
    from wobbl.solvers import base

    try:
        return base.get("getdp").locate() is not None
    except Exception:  # noqa: BLE001
        return False


needs_getdp = pytest.mark.skipif(not _have_getdp(), reason="GetDP not installed (wobbl setup --only getdp)")


# --- materials and values ------------------------------------------------------------
def test_magnetic_materials_and_knees():
    steel = get_magnetic("1018")
    assert steel.key == "1010-steel" and 1.4 < steel.knee() < 1.6
    assert get_magnetic("armco").knee() > 1.0
    assert get_magnetic("n42").br == pytest.approx(1.30)
    assert get_magnetic("copper").bh is None
    with pytest.raises(KeyError):
        get_magnetic("unobtainium")


def test_dielectrics():
    assert get_dielectric("fr4").epsr == pytest.approx(4.4)
    assert get_dielectric("Rogers").key == "ro4350b"
    assert get_dielectric("er=3.5").epsr == 3.5


def test_bh_tables_are_monotone_and_close_to_air_past_the_curve():
    s, nu, w = forms.bh_tables(*get_magnetic("1010-steel").bh)
    assert all(b > a for a, b in zip(s, s[1:]))
    assert all(b >= a for a, b in zip(w, w[1:]))
    # stored energy density at 1 T from the table matches the trapezoid of H dB
    b, h = get_magnetic("1010-steel").bh
    import numpy as np
    bs = np.linspace(0, 1, 2001)
    exact = np.trapezoid(np.interp(bs, b, h), bs)
    assert np.interp(1.0, np.sqrt(s), w) == pytest.approx(exact, rel=0.02)


def test_values_take_units():
    assert value({"x": "6 mm"}, "x", 0, "length") == pytest.approx(0.006)
    assert value({"x": "6"}, "x", 0, "length") == pytest.approx(0.006)
    assert value({"x": "30"}, "x", 0, "distance") == pytest.approx(30.0)
    assert value({"x": "2.4 GHz"}, "x", 0, "frequency") == pytest.approx(2.4e9)
    assert value({"x": "0.5ns"}, "x", 0, "fast_time") == pytest.approx(0.5e-9)
    assert value({"x": "100 mW"}, "x", 0, "dbm") == pytest.approx(20.0)
    assert value({"x": "-90dBm"}, "x", 0, "dbm") == pytest.approx(-90.0)
    with pytest.raises(ParamError):
        value({"x": "3 parsecs"}, "x", 0, "length")


# --- closed forms --------------------------------------------------------------------
def test_microstrip_and_stripline_closed_forms():
    z, e = board.hammerstad_jensen(1.9e-3, 1e-3, 0.0, 4.4)
    assert 48 < z < 52 and 3.2 < e < 3.5
    # Wheeler's thick strip tends to Cohn's exact zero thickness value within its stated accuracy
    assert board.wheeler_stripline(0.15e-3, 0.4e-3, 1e-9, 4.4) == pytest.approx(
        board.cohn_stripline(0.15e-3, 0.4e-3, 4.4), rel=0.01)
    assert board.coax_z(0.455e-3, 1.475e-3, 2.25) == pytest.approx(60 / 1.5 * math.log(1.475 / 0.455), rel=1e-3)
    z, e = board.gcpw(0.3e-3, 0.15e-3, 0.2e-3, 4.4)
    assert 45 < z < 60 and 2.5 < e < 3.5


def test_induced_emf_half_wave_dipole():
    lam = forms.C0 / 1e9
    z = antenna.induced_emf(lam / 2, lam / 2000, 1e9)
    assert z.real == pytest.approx(73.1, rel=0.005) and z.imag == pytest.approx(42.5, rel=0.01)


def test_link_budget_matches_friis_in_free_space():
    lk = antenna.link_budget({"frequency": "2.4 GHz", "tx_power": "0 dBm", "sensitivity": "-80 dBm",
                              "tx_gain": "0", "rx_gain": "0", "environment": "free"})
    lam = forms.C0 / 2.4e9
    assert lk["values"]["range_median_m"] == pytest.approx(lam / (4 * math.pi) * 10 ** (80 / 20), rel=1e-3)
    office = antenna.link_budget({"frequency": "2.4 GHz", "tx_power": "0 dBm", "sensitivity": "-80 dBm",
                                  "environment": "office"})["values"]
    assert office["range_reliable_m"] < office["range_median_m"] < office["range_free_space_m"]


def test_sound_and_electric_closed_forms():
    modes = sound.box_modes((5.0, 4.0, 3.0), 60.0)
    assert modes[0][0] == pytest.approx(343.2 / 10, rel=1e-6) and modes[0][1] == (1, 0, 0)
    assert sound.muffler_tl(9.0, 1.0, math.pi / 2) == pytest.approx(20 * math.log10((9 + 1 / 9) / 2), rel=1e-9)
    # a piston much larger than the wavelength radiates rho c S v^2 / 2
    p = sound.piston_power(1.0, 200.0, 1.0)
    assert p == pytest.approx(0.5 * sound.RHO_AIR * sound.C_AIR * math.pi, rel=0.01)
    # a sphere far from the plane is an isolated sphere, 4 pi eps0 R
    c, _ = electric.sphere_images(0.01, 10.0, 1000.0)
    assert c == pytest.approx(4 * math.pi * forms.EPS0 * 0.01, rel=1e-3)
    # discs very close together approach the parallel plate value
    assert electric.kirchhoff(0.1, 1e-5, 1e-6) == pytest.approx(forms.EPS0 * math.pi * 0.01 / 1e-5, rel=0.01)
    assert magnetic.magnet_axis_field(1.0, 0.01, 1.0, 0.0) == pytest.approx(0.5, rel=1e-3)


# --- problem files and templates -----------------------------------------------------
def test_problem_files_name_their_regions_and_outputs():
    tags = {"air": 1, "shell": 2, "coil": 3, "core": 4, "plunger": 5, "axis": 101, "outer": 102}
    p = forms.MagProblem("axi", [forms.MagRegion("air"), forms.MagRegion("coil", js=1e6),
                                 forms.MagRegion("core", bh=get_magnetic("1010-steel").bh),
                                 forms.MagRegion("plunger", bh=get_magnetic("1010-steel").bh)],
                         ["axis", "outer"], shell=(0.1, 0.12, 0, 0), coils=[forms.Coil("coil", 100, 1e-4)],
                         body=["plunger"])
    pro = forms.magnetostatic(p, tags)
    for s in ("VolAxiSquSphShell", "GroupsOfNodesOf[Body]", "dhdb[{d a}]", 'File "F.txt"', "Psi_R_coil"):
        assert s in pro
    e = forms.electrostatic(forms.ElecProblem("planar", {"air": 2.2}, {"inner": 1.0, "outer": 0.0}),
                            {"air": 1, "inner": 101, "outer": 102})
    assert "EdgesOf[All, Not Cond]" in e and 'File "Q_inner.txt"' in e
    h = forms.helmholtz(forms.SoundProblem("3d", modes=10, shift=1.0), {"air": 1})
    assert "EigenSolve[A, 10" in h and "DtDtDof" in h


def test_templates_refuse_bad_values():
    from wobbl.fields.tools import check_params

    check_params("magnetic", "solenoid", {"turns": "800", "stroke": "6mm"})
    check_params("board", "coupled", {"width": "0.2mm", "spacing": "0.15mm", "rise": "0.5ns"})
    check_params("sound", "room", {"size": "5x4x2.7"})
    with pytest.raises(ParamError):
        check_params("magnetic", "solenoid", {"stroke": "40mm"})      # pulls the plunger out of the coil
    with pytest.raises(ParamError):
        check_params("electric", "plates", {"colour": "red"})
    with pytest.raises(ValueError):
        check_params("sound", "cathedral", {})


def test_field_analyses_are_registered_and_old_commands_still_run():
    from click.testing import CliRunner

    from wobbl.cli.main import cli
    from wobbl.simulate import registry

    names = {a.key for a in registry.analyses()}
    assert {"part.magnetic", "part.electric", "part.acoustic", "board.parasitics", "field.magnetic", "field.electric",
            "field.trace", "field.antenna", "field.acoustic"} <= names
    for c in ("magnetic", "electric", "board", "antenna", "sound"):
        assert c in cli.commands        # hidden aliases of simulate, still callable
    r = CliRunner().invoke(cli, ["magnetic", "solenoid", "--set", "colour=red"])
    assert r.exit_code != 0 and "colour" in r.output


# --- on demand install ---------------------------------------------------------------
def _fake_archive(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in (("getdp-3.5.0-Linux64/bin/getdp", b"#!/bin/sh\necho 3.5.0\n"),
                           ("getdp-3.5.0-Linux64/share/doc/getdp/LICENSE.txt", b"GPL"),
                           ("getdp-3.5.0-Linux64/share/doc/getdp/examples/magnet.msh", b"x" * 5000)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    path = tmp_path / "getdp.tgz"
    path.write_bytes(buf.getvalue())
    return path


def test_getdp_install_keeps_only_the_binary(tmp_path, monkeypatch):
    from wobbl.setup import installer
    from wobbl.solvers import getdp

    arc = _fake_archive(tmp_path)
    sha = hashlib.sha256(arc.read_bytes()).hexdigest()
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(getdp, "platform_tag", lambda: "linux-64")
    monkeypatch.setitem(getdp.BUILDS, "linux-64", ("x/getdp.tgz", sha, "getdp-3.5.0-Linux64/bin/getdp"))
    monkeypatch.setattr(installer, "download", lambda url, dest, log=print: dest.write_bytes(arc.read_bytes()) and dest)
    out = getdp.install(lambda m: None)
    home = tmp_path / "home" / "getdp"
    kept = sorted(p.relative_to(home).as_posix() for p in home.rglob("*") if p.is_file())
    assert kept == ["LICENSE.txt", "bin/getdp"]
    assert out["unpacked_bytes"] > out["installed_bytes"]
    assert not (home / ".download").exists()
    monkeypatch.setitem(getdp.BUILDS, "linux-64", ("x/getdp.tgz", "0" * 64, "getdp-3.5.0-Linux64/bin/getdp"))
    with pytest.raises(installer.SetupError):
        getdp.install(lambda m: None, force=True)
    assert not (home / ".download").exists()


def test_getdp_is_a_registered_on_demand_recipe():
    from wobbl.setup import installer

    r = installer.RECIPES["getdp"]
    assert not r.core and "magnetic" in r.domains and r.installer is not None
    assert [x.name for x in installer.recipes_for("sound")] == ["getdp"]


# --- solves ------------------------------------------------------------------------------
@pytest.mark.solver
@needs_getdp
def test_coax_capacitance_matches_the_exact_value(tmp_path):
    o = electric.run_electric("coax", {"inner_d": "1 mm", "outer_d": "4 mm", "voltage": "1000 V",
                                        "dielectric": "ptfe"}, tmp_path / "coax")
    exact = electric.coax_capacitance(0.5e-3, 2e-3, 0.0, 2.1)
    assert o.values["capacitance_F"] == pytest.approx(exact, rel=0.005)
    assert o.confidence in ("high", "medium") and not (tmp_path / "coax" / "work").exists()


@pytest.mark.solver
@needs_getdp
def test_microstrip_matches_hammerstad_jensen(tmp_path):
    o = board.run_board("microstrip", {"width": "1 mm", "height": "0.5 mm", "thickness": "0.005 mm", "er": "4.4"},
                        tmp_path / "ms")
    assert o.values["z0_ohm"] == pytest.approx(board.hammerstad_jensen(1e-3, 0.5e-3, 5e-6, 4.4)[0], rel=0.01)
    assert {c.name: c.status for c in o.checks}["closed form"] == "pass"


@pytest.mark.solver
@needs_getdp
def test_air_core_solenoid_field(tmp_path):
    o = magnetic.run_magnetic("solenoid", {"core": "none", "coil_id": "20 mm", "coil_od": "21 mm", "length": "410 mm",
                                           "stroke": "1 mm"}, tmp_path / "sol")
    assert o.values["b_center_T"] == pytest.approx(o.values["b_exact_T"], rel=0.01)


@pytest.mark.solver
@needs_getdp
def test_enclosure_modes_match_the_box(tmp_path):
    o = sound.run_room("enclosure", {"size": "300x200x150", "fmax": "1300 Hz"}, tmp_path / "box")
    exact = [f for f, _ in sound.box_modes((0.3, 0.2, 0.15), 1300.0)]
    assert o.values["modes_Hz"][0] == pytest.approx(exact[0], rel=0.005)
    assert {c.name: c.status for c in o.checks}["closed form"] == "pass"
