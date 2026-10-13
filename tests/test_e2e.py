"""End to end runs with the real solvers (installed by 'wobbl setup')."""
import json

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import needs_ccx, needs_spice
from wobbl.cli.main import cli

pytestmark = pytest.mark.solver


@needs_ccx
def test_cantilever_matches_beam_theory(parts, tmp_path):
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(parts["cantilever"], "mild-steel", [parse_load("100 N -z @ x=max")],
                         [parse_fixture("x=min")], quality="fast", budget=120)
    o = analyze_strength(st, tmp_path / "out", render=False)
    E, I, L, F = 200000.0, 20 * 10 ** 3 / 12, 200.0, 100.0
    tip = F * L ** 3 / (3 * E * I)
    assert o.max_disp_mm == pytest.approx(tip, rel=0.05)
    beam = next(c for c in o.checks if c.name == "hand calculation")
    assert beam.status == "pass"
    bal = next(c for c in o.checks if c.name == "force balance")
    assert bal.status == "pass"
    assert o.verdict == "holds"
    assert (tmp_path / "out" / "report.html").exists()


@needs_ccx
def test_bolted_plate_cli_json(parts, tmp_path):
    r = CliRunner().invoke(cli, ["run", parts["mount"], "-m", "pla", "--fix", "bolt holes d=4.5",
                                 "--load", "50 N down @ top and within 8 of 30,20,5", "--quality", "fast",
                                 "--no-images", "--json", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output[r.output.index("{"):])
    assert d["verdict"] in ("holds", "marginal", "fails")
    assert d["confidence"] in ("high", "medium", "low")
    assert any(c["name"] == "bolt bearing" for c in d["checks"])


@needs_ccx
def test_bearing_load_from_the_cli(parts, tmp_path):
    r = CliRunner().invoke(cli, ["run", parts["plate_hole"], "-m", "6061-t6", "--fix", "clamp x=max",
                                 "--load", "300 N -x bearing @ hole d=8", "--quality", "fast", "--no-images",
                                 "--json", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output[r.output.index("{"):])
    assert any("cosine distribution" in a for a in d["assumptions"])
    bal = next(c for c in d["checks"] if c["name"] == "force balance")
    assert bal["status"] == "pass"


@needs_ccx
def test_material_swap_says_whether_it_scaled_or_solved_again(parts, tmp_path):
    from wobbl.fix.apply import SCALED, apply_fixes
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.report import write_outputs
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(parts["cantilever"], "6061-t6", [parse_load("100 N -z @ x=max")], [parse_fixture("x=min")],
                         quality="fast", budget=120)
    out = tmp_path / "out"
    o = analyze_strength(st, out, render=False, fixes=False)
    swap = {"kind": "material", "summary": "", "auto": True}
    o.fixes = [{**swap, "title": "steel", "params": {"material": "mild-steel"}},
               {**swap, "title": "low Poisson", "params": {"material": "6061-t6:nu=0.2"}}]
    write_outputs(o, out)
    res = apply_fixes(out, ["material"])
    steel, low_nu = res["after"]
    assert steel["method"] == SCALED
    assert steel["governing_MPa"] == pytest.approx(o.governing_MPa, rel=1e-6)
    assert steel["max_disp_mm"] == pytest.approx(o.max_disp_mm * 68900 / 200000, rel=1e-6)
    assert low_nu["method"].startswith("solved again, because its Poisson's ratio of 0.2")
    assert low_nu["stress_image"] and (out / "fix-material" / "outcome.json").exists()
    # same stiffness and a beam in bending: the new solve lands close to the old one, but it is a new solve
    assert low_nu["governing_MPa"] == pytest.approx(o.governing_MPa, rel=0.1)
    assert low_nu["max_disp_mm"] == pytest.approx(o.max_disp_mm, rel=0.05)


@needs_ccx
def test_unheld_part_is_refused(parts, tmp_path):
    r = CliRunner().invoke(cli, ["run", parts["cantilever"], "-m", "pla", "--fix", "roller z=min",
                                 "--load", "10 N down @ top", "--quality", "fast", "-o", str(tmp_path / "x")])
    assert r.exit_code != 0
    assert "free to" in r.output


@needs_ccx
def test_thermal_block_energy_balance(parts, tmp_path):
    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from

    st = study_from(parts["cantilever"], "6061-t6", power="5 W", films=["10 W/m2K @ all"], ambient="25 C",
                    quality="fast")
    o = run_thermal(st, tmp_path / "heat")
    area = 2 * (200 * 20 + 200 * 10 + 20 * 10) * 1e-6
    expect = 25 + 5 / (10 * area)
    assert o.values["t_max_C"] == pytest.approx(expect, rel=0.02)  # aluminum: nearly isothermal
    assert next(c for c in o.checks if c.name == "energy balance").status == "pass"


def _balance(o):
    return next(c for c in o.checks if c.name == "energy balance")


@needs_ccx
def test_transient_energy_balance_on_the_battery_pack(tmp_path):
    """The battery pack example exactly as its run.sh runs it: 6 W for an hour has to end up stored in the pack or
    shed by convection and radiation."""
    from pathlib import Path

    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from

    pack = Path(__file__).resolve().parents[1] / "examples" / "battery-pack" / "pack.py"
    st = study_from(str(pack), "li-ion-cell", "6 W", "", ["5 W/m2K @ all"], "35 C", [], "60 min", "60 C", 0.9)
    o = run_thermal(st, tmp_path / "pack")
    bal = _balance(o)
    assert bal.status == "pass", bal.summary
    d = bal.detail
    assert d["in_J"] == pytest.approx(6 * 3600) and d["held_J"] == 0
    assert d["stored_J"] > 0 and d["out_J"] > 0 and d["error"] < 0.02
    assert d["stored_J"] + d["out_J"] == pytest.approx(d["in_J"], rel=0.02)


@needs_ccx
def test_transient_energy_balance_on_the_lumped_validation_case(tmp_path):
    """The lumped cooling case makes no heat, so the heat the cube loses has to match what its surfaces shed."""
    from wobbl.outcome_generic import GenericOutcome
    from wobbl.validation.cases import run_cases

    row = run_cases(tmp_path, ["lumped"], log=lambda m: None)[0]
    assert row["pass"], row
    bal = _balance(GenericOutcome.load(tmp_path / "lumped" / "run" / "outcome.json"))
    assert bal.status == "pass", bal.summary
    d = bal.detail
    assert d["in_J"] == 0 and d["stored_J"] < 0
    assert -d["stored_J"] == pytest.approx(d["out_J"], rel=0.02)


@needs_ccx
def test_transient_energy_balance_with_a_held_face(tmp_path):
    """A plate held at the air temperature on its bottom face passes nearly all its heat out through that face."""
    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "plate.step", 60, 40, 5)
    st = study_from(str(part), "6061-t6", power="5 W", films=["10 W/m2K @ top"], fixed=["25 C @ bottom"],
                    ambient="25 C", duration="5 min")
    bal = _balance(run_thermal(st, tmp_path / "heat"))
    assert bal.status == "pass", bal.summary
    assert bal.detail["held_J"] == pytest.approx(-5 * 300, rel=0.01)
    assert "through held temperature faces" in bal.summary


def _aluminum_diffusivity():
    from wobbl.materials import get_material

    m = get_material("6061-t6")
    return m.conductivity / (m.density * m.specific_heat) * 1e6     # mm2/s


def _at(fields, t):
    i = int(np.argmin(np.abs(fields["hist_t"] - t)))
    assert fields["hist_t"][i] == pytest.approx(t, rel=1e-4)
    return fields["hist_T"][i]


@needs_ccx
def test_sudden_face_temperature_follows_the_erfc_solution(tmp_path):
    """A part at 20 C set on a surface held at 120 C, the case that used to stop CalculiX. A 60 mm aluminum bar
    whose end jumps to 120 C is a semi infinite solid for the first 2 s, so every node follows
    T = 20 + 100 erfc(x / (2 sqrt(alpha t))) within 1.5 C, and the heat its end lets in is the heat it stores."""
    from scipy.special import erf

    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "bar.step", 60, 10, 10)
    st = study_from(str(part), "6061-t6", films=["10 W/m2K @ x=max"], fixed=["120 C @ x=min"], ambient="20 C",
                    duration="2 s")
    o = run_thermal(st, tmp_path / "heat")
    f = np.load(tmp_path / "heat" / "fields.npz")
    alpha, x = _aluminum_diffusivity(), f["X"][:, 0]
    for t in (0.25, 0.5, 1.0, 2.0):
        exact = 120 - 100 * erf(x / (2 * np.sqrt(alpha * t)))
        assert np.abs(_at(f, t) - exact).max() < 1.5, t
    bal = _balance(o)
    assert bal.status == "pass", bal.summary
    assert bal.detail["held_J"] == pytest.approx(bal.detail["stored_J"], rel=0.02) and bal.detail["held_J"] > 0
    assert any("ramps from 20 C to 120 C in a straight line" in a for a in o.assumptions)


@needs_ccx
def test_two_held_faces_and_a_third_starting_temperature(tmp_path):
    """A 50 mm insulated aluminum bar at 40 C whose ends jump to 100 C and 0 C. It follows the Fourier series for a
    slab between two held temperatures within 1.5 C, the hot end lets heat in while the cold end lets it out, and
    the two together match what the bar stores."""
    from wobbl.thermal.study import run_thermal
    from wobbl.thermal.tools import study_from
    from wobbl.validation import geometry as G

    L = 50.0
    part = G.box(tmp_path / "bar.step", L, 10, 10)
    st = study_from(str(part), "6061-t6", films=["0 W/m2K @ all"], fixed=["100 C @ x=min", "0 C @ x=max"],
                    ambient="20 C", duration="4 s")
    st.initial = 40.0
    o = run_thermal(st, tmp_path / "heat")
    f = np.load(tmp_path / "heat" / "fields.npz")
    alpha, x = _aluminum_diffusivity(), f["X"][:, 0]
    n = np.arange(1, 400)
    b = 2 / (n * np.pi) * ((40 - 100) * (1 - (-1.0) ** n) + (0 - 100) * (-1.0) ** n)
    for t in (0.5, 1.0, 2.0, 4.0):
        decay = b * np.exp(-alpha * (n * np.pi / L) ** 2 * t)
        exact = 100 - 100 * x / L + np.sin(np.outer(x, n) * np.pi / L) @ decay
        assert np.abs(_at(f, t) - exact).max() < 1.5, t
    bal = _balance(o)
    assert bal.status == "pass", bal.summary
    hot, cold = bal.detail["held_faces_J"]
    assert hot > 0 > cold and "in and" in bal.summary
    assert hot + cold == pytest.approx(bal.detail["stored_J"], rel=0.02)
    assert any("x=min (100 C) and x=max (0 C) ramp to their targets" in a for a in o.assumptions)


@needs_spice
def test_led_and_rc(tmp_path):
    from wobbl.circuits.run import run_circuit

    led = run_circuit("led", {"supply": "5", "resistor": "150", "led": "red"}, tmp_path / "led")
    assert led.verdict == "ok"
    assert led.values["i_led_A"] == pytest.approx(0.0198, rel=0.05)
    assert all(c.status in ("pass", "info") for c in led.checks)
    rc = run_circuit("rc", {"r": "1k", "c": "1u", "v": "5"}, tmp_path / "rc")
    assert rc.values["v_at_tau"] == pytest.approx(5 * 0.63212, rel=0.005)


@needs_ccx
@needs_spice
def test_validation_subset(tmp_path):
    """A quick slice of the validation suite: exact answers and high confidence, including roller supports."""
    from wobbl.validation.cases import run_cases

    rows = run_cases(tmp_path, ["cantilever-tip", "lame", "slab", "led", "rc"], log=lambda m: None)
    for r in rows:
        assert r["pass"], r
        assert r["confidence"] == "high", r


@needs_ccx
def test_remote_load_reports_attachment_spread(parts, tmp_path):
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    st = StructuralStudy(parts["mount"], "6061-t6", [parse_load("200 N -z at 30,20,40 @ holes d=4.5")],
                         [parse_fixture("bottom")], quality="fast", budget=120)
    o = analyze_strength(st, tmp_path / "out", render=False, fixes=False)
    lp = next(c for c in o.checks if c.name == "load introduction")
    assert lp.status in ("pass", "info", "warn") and "rigid attachment" in lp.summary
    d = lp.detail
    assert d["alternative_levels"] and d["alternative_peak"] == pytest.approx(d["alternative_levels"][-1]["governing"])
    assert d["refined"] == bool(d["refine_reason"])


@needs_ccx
def test_load_attachment_variant_converges_at_its_own_peak(parts, tmp_path, monkeypatch):
    """When the other attachment needs it, the check refines that variant around its own peak and reports
    the same convergence evidence as the main study."""
    from wobbl.structural import run as run_mod
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    monkeypatch.setattr(run_mod, "variant_needs_refinement", lambda *a: "forced by the test")
    st = StructuralStudy(parts["mount"], "6061-t6", [parse_load("200 N -z at 30,20,40 @ holes d=4.5")],
                         [parse_fixture("bottom")], quality="fast", budget=300)
    o = analyze_strength(st, tmp_path / "out", render=False, fixes=False)
    c = next(c for c in o.checks if c.name == "load introduction")
    d = c.detail
    assert d["refined"] and d["refine_reason"] == "forced by the test"
    lv = d["alternative_levels"]
    assert len(lv) >= 2 and lv[1]["n_nodes"] > lv[0]["n_nodes"]
    assert d["alternative_peak"] == pytest.approx(lv[-1]["governing"])
    if d["converged"]:
        assert f"over {len(lv)} meshes refined around that peak" in c.summary
    else:
        assert "did not converge" in c.summary and "not a result" in c.summary


@needs_ccx
def test_the_same_study_twice_gives_identical_answers(parts, tmp_path):
    """Same input on the same machine: every number matches exactly, from the wall thickness through the meshes
    to the checks and the other load attachment's refinement."""
    from wobbl.structural.pipeline import analyze_strength
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    def once(k):
        st = StructuralStudy(parts["mount"], "6061-t6", [parse_load("200 N -z at 30,20,40 @ holes d=4.5 rigid")],
                             [parse_fixture("bottom")], quality="fast", budget=120)
        o = analyze_strength(st, tmp_path / f"run{k}", render=False, fixes=False)
        levels = [{k: v for k, v in L.items() if k not in ("t_mesh", "t_solve")} for L in o.levels]
        return (o.governing_MPa, o.governing_xyz, o.max_disp_mm, levels, o.hotspots, o.features,
                [(c.name, c.status, c.summary) for c in o.checks], o.confidence_reasons)

    first, second = once(1), once(2)
    assert first == second
    assert first[3][0]["n_nodes"] > 0 and any(c[0] == "load introduction" for c in first[6])
