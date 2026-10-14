"""Elastic plastic studies: the true stress strain curves, without a solver."""
import math

import pytest
from conftest import needs_ccx

from wobbl.materials import LIBRARY, get_material, hardening_curve, parse_custom


def test_mild_steel_follows_the_yun_gardner_quad_linear_model():
    """A36 at fy 250 and fu 400: a yield plateau to 1.5% strain, then hardening to fu at 22.5% (engineering)."""
    h = hardening_curve(get_material("mild-steel"))
    assert h.kind == "multilinear" and "Yun and Gardner" in h.source
    (s0, p0), (s1, p1), _, (s3, p3) = h.points
    assert p0 == 0 and s0 == pytest.approx(250 * (1 + 250 / 200000))
    assert s1 == pytest.approx(250 * 1.015) and p1 == pytest.approx(math.log(1.015) - 250 * 1.015 / 200000, abs=1e-6)
    assert s3 == pytest.approx(400 * 1.225) and p3 == pytest.approx(math.log(1.225) - 490 / 200000, abs=1e-6)


def test_aluminum_knee_starts_below_the_proof_stress_and_ends_at_the_true_tensile_strength():
    m = get_material("6061-t6")
    h = hardening_curve(m)
    assert "Steinhardt" in h.source
    assert 0.9 * m.yield_strength < h.points[0][0] < m.yield_strength
    # the 0.2% proof stress sits on the knee
    near = min(h.points, key=lambda p: abs(p[1] - 0.002))
    assert near[0] == pytest.approx(m.yield_strength, rel=0.02)
    e_u = 0.5 * m.elongation / 100
    assert h.points[-1][0] == pytest.approx(m.tensile_strength * (1 + e_u))


@pytest.mark.parametrize("key", [k for k, m in LIBRARY.items() if m.category == "metal"])
def test_every_metal_curve_rises_from_zero_plastic_strain(key):
    h = hardening_curve(get_material(key))
    assert h.points[0][1] == 0 and h.source
    for (s0, p0), (s1, p1) in zip(h.points, h.points[1:]):
        assert p1 > p0 and s1 >= s0


def test_plastic_section_of_a_rectangle_is_sigma_y_b_h2_over_4(parts, tmp_path):
    """The 200 x 20 x 10 cantilever with 100 N down at the tip: at mid span M = 10 kN mm, Z = 20 x 10^2 / 4."""
    import numpy as np

    from wobbl.fe import NodeConstraint, StaticModel
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.plastic.section import plastic_section
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path)
    mesh = build_mesh(prep, Sizing(h=4.0))
    root = np.nonzero(np.abs(mesh.X[:, 0]) < 1e-6)[0]
    tip = np.nonzero(np.abs(mesh.X[:, 0] - 200) < 1e-6)[0]
    F = np.zeros((mesh.n_nodes, 3))
    F[tip, 2] = -100.0 / len(tip)
    sm = StaticModel(mesh, 200000.0, 0.3, 7.85e-9, [NodeConstraint("fix1", root)], forces=F)
    s, why = plastic_section(mesh, [100.0, 10.0, 5.0], sm, 250.0)
    assert s is not None, why
    assert s["Z_mm3"] == pytest.approx(500.0, rel=0.005)
    assert s["factor"] == pytest.approx(250 * 500 / 1e4, rel=0.005)
    assert s["depth_mm"] == pytest.approx(10.0, rel=0.01) and s["bending_share"] == pytest.approx(1.0)


def test_verdicts_separate_yielding_from_collapse():
    from wobbl.plastic.study import verdict_for

    assert verdict_for(1.4, 2.5, 1.5) == "holds"
    assert verdict_for(0.6, 2.0, 1.5) == "yields"      # yields locally but carries the load with margin
    assert verdict_for(0.6, 1.2, 1.5) == "marginal"
    assert verdict_for(0.6, 0.9, 1.5) == "collapses"


def test_brittle_materials_are_turned_away_and_ductile_plastics_carry_a_caveat():
    from wobbl.plastic.study import check_metal
    from wobbl.structural.spec import SpecError

    for name in ("pla", "wood"):
        with pytest.raises(SpecError):
            check_metal(get_material(name))
    assert check_metal(get_material("6061-t6")) == ""
    assert check_metal(parse_custom("custom:E=200000,nu=0.3,yield=250")) == ""
    assert "creeps" in check_metal(get_material("pc"))
    assert "layers" in check_metal(get_material("petg"))


def test_any_stress_strain_curve_is_read_as_true_stress_against_plastic_strain(tmp_path):
    from wobbl.plastic.study import curve_for

    m = parse_custom("custom:E=200000,nu=0.3,yield=250")
    assert curve_for(m, "true 250@0, 300@0.02, 420@0.15").points == ((250, 0), (300, 0.02), (420, 0.15))
    # a tensile test: engineering stress at total strain, the first point on the elastic line; the engineering drop
    # after the peak is still a rise in true stress
    h = curve_for(m, "engineering 250@0.00125, 300@0.02, 400@0.2, 380@0.3")
    assert h.points[0] == (pytest.approx(250 * 1.00125), 0.0) and h.points[3][0] == pytest.approx(494)
    assert h.points[2] == (pytest.approx(480), pytest.approx(math.log(1.2) - 480 / 200000))
    # true stress at total strain, from a file with a header
    (tmp_path / "c.csv").write_text("stress_MPa,strain\n250,0.00125\n330,0.0217\n")
    h = curve_for(m, str(tmp_path / "c.csv"))
    assert h.points == ((250, 0.0), (330, pytest.approx(0.0217 - 330 / 200000)))
    assert curve_for(m, "perfect").points == ((250, 0.0),)


def test_collapse_fixes_aim_at_the_hinge_and_scale_with_the_shortfall():
    from wobbl.plastic.study import fixes

    section = {"depth_mm": 10.0, "area_mm2": 200.0, "bending_share": 1.0}
    out = fixes("collapses", get_material("mild-steel"), 0.5, 0.8, 1.5, section, "on face 3 at (0, 0, 0)")
    kinds = [f["kind"] for f in out]
    assert kinds == ["section", "material", "load"]
    assert "13.7 mm" in out[0]["summary"]          # sqrt(1.5 / 0.8) times deeper
    assert "469 MPa" in out[1]["title"] and "4140" in out[1]["summary"]
    assert fixes("yields", get_material("mild-steel"), 0.5, 2.0, 1.5, section, "") == []


def test_cli_turns_away_a_brittle_part_before_meshing(parts):
    from click.testing import CliRunner

    from wobbl.cli.main import cli

    r = CliRunner().invoke(cli, ["plastic", parts["cantilever"], "-m", "pla", "--fix", "x=min",
                                 "--load", "100 N -z @ x=max"])
    assert r.exit_code != 0 and "brittle" in r.output


@pytest.mark.solver
@needs_ccx
def test_end_moment_collapses_at_the_fully_plastic_moment(tmp_path):
    """A half width 100 x 10 x 10 steel bar under an end moment through a rigid end, its root a symmetry plane.
    Collapse comes at sigma_y b h^2 / 4, first yield at two thirds of that, and the part keeps a permanent set."""
    from wobbl.plastic.study import PlasticStudy, run_plastic
    from wobbl.validation import geometry as G

    part = G.box(tmp_path / "bar.step", 100.0, 5.0, 10.0)
    mp = 250.0 * 10 * 10 ** 2 / 4 / 2
    st = PlasticStudy(str(part), "custom:E=200000,nu=0.3,yield=250,uts=400",
                      [f"{0.8 * mp / 1000} N*m +y at 100,2.5,5 @ x=max rigid"],
                      ["roller x=min", "roller y=min", "fixed-z x=min and box -1,-1,4,1,6,6"], quality="fast")
    o = run_plastic(st, tmp_path / "out")
    assert o.values["collapse_factor"] * 0.8 == pytest.approx(1.0, rel=0.02)
    assert o.values["first_yield_factor"] * 0.8 == pytest.approx(2 / 3, rel=0.02)
    assert o.verdict == "marginal" and o.values["permanent_set_mm"] > 0
    names = {c.name: c.status for c in o.checks}
    assert names["force balance"] == "pass" and names["unloading balance"] == "pass"
    assert names["hand calculation"] == "pass" and names["collapse"] == "pass"
    assert not (tmp_path / "out" / "work").exists()      # the solver's scratch is gone unless kept
    from wobbl.viz.scene import load
    head = load((tmp_path / "out" / "scene.wobbl").read_bytes())[0]     # the 3D view, every frame solved
    assert head["domain"] == "plastic" and [e["name"] for e in head["events"]][-1] == "collapse"


def test_custom_metal_without_a_tensile_strength_is_perfectly_plastic():
    h = hardening_curve(parse_custom("custom:E=200000,nu=0.3,yield=250"))
    assert h.points == ((pytest.approx(250 * (1 + 250 / 200000)), 0.0),) and h.kind == "bilinear"


def test_playback_is_a_small_decimated_surface_with_quantized_frames(parts, tmp_path):
    import base64
    import json

    import numpy as np

    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.plastic.common import write_playback
    from wobbl.structural.run import describe_part

    prep, _ = describe_part(parts["cantilever"], tmp_path)
    mesh = build_mesh(prep, Sizing(h=2.0))
    U = np.zeros((mesh.n_nodes, 3))
    U[:, 2] = -0.01 * mesh.X[:, 0]
    frames = [{"t": k / 2, "label": "loading", "U": U * k / 2, "plastic_strain": mesh.X[:, 0] * k / 400,
               "values": {"load": k / 2}} for k in range(3)]
    path = write_playback(tmp_path / "p.json", mesh, frames, {"kind": "test"},
                          {"plastic_strain": {"unit": "%", "min": 0.0, "max": 1.0}})
    text = open(path).read()
    d = json.loads(text)

    def decode(a):
        v = np.frombuffer(base64.b64decode(a["data"]), a["dtype"]).reshape(a["shape"]).astype(float)
        return a.get("offset", 0) + a.get("scale", 1) * v

    assert d["format"] == "wobbl-playback" and d["version"] == 1
    pos, idx = decode(d["mesh"]["vertices"]), decode(d["mesh"]["triangles"]).astype(int)
    assert len(pos) <= 2600 < len(np.unique(mesh.tris[:, :3])) and idx.max() < len(pos)
    assert pos.min(0) == pytest.approx([0, 0, 0], abs=0.01) and pos.max(0) == pytest.approx([200, 20, 10], abs=0.01)
    u = decode(d["frames"][2]["displacement"])
    assert u[:, 2] == pytest.approx(-0.01 * pos[:, 0], abs=1e-3)
    pe = decode(d["frames"][2]["plastic_strain"])
    assert pe == pytest.approx(pos[:, 0] / 200, abs=0.5 / 255) and d["frames"][1]["values"] == {"load": 0.5}
    assert len(text) < 300_000


def test_plastic_section_cuts_square_across_a_hook_shank(tmp_path):
    """The J hook example: the line from the shank to the sling leaves the shank at 24 degrees, but the section is
    cut square to the shank, 12 x 10 mm, carrying 3 kN x 22 mm of bending and 3 kN of tension."""
    from pathlib import Path

    from wobbl.materials import resolve_material
    from wobbl.mesh.mesher import Sizing, build_mesh
    from wobbl.plastic.section import plastic_section
    from wobbl.structural.build import build_static
    from wobbl.structural.run import describe_part, resolve_selections
    from wobbl.structural.spec import StructuralStudy, parse_fixture, parse_load

    hook = str(Path(__file__).resolve().parents[1] / "examples" / "steel-hook" / "hook.py")
    st = StructuralStudy(hook, "mild-steel", [parse_load("3 kN -z @ within 5 of 22,0,-16")],
                         [parse_fixture("bolt holes d=8.5")])
    prep, feats = describe_part(hook, tmp_path)
    loads, fixes, _ = resolve_selections(st, feats)
    mesh = build_mesh(prep, Sizing(h=4.0))
    built = build_static(mesh, loads, fixes, resolve_material("mild-steel"), feats.faces)
    s, why = plastic_section(mesh, [6.0, -1.7, 20.8], built.model, 250.0)
    assert s is not None, why
    assert s["axis"] == pytest.approx([0, 0, 1]) and s["area_mm2"] == pytest.approx(120, rel=0.01)
    lam = (-66 / 90 + ((66 / 90) ** 2 + 4 * 0.1 ** 2) ** 0.5) / (2 * 0.1 ** 2)    # (lam M/Mp) + (lam N/Np)^2 = 1
    assert s["factor"] == pytest.approx(lam, rel=0.01)
