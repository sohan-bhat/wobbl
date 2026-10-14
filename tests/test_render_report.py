import sys
from pathlib import Path

import numpy as np

from wobbl.geometry.cad import prepare
from wobbl.mesh.mesher import Sizing, build_mesh
from wobbl.report.html import page, table
from wobbl.report.render import convergence_plot, line_plot, render_structural


def test_heatmaps_render(parts, tmp_path, monkeypatch):
    for mod in ("pyvista", "vtk", "vtkmodules"):   # the pictures need neither VTK nor OpenGL
        monkeypatch.setitem(sys.modules, mod, None)
    prep = prepare(parts["plate_hole"], tmp_path / "g")
    m = build_mesh(prep, Sizing(h=6.0))
    measure = np.linalg.norm(m.X - m.X.mean(0), axis=1)
    U = np.zeros_like(m.X)
    U[:, 2] = m.X[:, 0] / 1000.0
    files = render_structural(tmp_path / "img", m.X, m.tets, m.tris, measure, U, m.X[np.argmax(measure)].tolist(),
                              float(measure.max()), "von Mises", 100.0, 2.0, [])
    for key in ("stress", "zoom", "safety", "deformed"):
        assert Path(files[key]).exists() and Path(files[key]).stat().st_size > 1000
    from PIL import Image

    img = np.asarray(Image.open(files["stress"]).convert("RGB")).astype(int)
    assert img.shape == (1000, 1600, 3)
    part = np.abs(img - [14, 15, 17]).sum(2) > 30            # pixels that are not background
    assert 0.05 < part[:, :1300].mean() < 0.9                  # the plate fills a good share of the frame
    levels = [{"n_nodes": n, "governing": g} for n, g in ((1000, 10.0), (2000, 11.0), (4000, 11.2))]
    assert Path(convergence_plot(tmp_path / "c.png", levels, [])).exists()
    assert Path(line_plot(tmp_path / "sub" / "l.png", [0, 1, 2], {"a": [1, 2, 3]}, "s", "V", 2.5, "limit")).exists()


def test_html_page_is_self_contained(tmp_path):
    html = page("Strength report", "part.step", "will it hold?", "HOLDS", "holds", "Peak 10 MPa.",
                [("peak", "10 MPa")], "high", ["passed everything"],
                [table(["a", "b"], [["<script>", "x"]])])
    assert "<script>" not in html.split("<style>")[1].split("</style>")[1] or "&lt;script&gt;" in html
    assert "http" not in html.replace("https://", "") or "<link" not in html


def test_report_embeds_the_brand_fonts_and_never_inter():
    """Archivo and IBM Plex Mono travel inside the page as small woff2 files, and no stack names Inter or leans on
    system-ui, which reads like Inter on a Mac."""
    import base64
    import re

    html = page("Strength report", "part.step", "", "HOLDS", "holds", "Peak 10 MPa.", [("peak", "10 MPa")], "high",
                [], [])
    style = html.split("<style>")[1].split("</style>")[0]
    fonts = re.findall(r'font-family:"([^"]+)";[^}]*base64,([A-Za-z0-9+/=]+)', style)
    assert sorted({f for f, _ in fonts}) == ["Archivo", "IBM Plex Mono"] and len(fonts) == 3
    sizes = [len(base64.b64decode(d)) for _, d in fonts]
    assert all(base64.b64decode(d)[:4] == b"wOF2" for _, d in fonts) and sum(sizes) < 40_000
    stacks = re.sub(r"base64,[A-Za-z0-9+/=]+", "", style)
    assert "Inter" not in stacks and "system-ui" not in stacks and "-apple-system" not in stacks
    assert "HOLDS" in html


def test_rasterizer_fills_without_cracks_and_keeps_the_nearest():
    """Two triangles sharing a diagonal cover every pixel of a 40 x 30 square once, with no gap along the shared
    edge, and a nearer triangle over part of it wins those pixels."""
    from wobbl.report.render import _rasterize

    x = np.array([10.0, 50.0, 50.0, 10.0, 0.0, 30.0, 0.0])
    y = np.array([10.0, 10.0, 40.0, 40.0, 0.0, 0.0, 30.0])
    iz = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    tri, _, _, _ = _rasterize(x, y, iz, np.array([[0, 1, 2], [0, 2, 3]]), 64, 48)
    cov = (tri >= 0).reshape(48, 64)
    assert cov.sum() == 40 * 30 and cov[10:40, 10:50].all()
    tri, w1, w2, _ = _rasterize(x, y, iz, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]]), 64, 48)
    tri = tri.reshape(48, 64)
    assert tri[12, 12] == 2 and tri[35, 45] in (0, 1)
    assert 0 <= w1.min() and w1.max() <= 1 and 0 <= w2.min() and w2.max() <= 1
