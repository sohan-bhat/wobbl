from pathlib import Path

import numpy as np

from wobbl.geometry.cad import prepare
from wobbl.mesh.mesher import Sizing, build_mesh
from wobbl.report.html import page, table
from wobbl.report.render import convergence_plot, line_plot, render_structural


def test_heatmaps_render(parts, tmp_path):
    prep = prepare(parts["plate_hole"], tmp_path / "g")
    m = build_mesh(prep, Sizing(h=6.0))
    measure = np.linalg.norm(m.X - m.X.mean(0), axis=1)
    U = np.zeros_like(m.X)
    U[:, 2] = m.X[:, 0] / 1000.0
    files = render_structural(tmp_path / "img", m.X, m.tets, m.tris, measure, U, m.X[np.argmax(measure)].tolist(),
                              float(measure.max()), "von Mises", 100.0, 2.0, [])
    for key in ("stress", "safety", "deformed"):
        assert Path(files[key]).exists() and Path(files[key]).stat().st_size > 1000
    levels = [{"n_nodes": n, "governing": g} for n, g in ((1000, 10.0), (2000, 11.0), (4000, 11.2))]
    assert Path(convergence_plot(tmp_path / "c.png", levels, [])).exists()
    assert Path(line_plot(tmp_path / "sub" / "l.png", [0, 1, 2], {"a": [1, 2, 3]}, "s", "V", 2.5, "limit")).exists()


def test_html_page_is_self_contained(tmp_path):
    html = page("Strength report", "part.step", "will it hold?", "HOLDS", "holds", "Peak 10 MPa.",
                [("peak", "10 MPa")], "high", ["passed everything"],
                [table(["a", "b"], [["<script>", "x"]])])
    assert "<script>" not in html.split("<style>")[1].split("</style>")[1] or "&lt;script&gt;" in html
    assert "http" not in html.replace("https://", "") or "<link" not in html
    assert "HOLDS" in html
