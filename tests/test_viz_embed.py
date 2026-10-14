"""The viewer goes into report.html once, right after the verdict, and a run without a report gets its own page."""
import re

import numpy as np

from wobbl.viz import Scene, attach, view_path
from wobbl.viz.embed import END, START, bundle, insert, viewer_js

REPORT = """<!doctype html><html><body><main><header>h</header>
<section class="verdict"><div>HOLDS</div></section>
<h2>Checks</h2></main></body></html>"""


def small(out):
    s = Scene("tiny", domain="test")
    s.timeline([0.0, 1.0])
    s.body("sphere", [1.0], pos=np.zeros((2, 3)))
    s.readout("t", [0.0, 1.0], unit="s")
    return s.save(out)


def test_viewer_is_small_and_whole():
    js = viewer_js()
    assert len(js) < 60_000
    assert js.startswith("(() => {") and js.rstrip().endswith("})();")
    assert not re.search(r"font[^;{}]*\bInter\b", bundle())       # Archivo and IBM Plex Mono only


def test_attach_puts_the_view_after_the_verdict_once(tmp_path):
    small(tmp_path)
    (tmp_path / "report.html").write_text(REPORT)
    attach(tmp_path)
    attach(tmp_path)
    page = (tmp_path / "report.html").read_text()
    assert page.count(START) == 1 and page.count(END) == 1
    assert page.index('class="verdict"') < page.index(START) < page.index("<h2>Checks</h2>")
    assert view_path(tmp_path) == tmp_path / "report.html"


def test_a_run_without_a_report_gets_a_page(tmp_path):
    small(tmp_path)
    page = attach(tmp_path)
    assert page.name == "view.html" and "wobbl-scene-data" in page.read_text()


def test_nothing_to_show(tmp_path):
    assert attach(tmp_path) is None and view_path(tmp_path) is None


def test_insert_without_a_verdict_goes_before_main_ends():
    out = insert("<main><p>x</p></main>", START + "v" + END)
    assert out.index(START) > out.index("<p>x</p>")
