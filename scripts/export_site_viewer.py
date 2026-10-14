"""Put the 3D viewer and every example's scene where the website can load them.

    python scripts/export_site_viewer.py

writes site/public/viewer/wobbl-view.js (the same script every report embeds) and copies each committed
examples/<name>/<run>/scene.wobbl to site/public/data/scenes/<name>[-<run>].wobbl. A page then shows a scene with
<div data-wobbl-scene="/wobbl/data/scenes/<name>.wobbl"></div> and the script (docs/viz.md).
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from wobbl.viz.embed import viewer_js

    site = ROOT / "site" / "public"
    (site / "viewer").mkdir(parents=True, exist_ok=True)
    (site / "viewer" / "wobbl-view.js").write_text(viewer_js() + "\n", encoding="utf-8")
    scenes = site / "data" / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((ROOT / "examples").glob("*/**/scene.wobbl")):
        rel = p.parent.relative_to(ROOT / "examples").parts
        name = rel[0] + "".join(f"-{x}" for x in rel[1:] if x != "output")
        shutil.copyfile(p, scenes / f"{name}.wobbl")
        n += 1
    print(f"viewer {len(viewer_js()) / 1000:.1f} kB, {n} scenes in {scenes.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
