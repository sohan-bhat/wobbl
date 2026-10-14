"""Crop the example outputs and write them as WebP for the website.

Every picture on the site is a real Wobbl output from ``examples/``. This
script trims the empty background around each one and writes two widths so
the page can pick the right one with ``srcset``.

    python scripts/build_site_images.py            # writes site/public/img/*.webp
    python scripts/build_site_images.py enclosure-stress   # only the images named
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
OUT = ROOT / "site" / "public" / "img"
BG = np.array([0x0E, 0x0F, 0x11])

# name: (source, trim the empty background?, widths)
IMAGES = {
    "cast-mount-stress": (EX / "cast-motor-mount/output/images/stress.png", True, (1400, 800)),
    "bracket-as-drawn": (EX / "shelf-bracket/output/images/stress.png", True, (900, 560)),
    "bracket-fillet": (EX / "shelf-bracket/output/fix-fillet/images/stress.png", True, (900, 560)),
    "bracket-thicken": (EX / "shelf-bracket/output/fix-thicken/images/stress.png", True, (900, 560)),
    "bracket-gusset": (EX / "shelf-bracket/output/fix-gusset/images/stress.png", True, (900, 560)),
    "bracket-convergence": (EX / "shelf-bracket/output/images/convergence.png", False, (1200, 800)),
    "motor-mount-noribs": (EX / "motor-mount/output-noribs/images/stress.png", True, (1200, 800)),
    "motor-mount-ribs": (EX / "motor-mount/output/images/stress.png", True, (1200, 800)),
    "enclosure-stress": (EX / "enclosure/output/images/stress.png", True, (1200, 800)),
    "drone-arm-stress": (EX / "drone-arm/output/images/stress.png", True, (1200, 800)),
    "battery-temperature": (EX / "battery-pack/output/images/temperature.png", True, (1200, 800)),
    "heat-sink-temperature": (EX / "heat-sink/output/images/temperature.png", True, (1200, 800)),
    "battery-history": (EX / "battery-pack/output/images/history.png", False, (1200, 800)),
    "buck-waveform": (EX / "circuits/buck/images/waveform.png", False, (1200, 800)),
    "falling-rock-flight": (EX / "falling-rock/output/images/flight.png", False, (1200, 800)),
    "rocket-flight": (EX / "rocket-launch/output/images/flight.png", False, (1200, 800)),
    "rocket-path": (EX / "rocket-launch/output/images/path.png", False, (1200, 800)),
}


def trim(img: Image.Image, pad: int = 28) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(int)
    ink = np.abs(a - BG).sum(axis=2) > 24
    ys, xs = np.nonzero(ink)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.width)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.height)
    return img.crop((x0, y0, x1, y1))


def main(only: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, do_trim, widths) in IMAGES.items():
        if only and name not in only:
            continue
        img = Image.open(src).convert("RGB")
        if do_trim:
            img = trim(img)
        for i, w in enumerate(widths):
            w = min(w, img.width)
            h = round(img.height * w / img.width)
            out = OUT / (f"{name}.webp" if i == 0 else f"{name}-{w}.webp")
            img.resize((w, h), Image.LANCZOS).save(out, "WEBP", quality=82, method=6)
            print(f"{out.relative_to(ROOT)}  {w}x{h}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(sys.argv[1:])
