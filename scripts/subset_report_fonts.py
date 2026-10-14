"""Cut the site's Archivo and IBM Plex Mono down to the glyphs a report uses, for embedding in report.html.

Run it whenever the site's fonts or the character set change. It needs fontTools and brotli, which Wobbl itself
does not install:

    python -m pip install fonttools brotli
    python scripts/subset_report_fonts.py

Archivo is variable in weight and width; the report uses its normal width, so the width axis is pinned at 100
and the weight axis kept, which serves every weight from 400 to 800 from one file. IBM Plex Mono comes as static
400 and 600 files. Glyphs outside the set fall back to the next face in the report's font stack.
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site" / "public" / "fonts"
OUT = ROOT / "src" / "wobbl" / "report" / "fonts"
# printable ASCII, a no break space, and the symbols reports print: degree, plus or minus, squared, cubed, micro,
# middle dot, times, and curly quotes and the ellipsis
TEXT = "".join(chr(c) for c in range(0x20, 0x7F)) + " °±²³µ·×’“”…"
FONTS = {"archivo.woff2": ("archivo-latin.woff2", {"wdth": 100}),
         "plex-mono-400.woff2": ("ibm-plex-mono-400-latin.woff2", None),
         "plex-mono-600.woff2": ("ibm-plex-mono-600-latin.woff2", None)}


def build(source: Path, pin: dict | None) -> bytes:
    font = TTFont(source)
    if pin:
        font = instancer.instantiateVariableFont(font, pin)
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga", "tnum"]
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]     # keep the copyright and license records the OFL asks for
    opts.notdef_outline = True
    opts.hinting = False
    opts.desubroutinize = True
    s = subset.Subsetter(opts)
    s.populate(text=TEXT)
    s.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return buf.getvalue()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (source, pin) in FONTS.items():
        data = build(SITE / source, pin)
        (OUT / name).write_bytes(data)
        print(f"{name:<22} {len(data) / 1000:5.1f} kB from {source} ({(SITE / source).stat().st_size / 1000:.1f} kB)")
    for lic in ("OFL-Archivo.txt", "OFL-IBMPlexMono.txt"):
        shutil.copyfile(SITE / lic, OUT / lic)


if __name__ == "__main__":
    main()
