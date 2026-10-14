"""Pieces the plasticity, contact and bolted joint studies share: their outcome and report, light images, where a
point sits on the part, and removing the solver's scratch files after a run."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome

TITLES = {"plastic": "Plasticity report", "contact": "Contact report", "bolt": "Bolted joint report"}
VERDICT_CLASS = {"holds": "pass", "yields": "warn", "marginal": "warn", "collapses": "fail", "fails": "fail",
                 "slips": "fail", "separates": "fail"}


class NonlinearOutcome(GenericOutcome):
    """The generic outcome with these studies' report titles, verdict colors, tables and notes."""

    def write(self, out_dir: str | Path, sections_extra: list[str] | None = None) -> "NonlinearOutcome":
        from ..report.html import badge, esc, figure, page, table

        out = Path(out_dir)
        sections = []
        for key, cap in self.values.get("_figures", []):
            if self.images.get(key):
                sections.append(figure(self.images[key], cap))
        sections.append("<h2>Checks</h2>" + table(["status", "check", "result"],
                                                  [[badge(c.status), c.name, c.summary] for c in self.checks]))
        if self.fixes:
            sections.append("<h2>How to fix it</h2>" + "".join(
                f"<div class='fix'><div class='t'>{esc(f['title'])}</div><div>{esc(f['summary'])}</div></div>"
                for f in self.fixes))
        # tables travel as data (title, headers, rows) so a result read back from outcome.json rebuilds them
        for title, headers, rows in self.values.get("_tables", []):
            sections.append(f"<h2>{esc(title)}</h2>" + table(headers, rows))
        sections += sections_extra or []
        sections.append("<h2>Assumptions</h2><ol class='assume'>" +
                        "".join(f"<li>{esc(a)}</li>" for a in self.assumptions) + "</ol>")
        if self.notes:
            sections.append("<h2>Notes</h2><ul class='plain'>" + "".join(f"<li>{esc(n)}</li>" for n in self.notes)
                            + "</ul>")
        html = page(TITLES.get(self.domain, "Report"), Path(self.part).name if self.part else self.domain,
                    self.question, self.verdict.upper(), VERDICT_CLASS.get(self.verdict, "warn"), self.headline,
                    self.metrics, self.confidence, self.confidence_reasons, sections,
                    f"Solver {self.solver}. Total time {self.t_total:.0f} s.")
        self.report = str(out / "report.html")
        self.workdir = str(out)
        Path(self.report).write_text(html, encoding="utf-8")
        (out / "summary.txt").write_text(self.text() + "\n", encoding="utf-8")
        self.save(out / "outcome.json")
        from ..viz.embed import after_report
        after_report(out)
        return self


def light_png(path: str | Path) -> str:
    """Rewrite a rendered image with a 256 color palette, about a third of the size, so reports stay light."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            small = im.convert("RGB").quantize(256)
        small.save(path, optimize=True)
    except Exception:  # noqa: BLE001  a heavier image is still an image
        pass
    return str(path)


def clean_work(work: Path, keep: bool):
    """Remove the solver scratch of a finished run (decks, results, meshes) unless it is wanted for a closer look."""
    if not keep and Path(work).exists():
        shutil.rmtree(work, ignore_errors=True)


def where(mesh, faces: dict, xyz) -> str:
    """Name a point by the face it sits on: a fillet, a hole, or a face id, with the coordinates."""
    from ..structural.hotspots import nearest_face

    xyz = np.asarray(xyz, float)
    at = f"({', '.join(f'{v:.3g}' for v in xyz)})"
    fid = nearest_face(mesh, xyz)
    f = faces.get(fid) if fid is not None else None
    if f is None:
        return f"inside the part at {at}"
    if f.role == "fillet":
        return f"on the {f.radius:.3g} mm inside fillet (face {f.id}) at {at}"
    if f.role == "hole":
        return f"at the edge of the {2 * f.radius:.3g} mm hole (face {f.id}) at {at}"
    return f"on face {f.id} at {at}"


def write_playback(path: str | Path, mesh, frames: list[dict], head: dict, fields: dict[str, dict],
                   series: dict | None = None, body: np.ndarray | None = None) -> str:
    """playback.json for a player, in the encoding the other studies use (see life/playback.py): the decimated
    surface, whose vertices are original mesh nodes, then each saved frame of the run.

    ``frames`` come in as dicts with ``t`` (playback order), ``label``, ``values`` (numbers to show beside the
    picture), ``U`` (n, 3) on the mesh nodes and any field named in ``fields`` (n,). Each frame goes out with its
    displacement as int16 and each field as uint8 between the field's ``min`` and ``max``; every array carries the
    dtype, shape, scale and offset that turn the stored integers back into values. ``body`` (n,) numbers the bodies
    of an assembly.
    """
    from ..life.playback import encode, mesh_block, quantize, surface, write

    pts, faces, nodes = surface(mesh)
    out = []
    for fr in frames:
        f = {"t": round(float(fr["t"]), 6), "label": fr.get("label", ""),
             "values": {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in fr.get("values", {}).items()},
             "displacement": quantize(np.asarray(fr["U"], float)[nodes], "int16")}
        for name, spec in fields.items():
            if name in fr:
                lo, hi = float(spec["min"]), float(spec["max"])
                span = max(hi - lo, 1e-30)
                q = np.round(np.clip((np.asarray(fr[name], float)[nodes] - lo) / span, 0, 1) * 255)
                f[name] = encode(q.astype("uint8"), "uint8", scale=span / 255, offset=lo)
        out.append(f)
    doc = {"format": "wobbl-playback", "version": 1, **head, "mesh": mesh_block(pts, faces),
           "fields": fields, "frames": out, "series": series or {}}
    if body is not None:
        doc["mesh"]["body"] = encode(np.asarray(body)[nodes].astype("uint8"), "uint8")
    return write(Path(path), doc)


def load_size(loads, factor: float) -> str:
    """The first load scaled by a factor, in its own units, for a headline: '3.9 kN'."""
    from ..units import fmt, fmt_force

    if not loads:
        return ""
    ld = loads[0][0] if isinstance(loads[0], tuple) else loads[0]
    v = ld.magnitude * factor
    if ld.kind in ("force", "bolt", "remote"):
        return fmt_force(v, ld.imperial) if v < 1000 else f"{fmt(v / 1000)} kN"
    if ld.kind == "pressure":
        return f"{fmt(v)} MPa"
    if ld.kind == "moment":
        return f"{fmt(v / 1000)} N m"
    return f"{fmt(v / 9806.65)} g"
