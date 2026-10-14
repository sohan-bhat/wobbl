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


PLAYBACK_VERTICES = 1500    # surface points kept for a playback file, enough to see the shape move


def write_playback(path: str | Path, mesh, frames: list[dict], manifest: dict, fields: dict[str, dict],
                   series: dict | None = None, body: np.ndarray | None = None) -> str:
    """A compact record of a run for a player to animate: a decimated surface, then for each saved frame its
    displacement and fields, quantized and base64 coded.

    Each frame holds ``u`` (the displacement, int16 scaled by ``u_scale`` mm) and one uint8 array per field named in
    ``fields``, scaled 0..255 between the field's ``min`` and ``max``. ``frames`` come in as dicts with ``t``, a
    ``label``, ``U`` (n, 3) on the mesh nodes and any of the field arrays (n,); ``series`` holds whole run curves.
    """
    import base64
    import json

    tris = mesh.tris[:, :3]
    used = np.unique(tris)
    X = mesh.X[used]
    # vertex clustering on a grid sized to keep about PLAYBACK_VERTICES points
    span = np.ptp(X, axis=0) + 1e-9
    cell = float((np.prod(span[span > 1e-6]) / PLAYBACK_VERTICES) ** (1.0 / max(1, int((span > 1e-6).sum())))) \
        if len(used) > PLAYBACK_VERTICES else 0.0
    if cell > 0:
        key = np.floor((X - X.min(0)) / cell).astype(np.int64)
        _, cluster = np.unique(key, axis=0, return_inverse=True)
        cluster = cluster.ravel()
    else:
        cluster = np.arange(len(used))
    k = int(cluster.max()) + 1
    lut = np.full(mesh.n_nodes, -1, np.int64)
    lut[used] = cluster
    t = lut[tris]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]   # triangles a cluster swallowed
    _, first = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    t = t[np.sort(first)]
    count = np.bincount(cluster, minlength=k).astype(float)

    def mean(v):
        v = np.asarray(v, float)[used]
        out = np.zeros((k,) + v.shape[1:])
        np.add.at(out, cluster, v)
        return out / count.reshape((-1,) + (1,) * (v.ndim - 1))

    def b64(a) -> str:
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

    pos = mean(mesh.X).astype(np.float32)
    umax = max(float(np.abs(fr["U"]).max()) for fr in frames) if frames else 0.0
    u_scale = umax / 32767.0 if umax > 0 else 1.0
    out_frames = []
    for fr in frames:
        f = {"t": round(float(fr["t"]), 6), "label": fr.get("label", ""),
             "u": b64(np.round(mean(fr["U"]) / u_scale).astype(np.int16))}
        for name, spec in fields.items():
            if name in fr:
                lo, hi = spec["min"], spec["max"]
                v = np.clip((mean(fr[name]) - lo) / max(hi - lo, 1e-30), 0, 1)
                f[name] = b64(np.round(255 * v).astype(np.uint8))
        f.update({k2: v2 for k2, v2 in fr.items() if k2 in ("values",)})
        out_frames.append(f)
    doc = {"format": "wobbl-playback-1", **manifest,
           "mesh": {"vertices": int(k), "triangles": int(len(t)), "positions": b64(pos),
                    "indices": b64(t.astype(np.uint32)), "dtype": {"positions": "float32 x3", "indices": "uint32 x3",
                                                                   "u": "int16 x3 times u_scale", "fields": "uint8"}},
           "u_scale": u_scale, "fields": fields, "frames": out_frames, "series": series or {}}
    if body is not None:
        doc["mesh"]["body"] = b64(np.round(mean(body)).astype(np.uint8))
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))
    return str(path)


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
