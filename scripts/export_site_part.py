"""Export a Wobbl strength result as a compact mesh for the website's 3D viewer.

The site lets visitors orbit a real result, read the stress under the cursor
and drag a load slider. This script reads a finished study's ``result.vtu``
(second order tetrahedra with per node fields), keeps only the outer surface,
and writes it as one gzipped binary file plus the numbers the page needs.

    python scripts/export_site_part.py \
        --variant distributed=examples/cast-motor-mount/output \
        --variant rigid=path/to/rigid/run \
        --name cast-motor-mount --load-n 2000 --out site/data

Each ``--variant`` is a result folder (with ``result.vtu`` and
``outcome.json``). Variants are separate meshes because each study refines
its own mesh; nothing is interpolated.

Binary layout of ``<name>-<variant>.bin.gz`` after gunzip, little endian,
every block padded to 4 bytes, offsets listed in the JSON manifest:

    position  uint16 x3   quantized over the bounding box
    normal    int8   x4   unit normal * 127 (fourth byte unused)
    disp      int16  x3   displacement in mm, scaled by disp_max / 32767
    stress    uint16 x1   von Mises in MPa, scaled by stress_max / 65535
    excluded  uint8  x1   1 where Wobbl excludes the zone from the verdict
    index     uint16 or uint32 x3 per triangle
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def surface(result_vtu: Path, feature_angle: float):
    import pyvista as pv

    grid = pv.read(result_vtu)
    # One level of subdivision keeps the mid edge nodes of the quadratic
    # elements, so fillets read smooth and the peak is not averaged away.
    surf = grid.extract_surface(nonlinear_subdivision=1).triangulate()
    # Split vertices along sharp CAD edges so flat faces shade flat and
    # fillets shade smooth.
    surf = surf.compute_normals(split_vertices=True, feature_angle=feature_angle,
                                auto_orient_normals=False, consistent_normals=True)
    return surf


def pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def export_variant(name: str, folder: Path, out: Path, feature_angle: float) -> dict:
    surf = surface(folder / "result.vtu", feature_angle)
    outcome = json.loads((folder / "outcome.json").read_text())

    X = np.asarray(surf.points, float)
    N = np.asarray(surf.point_data["Normals"], float)
    U = np.asarray(surf.point_data["displacement_mm"], float)
    S = np.asarray(surf.point_data["von_mises_MPa"], float)
    E = np.asarray(surf.point_data["excluded_zone"], float) > 0.5
    T = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]

    lo, hi = X.min(0), X.max(0)
    q_pos = np.round((X - lo) / np.where(hi > lo, hi - lo, 1) * 65535).astype("<u2")
    q_nrm = np.zeros((len(N), 4), "<i1")
    q_nrm[:, :3] = np.round(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12) * 127)
    disp_max = float(np.abs(U).max()) or 1.0
    q_disp = np.round(U / disp_max * 32767).astype("<i2")
    stress_max = float(S.max()) or 1.0
    q_stress = np.round(np.clip(S, 0, None) / stress_max * 65535).astype("<u2")
    q_excl = E.astype("u1")
    index_type = "uint16" if len(X) < 65536 else "uint32"
    q_index = T.astype("<u2" if index_type == "uint16" else "<u4")

    blocks, layout, offset = [], {}, 0
    for key, arr in (("position", q_pos), ("normal", q_nrm), ("disp", q_disp), ("stress", q_stress),
                     ("excluded", q_excl), ("index", q_index)):
        raw = pad4(arr.tobytes())
        layout[key] = offset
        offset += len(raw)
        blocks.append(raw)
    blob = b"".join(blocks)
    path = out / f"{name}.bin.gz"
    path.write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))

    allowed = ~E
    gov = float(S[allowed].max()) if allowed.any() else stress_max
    gov_i = int(np.flatnonzero(allowed)[np.argmax(S[allowed])]) if allowed.any() else int(np.argmax(S))
    return {
        "file": path.name,
        "bytes_gzip": path.stat().st_size,
        "bytes_raw": len(blob),
        "vertices": int(len(X)),
        "triangles": int(len(T)),
        "index_type": index_type,
        "layout": layout,
        "bounds_min": lo.round(4).tolist(),
        "bounds_max": hi.round(4).tolist(),
        "disp_max_mm": disp_max,
        "stress_max_MPa": stress_max,
        "surface_peak_MPa": round(gov, 3),
        "surface_peak_xyz": X[gov_i].round(3).tolist(),
        "governing_MPa": round(float(outcome["governing_MPa"]), 3),
        "governing_xyz": [round(v, 3) for v in outcome["governing_xyz"]],
        "max_disp_mm": round(float(outcome["max_disp_mm"]), 4),
        "verdict": outcome["verdict"],
        "confidence": outcome["confidence"],
        "source": str(folder),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", action="append", required=True, help="name=result_folder")
    ap.add_argument("--name", required=True, help="file name prefix")
    ap.add_argument("--load-n", type=float, required=True, help="load the study was solved at, in N")
    ap.add_argument("--out", type=Path, default=Path("site/data"))
    ap.add_argument("--feature-angle", type=float, default=35.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    variants, first, study = {}, None, None
    for spec in a.variant:
        key, folder = spec.split("=", 1)
        folder = Path(folder)
        variants[key] = export_variant(f"{a.name}-{key}", folder, a.out, a.feature_angle)
        first = first or json.loads((folder / "outcome.json").read_text())
        study = study or json.loads((folder / "study.json").read_text())

    manifest = {
        "name": a.name,
        "load_N": a.load_n,
        "allowable_MPa": first["allowable_MPa"],
        "target_sf": first["target_sf"],
        "measure": first["measure"],
        "note": "Linear elastic: stress and displacement scale in proportion to the load.",
        "loads": [{k: l.get(k) for k in ("kind", "text", "magnitude", "direction", "point", "coupling")}
                  for l in study["loads"]],
        "fixtures": [f["text"] for f in study["fixtures"]],
        "variants": variants,
    }
    (a.out / f"{a.name}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for k, v in variants.items():
        print(f"{k}: {v['vertices']} vertices, {v['triangles']} triangles, "
              f"{v['bytes_gzip'] / 1024:.0f} KB gzipped ({v['bytes_raw'] / 1024:.0f} KB raw), "
              f"surface peak {v['surface_peak_MPa']} MPa, governing {v['governing_MPa']} MPa")


if __name__ == "__main__":
    main()
