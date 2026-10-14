"""Read print settings from a slicer's own files, so a warp study prints the part the way the printer will.

Cura keeps settings as ini style key = value lines (a .curaprofile is a zip of
those, and .inst.cfg files sit in its profile folders). Creality Print, Orca and
Bambu Studio keep JSON with their own key names, and a .3mf project from any of
them carries its settings inside. PrusaSlicer and SuperSlicer write .ini files.
Each slicer names the same setting its own way; the table below maps them onto
the handful a warp study needs, and the result says which file and key each
value came from.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

# setting -> keys the slicers use for it, first match wins
KEYS = {
    "layer_height": ("layer_height",),
    "bed_C": ("material_bed_temperature", "material_bed_temperature_layer_0", "bed_temperature", "hot_plate_temp",
              "textured_plate_temp", "eng_plate_temp", "cool_plate_temp", "first_layer_bed_temperature",
              "hot_plate_temp_initial_layer"),
    "chamber_C": ("build_volume_temperature", "chamber_temperature", "chamber_temperatures"),
    "nozzle_C": ("material_print_temperature", "temperature", "nozzle_temperature", "first_layer_temperature"),
    "infill": ("infill_sparse_density", "fill_density", "sparse_infill_density"),
    "walls": ("wall_line_count", "perimeters", "wall_loops"),
    "top_layers": ("top_layers", "top_solid_layers", "top_shell_layers"),
    "bottom_layers": ("bottom_layers", "bottom_solid_layers", "bottom_shell_layers"),
    "line_width": ("line_width", "wall_line_width", "extrusion_width", "perimeter_extrusion_width",
                   "outer_wall_line_width"),
    "brim_mm": ("brim_width",),
    "adhesion": ("adhesion_type", "brim_type"),
    "material": ("material_type", "filament_type", "material"),
    "fan": ("cool_fan_speed", "fan_max_speed", "max_fan_speed", "fan_cooling_layer_time"),
}


def _number(v) -> float | None:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else None


def _ini(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;[" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _flat(d, out=None) -> dict:
    """JSON settings, nested or not, as one key to value map (the last of a list stands for all its extruders)."""
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict,)):
                _flat(v, out)
            else:
                out[k] = v
    return out


def _texts(path: Path) -> list[tuple[str, str]]:
    """(name, text) of every settings file inside path: itself, or the members of a zip or .3mf."""
    data = path.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".cfg", ".config", ".ini", ".json", ".curaprofile"))
                     or "metadata/" in n.lower() and n.lower().endswith((".config", ".json"))]
            return [(n, z.read(n).decode("utf-8", errors="replace")) for n in names]
    return [(path.name, data.decode("utf-8", errors="replace"))]


def read_profile(path: str | Path) -> dict:
    """Settings found in a slicer file: {setting: value}, plus 'sources' {setting: 'file:key'} and 'slicer'."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"slicer profile not found: {path}")
    found, sources, slicer = {}, {}, "unknown"
    for name, text in _texts(p):
        t = text.lstrip()
        if t.startswith("{"):
            try:
                kv = _flat(json.loads(t))
            except json.JSONDecodeError:
                continue
            slicer = "Creality Print, Orca or Bambu Studio" if any(k in kv for k in ("hot_plate_temp",
                                                                                      "sparse_infill_density",
                                                                                      "wall_loops")) else "JSON"
        else:
            kv = _ini(text)
            if any(k.startswith(("material_", "infill_sparse", "wall_line")) for k in kv):
                slicer = "Cura"
            elif any(k in kv for k in ("perimeters", "fill_density", "bed_temperature")):
                slicer = "PrusaSlicer"
        for setting, keys in KEYS.items():
            if setting in found:
                continue
            for k in keys:
                if k in kv and str(kv[k]).strip() not in ("", "nil", "[]"):
                    v = kv[k]
                    if setting in ("adhesion", "material"):
                        val = (v[0] if isinstance(v, list) and v else str(v)).strip().strip('"').lower()
                    else:
                        val = _number(v)
                        if val is None:
                            continue
                    found[setting] = val
                    sources[setting] = f"{name}:{k}"
                    break
    if "infill" in found and found["infill"] > 1.0:
        found["infill"] = found["infill"] / 100.0           # every slicer writes infill in percent
    found["sources"] = sources
    found["slicer"] = slicer
    return found
