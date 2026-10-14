"""The peaks in a CalculiX .frd, read as the file grows, with nothing but the standard library.

A big job's .frd can run to gigabytes, so this never holds a field in memory. It reads whole result blocks
from where it stopped last time, keeps the largest displacement, von Mises stress and temperature of each
block with the node where it happens, and looks up those nodes' coordinates at the end. The same file runs on
a remote machine that has Python but not Wobbl, as 'python3 frdsummary.py job.frd > summary.json', so only
the summary comes back over the network.
"""
from __future__ import annotations

import json
import math
import sys


def _values(body: str) -> list[float]:
    out = []
    for i in range(0, len(body) - 11, 12):
        try:
            out.append(float(body[i:i + 12]))
        except ValueError:
            out.append(0.0)
    return out


def _measure(name: str, v: list[float]) -> tuple[float, float]:
    """The quantity ranked in a block and, for temperatures, the lowest value too."""
    if name.startswith("DISP") and len(v) >= 3:
        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2), 0.0
    if name.startswith("STRESS") and len(v) >= 6:
        sxx, syy, szz, sxy, syz, szx = v[:6]
        return math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                         + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)), 0.0
    return (v[0], v[0]) if v else (0.0, 0.0)


QUANTITY = {"DISP": "displacement", "STRESS": "von Mises stress", "NDTEMP": "temperature"}


def scan(path: str, state: dict | None = None) -> dict:
    """Read the result blocks written since the last call. ``state`` carries the file offset and every block
    seen so far, so a growing file is read once in total."""
    st = state if state is not None else {}
    for k, v in (("offset", 0), ("blocks", []), ("step", 0), ("nodes", 0)):
        st.setdefault(k, v)
    try:
        fh = open(path, "rb")
    except OSError:
        return st
    with fh:
        fh.seek(st["offset"])
        cur = None
        last_good = st["offset"]
        pos = st["offset"]
        for raw in fh:
            pos += len(raw)
            if not raw.endswith(b"\n"):
                break                                    # a line still being written
            line = raw.decode("ascii", "replace")
            if line.startswith("    2C"):
                try:
                    st["nodes"] = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("  100C"):
                parts = line.split()
                try:
                    t = float(parts[2])
                except (IndexError, ValueError):
                    t = 0.0
                step = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else st["step"]
                cur = {"time": t, "step": step}
            elif line.startswith(" -4") and cur is not None:
                cur["name"] = line.split()[1]
                cur["max"], cur["min"], cur["node"], cur["node_min"] = -math.inf, math.inf, 0, 0
            elif line.startswith(" -1") and cur is not None and "name" in cur:
                nid = int(line[3:13])
                val, low = _measure(cur["name"], _values(line[13:].rstrip("\n")))
                if val > cur["max"]:
                    cur["max"], cur["node"] = val, nid
                if cur["name"].startswith("NDTEMP") and low < cur["min"]:
                    cur["min"], cur["node_min"] = low, nid
            elif line.startswith(" -3"):
                if cur is not None and "name" in cur:
                    st["blocks"].append({k: v for k, v in cur.items()
                                         if not (isinstance(v, float) and math.isinf(v))})
                    cur = {"time": cur["time"], "step": cur["step"]}
                last_good = pos
            elif line.startswith("9999"):
                last_good = pos
        st["offset"] = last_good
    return st


def coordinates(path: str, ids: set[int]) -> dict[int, list[float]]:
    """Coordinates of a few nodes, read from the node block at the head of the file."""
    out: dict[int, list[float]] = {}
    with open(path, "r", errors="replace") as fh:
        inside = False
        for line in fh:
            if line.startswith("    2C"):
                inside = True
            elif inside and line.startswith(" -1"):
                nid = int(line[3:13])
                if nid in ids:
                    out[nid] = _values(line[13:].rstrip("\n"))[:3]
                    if len(out) == len(ids):
                        break
            elif inside and line.startswith(" -3"):
                break
    return out


def summary(path: str, state: dict | None = None) -> dict:
    """Every block's peak, and for each quantity the largest over the whole run with its time and place."""
    st = scan(path, state)
    peaks: dict[str, dict] = {}
    for b in st["blocks"]:
        name = b.get("name", "")
        key = next((k for k in QUANTITY if name.startswith(k)), None)
        if key is None:
            continue
        p = peaks.get(key)
        if p is None or b.get("max", -math.inf) > p["max"]:
            peaks[key] = {"quantity": QUANTITY[key], "max": b.get("max"), "node": b.get("node"),
                          "time": b["time"], "step": b["step"]}
        if key == "NDTEMP" and "min" in b:
            q = peaks[key]
            if "min" not in q or b["min"] < q["min"]:
                q["min"], q["node_min"] = b["min"], b.get("node_min")
    ids = {p["node"] for p in peaks.values() if p.get("node")}
    xyz = st.setdefault("xyz", {})             # coordinates already looked up, so a growing file is not reread
    if ids - set(xyz):
        xyz.update(coordinates(path, ids - set(xyz)))
    for p in peaks.values():
        if p.get("node") in xyz:
            p["xyz"] = xyz[p["node"]]
    history = {}
    for b in st["blocks"]:
        key = next((k for k in QUANTITY if b.get("name", "").startswith(k)), None)
        if key:
            history.setdefault(QUANTITY[key], []).append([b["time"], b.get("max")])
    return {"nodes": st["nodes"], "blocks": len(st["blocks"]), "peaks": list(peaks.values()), "history": history}


if __name__ == "__main__":
    json.dump(summary(sys.argv[1]), sys.stdout, indent=1)
    sys.stdout.write("\n")
