"""CalculiX decks for buckling, natural frequencies and steady state dynamics, built from a static model.

The strength study already turns loads and supports into a CalculiX deck. These steps reuse that deck exactly as
the static writer makes it (mesh, material, supports, rigid attachments and loads), so whatever it learns flows
into them, and replace only the step: *BUCKLE for load factors, *FREQUENCY for modes and *STEADY STATE DYNAMICS
for a harmonic response. The harmonic job reads the modes back from the .eig file of the frequency job instead of
solving the eigenproblem again.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..fe import RigidCoupling, StaticModel
from ..solvers.calculix import _run, write_static_inp

STEP_SUPPORTS = ("*BOUNDARY",)
STEP_LOADS = ("*CLOAD", "*DLOAD")


@dataclass
class Attached:
    """A mass the part carries, such as a motor or a battery, at a point and held by some faces."""
    name: str
    nodes: np.ndarray            # mesh nodes of the faces that carry it
    point: np.ndarray            # its center of mass, mm
    mass: float                  # t
    rigid: bool = True           # rigid: the faces move with it as one body; else spread over them as nodal masses
    weights: np.ndarray | None = None   # distributed: the share of each node, summing to one


@dataclass
class Deck:
    head: str                    # everything before the first *STEP: mesh, sets, material, rigid bodies, masses
    supports: str                # the *BOUNDARY lines of the static step
    loads: str                   # its *CLOAD and *DLOAD lines
    n_nodes: int
    extra: dict[str, tuple[int, int]] = field(default_factory=dict)   # rigid body -> (ref node, rot node), 0 based
    mass_nodes: dict[str, np.ndarray] = field(default_factory=dict)   # attached mass -> nodes carrying it, 0 based
    mass_values: dict[str, np.ndarray] = field(default_factory=dict)  # attached mass -> mass on each of them, t
    rigid_sets: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # ref node -> (nodes, point)
    notes: list[str] = field(default_factory=list)


def make_deck(model: StaticModel, path: Path, attached: list[Attached] = ()) -> Deck:
    """Write the static deck for ``model`` with the attached masses, then split it into the model and the step."""
    rigid = list(model.rigid)
    notes = []
    holder: dict[str, str] = {}
    held = [c.nodes for c in model.constraints] + [c.nodes for c in model.normal_constraints]
    held = np.unique(np.concatenate(held)) if held else np.zeros(0, int)
    for a in attached:
        if not a.rigid:
            continue
        same = next((r for r in rigid if np.intersect1d(r.nodes, a.nodes).size), None)
        if same is not None:
            # a node can belong to one rigid body only: the mass rides on the attachment already there
            holder[a.name] = same.name
            notes.append(f"{a.name} shares its faces with the rigid attachment of {same.name}, so its mass sits at "
                         f"that attachment's point ({', '.join(f'{v:.3g}' for v in same.ref_point)}).")
            continue
        nodes = np.setdiff1d(a.nodes, held)
        if len(nodes) < 3:
            raise ValueError(f"the faces holding {a.name} are all held by supports, so attach it to free faces")
        rigid.append(RigidCoupling(a.name, nodes, np.asarray(a.point, float)))
        holder[a.name] = a.name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = write_static_inp(path, replace(model, rigid=rigid))
    text = path.read_text()
    k = re.search(r"^\*STEP", text, re.M).start()
    head, step = text[:k], text[k:]
    blocks = _blocks(step)
    supports = "".join(b for b in blocks if b.upper().startswith(STEP_SUPPORTS))
    loads = "".join(b for b in blocks if b.upper().startswith(STEP_LOADS))
    n = model.mesh.n_nodes
    deck = Deck(head, supports, loads, n, extra, notes=notes)
    for r in rigid:
        deck.rigid_sets[extra[r.name][0]] = (np.asarray(r.nodes), np.asarray(r.ref_point, float))
    eid = _max_element(head)
    lines = []
    for a in attached:
        if a.rigid:
            nodes = np.array([extra[holder[a.name]][0]])
            values = np.array([a.mass])
        else:
            w = np.asarray(a.weights, float)
            nodes, values = np.asarray(a.nodes), a.mass * w / w.sum()
            keep = values > 0
            nodes, values = nodes[keep], values[keep]
        deck.mass_nodes[a.name], deck.mass_values[a.name] = nodes, values
        for i, (node, m) in enumerate(zip(nodes, values)):
            eid += 1
            tag = f"EM{len(deck.mass_nodes)}_{i + 1}"
            lines.append(f"*ELEMENT, TYPE=MASS, ELSET={tag}\n{eid},{int(node) + 1}\n*MASS, ELSET={tag}\n{m:.10g}\n")
    deck.head = head + "".join(lines)
    return deck


def fill_rigid(U: np.ndarray, X: np.ndarray, deck: Deck) -> np.ndarray:
    """Set the motion of every rigid attachment's reference node from the motion of the faces it holds.

    CalculiX writes zero for a rigid body's reference node in a frequency step, but the faces move as one body,
    u = u0 + theta x (x - p), so a least squares fit of u0 and theta over them recovers the motion at the point,
    which is where an attached mass sits. U is (n_all, 3) or (m, n_all, 3).
    """
    out = np.array(U, dtype=float, copy=True)
    single = out.ndim == 2
    if single:
        out = out[None]
    for ref, (nodes, point) in deck.rigid_sets.items():
        r = X[nodes] - point
        A = np.zeros((3 * len(nodes), 6))
        for k in range(3):
            A[k::3, k] = 1.0
        # theta x r = -r x theta: rows of the cross product matrix of r, negated
        A[0::3, 3:] = np.column_stack([np.zeros(len(r)), r[:, 2], -r[:, 1]])
        A[1::3, 3:] = np.column_stack([-r[:, 2], np.zeros(len(r)), r[:, 0]])
        A[2::3, 3:] = np.column_stack([r[:, 1], -r[:, 0], np.zeros(len(r))])
        P = np.linalg.pinv(A)
        for j in range(len(out)):
            out[j, ref] = (P @ out[j, nodes].reshape(-1))[:3]
    return out[0] if single else out


def _blocks(step: str) -> list[str]:
    """The keyword blocks of a step: each keyword line with the data lines under it."""
    out, cur = [], []
    for line in step.splitlines(keepends=True):
        if line.startswith("*") and not line.startswith("**"):
            if cur:
                out.append("".join(cur))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        out.append("".join(cur))
    return out


def _max_element(head: str) -> int:
    """The largest element number in the deck, so added mass elements never collide with one."""
    top, inside = 0, False
    for line in head.splitlines():
        if line.startswith("*"):
            inside = line.upper().startswith("*ELEMENT")
            continue
        if inside:
            try:
                top = max(top, int(line.split(",", 1)[0]))
            except ValueError:
                pass
    return top


def body_load(deck: Deck, accel: float, direction) -> str:
    """Loads for a base acceleration: the inertia of the part and of every attached mass, opposite the motion."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    s = f"*DLOAD\nEALL,GRAV,{accel:.10g},{d[0]:.10g},{d[1]:.10g},{d[2]:.10g}\n"
    rows = []
    for name, nodes in deck.mass_nodes.items():
        for node, m in zip(nodes, deck.mass_values[name]):
            for j in range(3):
                if abs(d[j]) > 1e-12:
                    rows.append(f"{int(node) + 1},{j + 1},{m * accel * d[j]:.10e}\n")
    return s + ("*CLOAD\n" + "".join(rows) if rows else "")


def buckle_inp(deck: Deck, n_modes: int) -> str:
    """A buckling step. CalculiX first solves the part under the loads, written as the first result set with its
    reactions, then the buckling factors and their mode shapes."""
    return (deck.head + f"*STEP\n*BUCKLE\n{n_modes}\n" + deck.supports + deck.loads +
            "*NODE FILE\nU,RF\n*EL FILE\nS\n*END STEP\n")


def frequency_inp(deck: Deck, n_modes: int, stress: bool = True) -> str:
    out = "*NODE FILE\nU\n" + ("*EL FILE\nS\n" if stress else "")
    return deck.head + f"*STEP\n*FREQUENCY, STORAGE=YES\n{n_modes}\n" + deck.supports + out + "*END STEP\n"


def harmonic_inp(deck: Deck, loads: str, sweeps: list[tuple[float, float, int, float]], damping: float,
                 n_modes: int, print_set: np.ndarray | None = None) -> str:
    """Steady state dynamics steps that read the modes from the .eig file of a frequency job with the same mesh.

    Each sweep is (low Hz, high Hz, points per interval, bias). CalculiX puts its points at every natural frequency
    inside the range and spreads the rest between them, denser near the modes when the bias is above one. Every
    sweep prints the motion of ``print_set`` to the .dat file, and only the last one writes the whole field, so the
    .frd holds just the frequencies whose stress is wanted. File output requests carry over to later steps in
    CalculiX, which is why the field step comes last.
    """
    s = deck.head
    printing = print_set is not None and len(print_set) > 0
    if printing:
        s += _nset_text("NRESP", print_set)
    for k, (lo, hi, pts, bias) in enumerate(sweeps):
        s += (f"*STEP\n*STEADY STATE DYNAMICS\n{lo:.10g},{hi:.10g},{int(pts)},{bias:.10g}\n" + deck.supports +
              f"*MODAL DAMPING\n1,{n_modes},{damping:.10g}\n" + loads)
        if printing:
            s += "*NODE PRINT, NSET=NRESP\nU\n"
        if k == len(sweeps) - 1:
            s += "*NODE FILE\nU\n*EL FILE\nS\n"
        s += "*END STEP\n"
    return s


def _nset_text(name: str, nodes: np.ndarray) -> str:
    ids = np.asarray(nodes, dtype=np.int64) + 1
    rows = [", ".join(map(str, ids[i:i + 12])) for i in range(0, len(ids), 12)]
    return f"*NSET, NSET={name}\n" + "\n".join(rows) + "\n"


def run(ccx: Path, work: Path, job: str, text: str, timeout: float = 3600, eig_from: str | None = None) -> str:
    """Write the deck and run it. ``eig_from`` names a job in the same folder whose modes this one reuses."""
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{job}.inp").write_text(text)
    if eig_from:
        shutil.copyfile(work / f"{eig_from}.eig", work / f"{job}.eig")
    log, _ = _run(ccx, work, job, timeout)
    return log


# --------------------------------------------------------------------------------------------- readers
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def read_buckling_factors(path: Path) -> list[float]:
    text = Path(path).read_text(errors="replace")
    part = text.split("B U C K L I N G   F A C T O R   O U T P U T", 1)
    if len(part) < 2:
        return []
    out = []
    for line in part[1].splitlines():
        m = re.match(rf"^\s*(\d+)\s+({_NUM})\s*$", line)
        if m:
            out.append(float(m.group(2)))
        elif out and line.strip():
            break
    return out


def _table(text: str, title: str, cols: int) -> np.ndarray:
    """Rows 'mode value value ...' under a spaced out title in the .dat file."""
    part = text.split(title, 1)
    if len(part) < 2:
        return np.zeros((0, cols))
    rows = []
    for line in part[1].splitlines():
        vals = line.split()
        if len(vals) == cols + 1 and vals[0].isdigit():
            rows.append([float(v) for v in vals[1:]])
        elif rows and (vals[:1] == ["TOTAL"] or not vals or not vals[0].isdigit()):
            if vals:
                break
    return np.array(rows) if rows else np.zeros((0, cols))


def read_eigen(path: Path) -> dict:
    """Natural frequencies, participation factors and effective modal masses from a frequency job's .dat file.

    CalculiX normalizes each mode to unit modal mass, so a participation factor is in sqrt(t) and an effective
    mass in t. Columns are x, y, z and rotations about x, y and z.
    """
    text = Path(path).read_text(errors="replace")
    eig = _table(text, "E I G E N V A L U E   O U T P U T", 4)
    part = _table(text, "P A R T I C I P A T I O N   F A C T O R S", 6)
    eff = _table(text, "E F F E C T I V E   M O D A L   M A S S", 6)
    total = np.zeros(6)
    t = text.split("T O T A L   E F F E C T I V E   M A S S", 1)
    if len(t) == 2:
        nums = [float(v) for v in re.findall(_NUM, t[1].split("F R A C T I O N", 1)[0])]
        if len(nums) >= 6:
            total = np.array(nums[:6])
    return {"eigenvalue": eig[:, 0] if len(eig) else np.zeros(0), "freq": eig[:, 2] if len(eig) else np.zeros(0),
            "participation": part, "effective_mass": eff, "total_mass": total}


def read_node_print(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Printed displacements of a steady state job: frequency -> (node ids, real part (k,3), imaginary part (k,3)).

    CalculiX prints each frequency twice, the real part first and the imaginary part second.
    """
    text = Path(path).read_text(errors="replace")
    out: dict[float, list] = {}
    for m in re.finditer(rf"displacements \(vx,vy,vz\) for set \S+ and time\s+({_NUM})\s*\n\s*\n((?:\s+\d+(?:\s+{_NUM}){{3}}\s*\n)+)",
                         text):
        f = float(m.group(1))
        rows = np.array([[float(v) for v in ln.split()] for ln in m.group(2).strip().splitlines()])
        out.setdefault(f, []).append(rows)
    res = {}
    for f, parts in out.items():
        if len(parts) >= 2:
            res[f] = (parts[0][:, 0].astype(int) - 1, parts[0][:, 1:4], parts[1][:, 1:4])
    return res


def read_blocks(path: Path, n: int, names: tuple[str, ...] = ("DISP", "STRESS")) -> dict[str, list]:
    """Nodal result blocks of an ASCII .frd: name -> [(time, (n, k) float32 array), ...] for the names asked.

    The values sit in fixed 12 character columns, often with no space between them, so each block's numbers are
    cut out as bytes and converted by numpy in one go. A modal job on a large mesh writes hundreds of megabytes,
    which a line by line reader would hold as Python objects many times over.
    """
    out: dict[str, list] = {}
    t, name, k, ids, vals = 0.0, None, 0, [], []
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"  100C"):
                try:
                    t = float(line.split()[2])
                except (IndexError, ValueError):
                    t = 0.0
            elif line.startswith(b" -4"):
                parts = line.split()
                name, k = parts[1].decode(), int(parts[2])
                ids, vals = [], []
                if name not in names:
                    name = None
            elif name is None:
                continue
            elif line.startswith(b" -1"):
                ids.append(line[3:13])
                vals.append(line[13:].rstrip(b"\r\n").ljust(12 * k)[:12 * k])
            elif line.startswith(b" -3"):
                arr = np.zeros((n, k), np.float32)
                if ids:
                    idx = np.array(ids, dtype="S10").astype(np.int64) - 1
                    v = np.char.strip(np.frombuffer(b"".join(vals), dtype="S12").reshape(len(vals), k))
                    v[v == b""] = b"0"
                    keep = (idx >= 0) & (idx < n)
                    arr[idx[keep]] = v[keep].astype(np.float64)
                out.setdefault(name, []).append((t, arr))
                name = None
    return out


def read_modes(path: Path, n: int, n_extra: int = 0) -> tuple[list[float], np.ndarray, np.ndarray | None]:
    """Mode shapes from a .frd: (labels, displacements (m, n + n_extra, 3), stresses (m, n, 6) or None).

    Each mode block carries its natural frequency, or its buckling factor, as the time.
    """
    blocks = read_blocks(path, n + n_extra)
    disp = blocks.get("DISP", [])
    labels = [t for t, _ in disp]
    U = np.array([a[:, :3] for _, a in disp], dtype=np.float32) if disp else np.zeros((0, n + n_extra, 3), np.float32)
    S = None
    if "STRESS" in blocks:
        S = np.array([a[:n, :6] for _, a in blocks["STRESS"]], dtype=np.float32)
    return labels, U, S


def read_harmonic_field(path: Path, n: int) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """The complex displacement and stress of the field writing step: frequency -> (U (n,3), S (n,6)) complex."""
    b = read_blocks(path, n, ("DISP", "DISPI", "STRESS", "STRESSI"))
    out = {}
    if not all(k in b for k in ("DISP", "DISPI", "STRESS", "STRESSI")):
        return out
    for (f, ur), (_, ui), (_, sr), (_, si) in zip(b["DISP"], b["DISPI"], b["STRESS"], b["STRESSI"]):
        out[f] = (ur[:, :3] + 1j * ui[:, :3], sr[:, :6] + 1j * si[:, :6])
    return out
