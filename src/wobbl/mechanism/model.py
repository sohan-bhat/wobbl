"""Load any MJCF or URDF model the way a user hands it over, and describe what is in it.

Real robot files are rarely tidy. A URDF points at meshes through ``package://`` paths, keeps some of them as
COLLADA files MuJoCo cannot read, leaves the inertia out of links that only carry a frame, or draws a part with a
visual shape and no collision shape. An MJCF file can name a texture that is not there. Each of these is handled
here the same way for every model, and each is written down as a note so the report can say what was assumed.

The result is an editable MjSpec, so an analysis can add a payload, a floor or a free base before compiling.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .install import require as _mj
from .spec import SpecError

MESH_TYPES = (".stl", ".obj", ".msh")


@dataclass
class Loaded:
    spec: object                          # mujoco.MjSpec
    path: Path
    kind: str                             # urdf or mjcf
    notes: list[str] = field(default_factory=list)
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)   # joint: (N m or N, rad/s or m/s)
    massless: list[str] = field(default_factory=list)   # bodies whose mass came from a file that is missing

    def compile(self):
        mujoco = _mj()
        try:
            model = self.spec.compile()
        except ValueError as e:
            raise SpecError(f"MuJoCo could not build {self.path.name}: {e}") from e
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        return model


# --- URDF ------------------------------------------------------------------------------------------------------
def find_file(ref: str, base: Path) -> Path | None:
    """A mesh a URDF names, wherever it really is: package:// paths are looked up by package folder name up the
    tree from the URDF, then by the rest of the path, then by file name anywhere near the URDF."""
    ref = ref.strip()
    if ref.startswith("file://"):
        ref = ref[7:]
    rest = ref
    pkg = ""
    if ref.startswith("package://"):
        pkg, _, rest = ref[10:].partition("/")
    p = Path(rest)
    if p.is_absolute() and p.exists():
        return p
    for a in [base, *base.parents][:6]:
        cands = ([a / pkg / rest] if pkg else []) + [a / rest]
        for c in cands:
            if c.is_file():
                return c.resolve()
    return _search(p.name, [base, base.parent])


def _search(name: str, roots: list[Path], max_depth: int = 4, budget: int = 20000) -> Path | None:
    """A file by name in the folders below ``roots``, looking a few levels down and giving up after ``budget``
    entries, so a model left in a home folder never sets off a crawl of the disk."""
    import os

    seen = 0
    for r in roots:
        top = len(r.parts)
        for here, dirs, files in os.walk(r):
            seen += len(files) + len(dirs)
            if name in files:
                return (Path(here) / name).resolve()
            if seen > budget:
                return None
            dirs[:] = sorted(d for d in dirs if not d.startswith(".")) if len(Path(here).parts) - top < max_depth \
                else []
    return None


def _shape_of(geo: ET.Element) -> ET.Element | None:
    return next(iter(geo), None) if geo is not None else None


def prepare_urdf(path: Path) -> tuple[str, list[str]]:
    """The URDF's text with every mesh path made absolute, meshes MuJoCo cannot use dropped, visual only parts
    given a collision shape and the compiler told how to read the rest. Returns the text and its notes."""
    root = ET.parse(path).getroot()
    notes, dropped, unreadable, copied, no_mass = [], set(), set(), [], []
    # simulator extras (ROS transmissions, Gazebo plugins, Drake's own tags) mean nothing to MuJoCo, and a
    # transmission's own <joint> would read as a joint with no parent
    for parent in list(root.iter()):
        for el in list(parent):
            if el.tag in ("transmission", "gazebo") or el.tag.startswith("{"):
                parent.remove(el)
    for el in root.iter():
        for k, v in el.attrib.items():
            if "$(" in v:
                el.set(k, _substitute(v, path.parent))
    links = {lk.get("name") for lk in root.iter("link")}
    for j in root.iter("joint"):     # Drake and Gazebo let a joint hang off a "world" link nobody wrote down
        par = j.find("parent")
        if par is not None and par.get("link") not in links and par.get("link") == "world":
            root.insert(0, ET.Element("link", name="world"))
            links.add("world")
    for link in root.iter("link"):
        for tag in ("collision", "visual"):
            for el in list(link.findall(tag)):
                shape = _shape_of(el.find("geometry"))
                if shape is None or shape.tag != "mesh":
                    continue
                ref = shape.get("filename", "")
                name = Path(ref.replace("package://", "")).name
                found = None if not name.lower().endswith(MESH_TYPES) else find_file(ref, path.parent)
                if found is None:
                    link.remove(el)
                    # a missing picture matters only when the part has no collision shape to fall back on
                    if tag == "collision" or link.find("collision") is None:
                        (dropped if name.lower().endswith(MESH_TYPES) else unreadable).add(name)
                    continue
                shape.set("filename", str(found))
        if link.find("collision") is None:
            for v in link.findall("visual"):
                if v.find("geometry") is not None and _shape_of(v.find("geometry")) is not None:
                    c = ET.SubElement(link, "collision")
                    for child in v:
                        if child.tag in ("origin", "geometry"):
                            c.append(child)
                    copied.append(link.get("name"))
                    break
        if link.find("inertial") is None and link.find("collision") is not None and _moves(root, link.get("name")):
            no_mass.append(link.get("name"))
    for j in root.iter("joint"):     # MuJoCo reads only these joint types
        if j.get("type") == "planar":
            raise SpecError(f"joint '{j.get('name')}' is planar, which MuJoCo's URDF reader does not take")
    mj = root.find("mujoco")
    if mj is None:
        mj = ET.SubElement(root, "mujoco")
    comp = mj.find("compiler")
    if comp is None:
        comp = ET.SubElement(mj, "compiler")
    for k, v in (("strippath", "false"), ("discardvisual", "true"), ("fusestatic", "false"),
                 ("balanceinertia", "true"), ("inertiafromgeom", "auto"), ("boundmass", "1e-6"),
                 ("boundinertia", "1e-12"), ("angle", "radian"), ("autolimits", "true")):
        comp.attrib.setdefault(k, v)
    if dropped:
        notes.append(f"{len(dropped)} mesh file{'s' if len(dropped) > 1 else ''} the URDF names could not be "
                     f"found ({_few(dropped)}), so those parts keep their mass but have no shape to collide with.")
    if unreadable:
        notes.append(f"{_few(unreadable)} {'are' if len(unreadable) > 1 else 'is'} in a mesh format MuJoCo cannot "
                     "read (it takes STL, OBJ and MSH), so those parts keep their mass but have no shape to collide "
                     "with.")
    if copied:
        notes.append(f"{_few(copied)} had only a visual shape, which is used as {'their' if len(copied) > 1 else 'its'}"
                     " collision shape.")
    if no_mass:
        notes.append(f"{_few(no_mass)} gave no mass, so Wobbl took it from {'their' if len(no_mass) > 1 else 'its'} "
                     "collision shape at 1000 kg/m3.")
    return ET.tostring(root, encoding="unicode"), notes


def _moves(root: ET.Element, link: str) -> bool:
    """Whether a URDF link can move, that is whether any joint between it and the root is not fixed."""
    up = {}
    for j in root.iter("joint"):
        c, p = j.find("child"), j.find("parent")
        if c is not None and p is not None:
            up[c.get("link")] = (p.get("link"), j.get("type"))
    seen = set()
    while link in up and link not in seen:
        seen.add(link)
        link, kind = up[link]
        if kind != "fixed":
            return True
    return False


def _substitute(v: str, base: Path) -> str:
    """What a URDF exported from xacro still says in ROS launch syntax: ``$(optenv NAME default)`` becomes its
    default and ``$(find pkg)`` the package folder. Anything else needs the xacro run first."""
    import re

    def one(m):
        word, _, rest = m.group(1).strip().partition(" ")
        if word == "optenv":
            return rest.partition(" ")[2]
        if word == "find":
            for a in [base, *base.parents][:6]:
                if (a / rest).is_dir():
                    return str(a / rest)
                if a.name == rest:
                    return str(a)
            return str(base)
        raise SpecError(f"the URDF still holds '$({m.group(1)})', so it needs xacro run on it first")

    return re.sub(r"\$\(([^)]*)\)", one, v)


def _few(items) -> str:
    items = sorted(items)
    return ", ".join(items[:4]) + (f" and {len(items) - 4} more" if len(items) > 4 else "")


def urdf_limits(text: str) -> dict[str, tuple[float, float]]:
    out = {}
    for j in ET.fromstring(text).iter("joint"):
        lim = j.find("limit")
        if lim is not None and j.get("type") in ("revolute", "continuous", "prismatic"):
            effort = float(lim.get("effort", 0) or 0)
            if effort > 0:
                out[j.get("name")] = (effort, float(lim.get("velocity", 0) or 0) or math.inf)
    return out


# --- loading ---------------------------------------------------------------------------------------------------
def load(path: str | Path) -> Loaded:
    """Any MJCF or URDF file as an editable MjSpec, with what Wobbl had to assume written down."""
    mujoco = _mj()

    p = Path(path).resolve()
    if not p.exists():
        raise SpecError(f"model file not found: {path}")
    head = p.read_text(encoding="utf-8", errors="replace")[:4000].lower()
    try:
        if p.suffix.lower() == ".urdf" or "<robot" in head and "<mujoco" not in head.split("<robot")[0]:
            text, notes = prepare_urdf(p)
            out = Loaded(mujoco.MjSpec.from_string(text), p, "urdf", notes, urdf_limits(text))
        else:
            out = Loaded(mujoco.MjSpec.from_file(str(p)), p, "mjcf")
            _fix_assets(out)
    except ValueError as e:
        raise SpecError(f"MuJoCo could not read {p.name}: {_short(str(e))}") from e
    model = out.compile()
    if out.kind == "mjcf":
        out.limits = actuator_limits(model)
    return out


def _fix_assets(ld: Loaded) -> None:
    """An MJCF file shared without all its files still runs: a missing texture is dropped, since it only colors
    the model, and a missing mesh is dropped with the shapes made from it, keeping the mass the model states for
    each body. A body whose mass came only from such a mesh is named, because its mass is then missing too."""
    spec = ld.spec
    base = Path(spec.modelfiledir or ld.path.parent)

    def where(kind_dir: str, f: str) -> Path:
        p = Path(f)
        return p if p.is_absolute() else base / (kind_dir or "") / p

    gone_tex = [t for t in spec.textures if t.file and not where(spec.texturedir, t.file).exists()]
    for t in gone_tex:
        for m in spec.materials:
            m.textures = ["" if x == t.name else x for x in m.textures]
        spec.delete(t)
    gone = {m.name or Path(m.file).stem: m for m in spec.meshes if m.file and not where(spec.meshdir, m.file).exists()}
    light = set()
    if gone:
        for g in list(spec.geoms):
            if g.meshname in gone:
                body = g.parent
                if not body.explicitinertial and (g.contype or g.conaffinity or g.mass or g.group < 3):
                    light.add(body.name)
                spec.delete(g)
        for m in gone.values():
            spec.delete(m)
    if gone_tex:
        ld.notes.append(f"{len(gone_tex)} texture file{'s are' if len(gone_tex) > 1 else ' is'} missing, which only "
                        "changes how the model looks.")
    if gone:
        ld.notes.append(f"{len(gone)} mesh file{'s' if len(gone) > 1 else ''} the model names "
                        f"{'are' if len(gone) > 1 else 'is'} missing ({_few(Path(m.file).name for m in gone.values())})"
                        ", so the shapes made from them are left out. Bodies keep the mass and inertia the model "
                        "states, and only their remaining shapes can collide.")
    if light:
        ld.notes.append(f"{_few(light)} got {'their' if len(light) > 1 else 'its'} mass from a missing mesh, so "
                        f"{'their' if len(light) > 1 else 'its'} mass is missing from every result.")
        ld.massless = sorted(light)


def _short(msg: str) -> str:
    return " ".join(msg.replace("Error:", "").split())[:300]


def actuator_limits(model) -> dict[str, tuple[float, float]]:
    """The torque each joint's actuator can give: its force range, or for a plain motor its control range, times
    its gear. Joint name to (N m or N, inf)."""
    mujoco = _mj()

    out = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
        gear = abs(float(model.actuator_gear[a, 0])) or 1.0
        if model.actuator_forcelimited[a]:
            lim = float(max(abs(model.actuator_forcerange[a])))
        elif model.actuator_ctrllimited[a] and model.actuator_gaintype[a] == 0 and model.actuator_biastype[a] == 0:
            lim = float(max(abs(model.actuator_ctrlrange[a]))) * float(model.actuator_gainprm[a, 0])
        else:
            continue
        if lim > 0 and jname:
            out[jname] = (lim * gear, math.inf)
    return out


# --- edits an analysis may make ------------------------------------------------------------------------------
def root_bodies(spec) -> list:
    return [b for b in spec.worldbody.bodies]


def is_floating(spec) -> bool:
    mujoco = _mj()
    return any(any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints) for b in root_bodies(spec))


def free_base(ld: Loaded) -> None:
    """Let the model's base move, as if it stood loose on a floor."""
    if is_floating(ld.spec):
        return
    roots = root_bodies(ld.spec)
    if len(roots) != 1:
        raise SpecError("the model has several separate bases, so Wobbl cannot tell which one stands on the floor")
    roots[0].add_freejoint()
    ld.notes.append(f"The base, {roots[0].name or 'the root body'}, stands loose on the floor instead of being "
                    "fixed in place.")


def has_floor(model) -> bool:
    return bool(np.any((model.geom_type == 0) & (model.geom_bodyid == 0)))


def add_floor(ld: Loaded, z: float, friction: float = 0.8) -> None:
    mujoco = _mj()
    ld.spec.worldbody.add_geom(name="wobbl_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                               pos=[0, 0, z], friction=[friction, 0.005, 0.0001], rgba=[0.2, 0.21, 0.23, 1],
                               contype=1, conaffinity=1)


def add_payload(ld: Loaded, mass: float, body: str, pos) -> None:
    """A load the robot carries, as a small solid ball rigidly fixed to ``body`` at ``pos`` in its frame."""
    mujoco = _mj()
    target = ld.spec.body(body)
    if target is None:
        raise SpecError(f"no body named '{body}' to carry the payload")
    r = (3 * mass / (4 * math.pi * 7800.0)) ** (1 / 3)
    p = target.add_body(name="wobbl_payload", pos=list(pos), mass=mass, ipos=[0, 0, 0],
                        inertia=[0.4 * mass * r * r] * 3, explicitinertial=True)
    p.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], contype=0, conaffinity=0, group=1,
               rgba=[0.85, 0.86, 0.88, 1], mass=0, density=0)


# --- what is in a compiled model ------------------------------------------------------------------------------
@dataclass
class Info:
    joints: list[dict]                    # hinge and slide joints: name, kind, dof, qadr, range, limited, body
    groups: np.ndarray                    # body to rigid group (bodies welded together share one)
    adjacent: set                         # pairs of groups joined directly by a joint
    floating: bool
    floor: bool
    mass: float
    leaves: list[int]                     # bodies at the ends of the chains, deepest first
    names: dict                           # body and joint names


def describe(model) -> Info:
    mujoco = _mj()

    nb = model.nbody
    bname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body {i}" for i in range(nb)]
    jname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint {i}" for i in range(model.njnt)]
    joints = []
    for i in range(model.njnt):
        t = int(model.jnt_type[i])
        if t not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        joints.append({"name": jname[i], "id": i, "kind": "hinge" if t == int(mujoco.mjtJoint.mjJNT_HINGE) else
                       "slide", "dof": int(model.jnt_dofadr[i]), "qadr": int(model.jnt_qposadr[i]),
                       "range": tuple(float(x) for x in model.jnt_range[i]), "limited": bool(model.jnt_limited[i]),
                       "body": int(model.jnt_bodyid[i])})
    # bodies with no joint of their own move with their parent, so they form one rigid group
    groups = np.arange(nb)
    for b in range(1, nb):
        if model.body_jntnum[b] == 0:
            groups[b] = groups[model.body_parentid[b]]
    adjacent = set()
    for b in range(1, nb):
        if model.body_jntnum[b] > 0:
            adjacent.add(frozenset((int(groups[b]), int(groups[model.body_parentid[b]]))))
    # a base that can lift off: a free joint, or a planar root (a vertical slide among its joints) as in many
    # walkers that move only in their own plane
    floating = any(model.body_parentid[model.jnt_bodyid[i]] == 0 and (
        int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE) or
        int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_SLIDE) and abs(float(model.jnt_axis[i, 2])) > 0.5)
        for i in range(model.njnt))
    children = [[] for _ in range(nb)]
    for b in range(1, nb):
        children[model.body_parentid[b]].append(b)
    depth = np.zeros(nb, int)
    for b in range(1, nb):
        depth[b] = depth[model.body_parentid[b]] + (1 if model.body_jntnum[b] else 0)
    heavy = [b for b in range(1, nb) if model.body_mass[b] > 1e-4 or model.body_geomnum[b] > 0]
    leaves = sorted((b for b in heavy if not any(c in heavy for c in _below(b, children))),
                    key=lambda b: -depth[b])
    return Info(joints, groups, adjacent, floating, has_floor(model), float(model.body_mass[1:].sum()), leaves,
                {"body": bname, "joint": jname})


def _below(b: int, children: list[list[int]]) -> list[int]:
    out, stack = [], list(children[b])
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(children[c])
    return out


def end_effector(model, info: Info) -> tuple[int, np.ndarray]:
    """Where a robot holds what it carries: the deepest body, or for a hand whose fingers branch off one palm, the
    palm, with the point midway between the finger tips. Returns the body and the point in its frame."""
    mujoco = _mj()

    if not info.leaves:
        raise SpecError("the model has no moving body to carry a payload")
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    tip = info.leaves[0]
    parent = int(model.body_parentid[tip])
    fingers = [b for b in info.leaves if model.body_parentid[b] == parent and model.body_jntnum[b] > 0]
    if parent > 0 and len(fingers) >= 2:
        body, pts = parent, [d.xipos[b] for b in fingers]
    else:
        body, pts = tip, [d.xipos[tip]]
    world = np.mean(pts, axis=0)
    R = d.xmat[body].reshape(3, 3)
    return body, R.T @ (world - d.xpos[body])


def lowest_point(model) -> float:
    """The lowest point of every collision shape at the model's starting pose, m."""
    mujoco = _mj()

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    low = math.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == 0 or (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0):
            continue
        c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3]
        half = np.abs(d.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g, 3:]
        low = min(low, float(c[2] - half[2]))
    return low if math.isfinite(low) else 0.0
