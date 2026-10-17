"""What the per example scripts share: the repository paths, a run's scene, each object's placement in millimetres,
and moving an assembly built in the scene's world into the frame of the object it will hang under.

An assembly is built where it stands at the frame the page opens on, in millimetres and in the scene's world
axes. Hung under its object, the runtime moves it with the object's live pose, so a joint built on the simulated
axis stays on it through the whole run.
"""
import os

import numpy as np

from scene import Scene

from .util import place

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))


def scene(rel):
    """The scene.wobbl of a run, from its folder under examples/ (desktop-arm/output)."""
    return Scene(os.path.join(REPO, 'examples', rel, 'scene.wobbl'))


def mm_scale(sc):
    """Millimetres per scene unit."""
    return {'m': 1000.0, 'mm': 1.0}.get(sc.h.get('units', {}).get('length', 'mm'), 1.0)


def pose_mm(sc, i, frame=None):
    """Object i's 4x4 placement at a frame (the opening frame by default) with its translation in millimetres."""
    M = np.array(sc.matrix(i, frame), dtype=np.float64)
    M[:3, 3] *= mm_scale(sc)
    return M


def into(parts, M_obj):
    """[(shape, material)] built in the world (mm) moved into the frame of an object placed at M_obj (mm)."""
    return place(parts, np.linalg.inv(M_obj))


def index(sc):
    """Object indexes by name (the first of each name)."""
    out = {}
    for i, o in enumerate(sc.objects):
        out.setdefault(o.get('name'), i)
    return out


def at(sc, name, frame=None):
    """An object's position in millimetres at a frame."""
    return pose_mm(sc, index(sc)[name], frame)[:3, 3].copy()


def publish(key, files, unit=1.0, extra=None):
    """Export each model and merge the example's manifest entry. files: {slug: (nodes, entry fields)}, where the
    fields say how the runtime hangs the file ({'object': name} to replace a scene object, {'covers': [...]} for an
    added model). extra: more entry keys, such as 'hide'. Returns the total size in bytes."""
    from gltf import export, write_manifest
    replace, add, total = [], [], 0
    for slug, (nodes, fields) in files.items():
        rel = f'{key.replace("/", "-")}/{slug}.glb'
        total += export(rel, nodes, unit=unit)
        e = {**({'object': fields['object']} if 'object' in fields else {}), 'file': rel,
             **{k: v for k, v in fields.items() if k != 'object'}}
        (replace if 'object' in fields else add).append(e)
    entry = {}
    if replace:
        entry['replace'] = replace
    if add:
        entry['add'] = add
    entry.update(extra or {})
    write_manifest(key, entry)
    print(f'[cad] {key} {total / 1024:.0f} KB in {len(files)} files')
    return total
