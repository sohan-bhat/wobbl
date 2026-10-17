"""Writes CAD models as the glTF files the example pages load: named nodes, the shared materials (looks.py), meshopt
compressed by gltfpack, plus the manifest entry that tells the runtime which scene object each file dresses.

Models are built Z up in millimetres, in the frame of the scene object they stand for. export turns them to glTF's
Y up (the runtime turns the root back by +90 degrees about x, so gltf = (x, z, -y)) and scales them by unit: 1 for
a scene in millimetres, 0.001 for a scene in metres.

    from gltf import Node, export
    export('circuits-led/breadboard.glb', [Node('breadboard', [(shape, 'plastic_white')])])
"""
import json
import os
import struct
import subprocess
import tempfile

import numpy as np

import looks
import mesh as meshing

SITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(SITE, 'public', 'models')
_C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)   # Z up to Y up


class Node:
    """A named node: parts are (shape, material name) pairs, or (('mesh', P, N, T), material) for ready triangles;
    loc is an optional 4x4 placement in the parent's frame (Z up, millimetres)."""

    def __init__(self, name, parts=(), children=(), loc=None, tol=None, ang=None):
        self.name, self.parts, self.children, self.loc = name, list(parts), list(children), loc
        self.tol, self.ang = tol, ang


def _gltfpack():
    env = os.environ.get('GLTFPACK')
    if env:
        return env.split()
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, '..', 'node_modules', 'gltfpack', 'cli.js'),
              os.path.join(tempfile.gettempdir(), 'gltfpack', 'node_modules', 'gltfpack', 'cli.js')]:
        if os.path.exists(p):
            return ['node', p]
    return ['gltfpack']


def triangles(shape, tol, ang):
    if isinstance(shape, tuple) and shape and shape[0] == 'mesh':
        return shape[1], shape[2], np.asarray(shape[3], dtype=np.int64)
    return meshing.tessellate(shape, tol, ang)


def export(rel, nodes, unit=1.0, tol=0.02, ang=0.2, quiet=False):
    """Write site/public/models/<rel> from a list of Nodes and return its size in bytes."""
    nodes = nodes if isinstance(nodes, (list, tuple)) else [nodes]
    js = {'asset': {'version': '2.0', 'generator': 'wobbl site/cad (build123d, OpenCASCADE)'}, 'scene': 0,
          'scenes': [{'nodes': []}], 'nodes': [], 'meshes': [], 'materials': [], 'accessors': [],
          'bufferViews': [], 'buffers': []}
    blob = bytearray()
    mats, texs, used = {}, {}, set()

    def view(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        bv = {'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)}
        if target:
            bv['target'] = target
        blob.extend(data)
        js['bufferViews'].append(bv)
        return len(js['bufferViews']) - 1

    def accessor(arr, kind, ctype, target, minmax=False):
        a = {'bufferView': view(arr.tobytes(), target), 'componentType': ctype, 'count': len(arr), 'type': kind}
        if minmax:
            a['min'] = arr.min(0).tolist()
            a['max'] = arr.max(0).tolist()
        js['accessors'].append(a)
        return len(js['accessors']) - 1

    def tex_index(name):
        if name not in texs:
            js.setdefault('images', []).append({'name': name, 'mimeType': 'image/webp', 'bufferView': view(looks.webp(name))})
            js.setdefault('samplers', [{'magFilter': 9729, 'minFilter': 9987}])
            js.setdefault('textures', []).append({'sampler': 0, 'extensions': {'EXT_texture_webp': {'source': len(js['images']) - 1}}})
            texs[name] = len(js['textures']) - 1
            used.add('EXT_texture_webp')
        return texs[name]

    def material(name):
        if name not in mats:
            m = looks.gltf_material(name, tex_index)
            used.update(m.get('extensions', {}).keys())
            js['materials'].append(m)
            mats[name] = len(js['materials']) - 1
        return mats[name]

    def add(node):
        groups = {}
        for shape, mname in node.parts:
            P, N, T = triangles(shape, node.tol or tol, node.ang or ang)
            if not len(T):
                continue
            g = groups.setdefault(mname, [[], [], [], 0])
            g[0].append(P)
            g[1].append(N)
            g[2].append(T + g[3])
            g[3] += len(P)
        out = {'name': node.name}
        prims = []
        for mname, (Ps, Ns, Ts, _) in groups.items():
            P, N, T = np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Ts)
            P, N, UV, T = meshing.box_uv(P, N, T, looks.MATERIALS[mname]['tiles'] / 1000.0)
            Pg = (P @ _C.T * unit).astype(np.float32)
            Ng = (N @ _C.T).astype(np.float32)
            UV = UV.astype(np.float32)
            UV[:, 1] = -UV[:, 1]
            prims.append({'attributes': {'POSITION': accessor(Pg, 'VEC3', 5126, 34962, True),
                                         'NORMAL': accessor(Ng, 'VEC3', 5126, 34962),
                                         'TEXCOORD_0': accessor(UV, 'VEC2', 5126, 34962)},
                          'indices': accessor(T.astype(np.uint32).reshape(-1), 'SCALAR', 5125, 34963),
                          'material': material(mname)})
        if prims:
            js['meshes'].append({'name': node.name, 'primitives': prims})
            out['mesh'] = len(js['meshes']) - 1
        if node.loc is not None:
            L = np.asarray(node.loc, dtype=np.float64)
            G = np.eye(4)
            G[:3, :3] = _C @ L[:3, :3] @ _C.T
            G[:3, 3] = _C @ L[:3, 3] * unit
            out['matrix'] = G.T.reshape(-1).tolist()
        js['nodes'].append(out)
        me = len(js['nodes']) - 1
        kids = [add(c) for c in node.children]
        if kids:
            js['nodes'][me]['children'] = kids
        return me

    for n in nodes:
        js['scenes'][0]['nodes'].append(add(n))
    if used:
        js['extensionsUsed'] = sorted(used)
    if 'EXT_texture_webp' in used:
        js['extensionsRequired'] = ['EXT_texture_webp']
    while len(blob) % 4:
        blob.append(0)
    js['buffers'] = [{'byteLength': len(blob)}]
    j = json.dumps(js, separators=(',', ':')).encode()
    j += b' ' * ((4 - len(j) % 4) % 4)
    glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(j) + 8 + len(blob)) + struct.pack('<II', len(j), 0x4E4F534A) + j \
        + struct.pack('<II', len(blob), 0x004E4942) + bytes(blob)
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, 'raw.glb')
        open(raw, 'wb').write(glb)
        subprocess.run([*_gltfpack(), '-i', raw, '-o', path, '-cc', '-km', '-kn', '-kv'], check=True,
                       stdout=subprocess.DEVNULL)
    size = os.path.getsize(path)
    if not quiet:
        tris = sum(js['accessors'][p['indices']]['count'] // 3 for m in js['meshes'] for p in m['primitives'])
        print(f'[cad] {rel} {size / 1024:.0f} KB, {tris} triangles')
    return size


def write_manifest(example, entry):
    """Merge one example's entry into site/public/models/manifest.json, one model to a line, leaving every other
    example's entry as it was."""
    path = os.path.join(OUT, 'manifest.json')
    m = {'version': 1, 'examples': {}}
    if os.path.exists(path):
        m = json.load(open(path))
    if entry is None:
        m['examples'].pop(example, None)
    else:
        m['examples'][example] = entry
    rows = []
    for ex in sorted(m['examples']):
        parts = []
        for key, val in m['examples'][ex].items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                items = ',\n'.join('      ' + json.dumps(v, separators=(', ', ': ')) for v in val)
                parts.append(f'    "{key}": [\n{items}\n    ]')
            else:
                parts.append(f'    "{key}": {json.dumps(val)}')
        rows.append(f'  "{ex}": {{\n' + ',\n'.join(parts) + '\n  }')
    with open(path, 'w') as f:
        f.write('{\n"version": 1,\n"examples": {\n' + ',\n'.join(rows) + '\n}\n}\n')
    json.load(open(path))
