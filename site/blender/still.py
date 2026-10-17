"""Renders one scene run's gallery still in Cycles from what still-scene.mjs laid out: the run's own objects with
their live colors, the context models from site/public/models/manifest.json in place of the primitives they
stand for, a studio HDRI for light, soft shadows, a slight depth of field and the site's own background.

    blender -b --factory-startup -P site/blender/still.py -- <layout base> <example key> <out.png> [samples] [card.webp]
"""
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import bmesh  # noqa: E402
import bpy  # noqa: E402
import kit  # noqa: E402
import numpy as np  # noqa: E402
import sources  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

BG = (14 / 255, 15 / 255, 17 / 255)
EYE = [0.0, 0.0, 0.0]     # the camera of the view being built, for shells drawn cut open
PLAIN = os.path.join(sources.CACHE, 'plain')


def lin(c):
    """sRGB bytes or 0 to 1 values to linear."""
    c = [x / 255 if max(c) > 1 else x for x in c]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


# ---------------------------------------------------------------- materials for the run's own objects

_mats = {}


def part_material(kind, rgb=None, opacity=1.0, cut=False, glow=0.0, prop=False, emissive=False):
    """The run's part in its field colors (vertex colors when rgb is None), a prop in its own color."""
    key = (kind, tuple(rgb) if rgb else None, round(opacity, 3), cut, round(glow, 3), prop, emissive)
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(f'{kind}')
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    out = nt.nodes['Material Output']
    if rgb is None:
        ca = nt.nodes.new('ShaderNodeVertexColor')
        ca.layer_name = 'Col'
        color = ca.outputs['Color']
        nt.links.new(color, b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = (*lin(rgb), 1)
        color = None
    b.inputs['Roughness'].default_value = 0.62 if prop else 0.42
    b.inputs['Coat Weight'].default_value = 0.0 if prop else 0.25
    b.inputs['Coat Roughness'].default_value = 0.2
    if emissive or glow:
        if color is not None:
            nt.links.new(color, b.inputs['Emission Color'])
        else:
            b.inputs['Emission Color'].default_value = (*lin(rgb), 1)
        b.inputs['Emission Strength'].default_value = 1.0 if emissive else 2.5 * glow
    shader = b.outputs['BSDF']
    if cut:
        # a shell drawn cut open shows only its far side, as the page draws it: the faces turned toward the camera
        # let everything through, for light as well, so what the shell holds is lit and not shut in the dark
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        eye = nt.nodes.new('ShaderNodeCombineXYZ')
        for i, k in enumerate('XYZ'):
            eye.inputs[k].default_value = EYE[i]
        to_eye = nt.nodes.new('ShaderNodeVectorMath')
        to_eye.operation = 'SUBTRACT'
        nt.links.new(eye.outputs['Vector'], to_eye.inputs[0])
        nt.links.new(geo.outputs['Position'], to_eye.inputs[1])
        facing = nt.nodes.new('ShaderNodeVectorMath')
        facing.operation = 'DOT_PRODUCT'
        nt.links.new(to_eye.outputs['Vector'], facing.inputs[0])
        nt.links.new(geo.outputs['True Normal'], facing.inputs[1])
        # Cycles turns the true normal toward each ray, so turn it back to the face's own side first
        side = nt.nodes.new('ShaderNodeMath')
        side.operation = 'MULTIPLY_ADD'
        nt.links.new(geo.outputs['Backfacing'], side.inputs[0])
        side.inputs[1].default_value = -2.0
        side.inputs[2].default_value = 1.0
        own = nt.nodes.new('ShaderNodeMath')
        own.operation = 'MULTIPLY'
        nt.links.new(facing.outputs['Value'], own.inputs[0])
        nt.links.new(side.outputs['Value'], own.inputs[1])
        away = nt.nodes.new('ShaderNodeMath')
        away.operation = 'LESS_THAN'
        nt.links.new(own.outputs['Value'], away.inputs[0])
        away.inputs[1].default_value = 0.0
        tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        mix = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(away.outputs['Value'], mix.inputs['Fac'])
        nt.links.new(tr.outputs['BSDF'], mix.inputs[1])
        nt.links.new(shader, mix.inputs[2])
        shader = mix.outputs['Shader']
    elif opacity < 1:
        b.inputs['Alpha'].default_value = max(opacity, 0.15)
    nt.links.new(shader, out.inputs['Surface'])
    _mats[key] = m
    return m


def mesh_from(name, P, T, C=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(P))
    me.vertices.foreach_set('co', P.reshape(-1))
    me.loops.add(len(T) * 3)
    me.polygons.add(len(T))
    me.loops.foreach_set('vertex_index', T.reshape(-1).astype(np.int32))
    me.polygons.foreach_set('loop_start', np.arange(0, len(T) * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if C is not None:
        ca = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
        rgba = np.ones((len(P), 4), dtype=np.float32)
        rgba[:, :3] = _lin_np(C.reshape(-1, 3))
        ca.data.foreach_set('color', rgba.reshape(-1))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if smooth:
        kit.smooth_by_angle(ob, 35)
    return ob


def _lin_np(c):
    c = c.astype(np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def primitive(shape, s):
    bm = bmesh.new()
    if shape == 'box' or shape not in ('sphere', 'ellipsoid', 'cylinder', 'hex', 'capsule', 'cone', 'torus', 'plane'):
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.scale(bm, vec=(s[0], s[1], s[2] if len(s) > 2 else s[1]), verts=bm.verts)
    elif shape in ('sphere', 'ellipsoid'):
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=1.0)
        sc = (s[0], s[0], s[0]) if shape == 'sphere' else tuple(s[:3])
        bmesh.ops.scale(bm, vec=sc, verts=bm.verts)
    elif shape in ('cylinder', 'hex', 'cone'):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=6 if shape == 'hex' else 48, radius1=s[0], radius2=0 if shape == 'cone' else s[0], depth=2 * s[1])
    elif shape == 'capsule':
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=24, radius=s[0])
        for v in bm.verts:
            v.co.z += s[1] if v.co.z > 0 else -s[1]
    elif shape == 'torus':
        bm.free()
        bpy.ops.mesh.primitive_torus_add(major_radius=s[0], minor_radius=s[1], major_segments=72, minor_segments=24)
        return bpy.context.view_layer.objects.active
    elif shape == 'plane':
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
        bmesh.ops.scale(bm, vec=(s[0], s[1], 1), verts=bm.verts)
    me = bpy.data.meshes.new(shape)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(shape, me)
    bpy.context.scene.collection.objects.link(ob)
    if shape in ('box', 'hex'):
        kit.bevel(ob, 0.04 * min(s[:3]) if shape == 'box' else 0.03 * s[0], 2, harden=False)
    elif shape not in ('plane',):
        kit.smooth_by_angle(ob, 40)
    return ob


# ---------------------------------------------------------------- context models

def plain_copy(rel):
    """The committed model with its meshopt compression undone, which Blender's glTF importer cannot read."""
    src = os.path.join(kit.OUT, rel)
    dst = os.path.join(PLAIN, rel)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pack = os.environ.get('GLTFPACK', 'gltfpack')
        subprocess.run([*pack.split(), '-i', src, '-o', dst, '-noq', '-kn', '-km', '-kv'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def load_model(rel, M, tint=None, rgb=None):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=plain_copy(rel))
    new = [o for o in bpy.data.objects if o not in before]
    holder = bpy.data.objects.new(os.path.basename(rel), None)
    bpy.context.scene.collection.objects.link(holder)
    for o in new:
        if o.parent is None:
            o.parent = holder
        if o.type == 'MESH' and tint and rgb is not None:
            for i, m in enumerate(o.data.materials):
                if m and m.name.split('.')[0] in tint:
                    c = m.copy()
                    c.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*lin(rgb), 1)
                    o.data.materials[i] = c
    holder.matrix_world = M
    return holder


# ---------------------------------------------------------------- the scene

def build(lay, view, blob, entry):
    replace = {}
    for e in entry.get('replace', []):
        replace.setdefault(e['object'], []).append(e)
    hidden = set(entry.get('hide', []))
    for e in entry.get('add', []):
        hidden.update(e.get('covers', []))
    R = lay['radius']

    def arr(ref, dt):
        return np.frombuffer(blob, dtype=dt, count=ref['n'], offset=ref['o'])
    for o in view['objects']:
        if o['name'] in hidden or o['kind'] == 'ground' and 'ground' in hidden:
            continue
        M = Matrix(o['matrix']) if o.get('matrix') else Matrix.Identity(4)
        rep = next((e for e in replace.get(o['name'], []) if e.get('index', o['oi']) == o['oi']), None)
        if rep and os.path.exists(os.path.join(kit.OUT, rep['file'])):
            load_model(rep['file'], M, rep.get('tint'), o['color'] if o.get('field') else None)
            continue
        a = o.get('opacity', 1)
        if o['kind'] == 'body':
            ob = primitive(o['shape'], o['size'])
            ob.matrix_world = M
            cut = a < 1 and o['shape'] != 'plane'
            kit.assign(ob, part_material('body', o['color'], a, cut, o.get('glow', 0), o['prop']))
        elif o['kind'] == 'mesh':
            ob = mesh_from(o['name'] or 'mesh', arr(o['P'], np.float32).reshape(-1, 3), arr(o['T'], np.uint32).reshape(-1, 3),
                           arr(o['C'], np.uint8), smooth=not o.get('flat'))
            ob.matrix_world = M
            cut = a < 1 and (o.get('closed') or a < 0.5)
            kit.assign(ob, part_material('mesh', None, a, cut, o.get('glow', 0), o['prop']))
        elif o['kind'] == 'lines':
            ob = mesh_from('line', arr(o['P'], np.float32).reshape(-1, 3), arr(o['T'], np.uint32).reshape(-1, 3), arr(o['C'], np.uint8))
            kit.assign(ob, part_material('line', None, 1, False, 0, False, emissive=True))
        elif o['kind'] in ('points', 'atoms'):
            X = arr(o['X'], np.float32).reshape(-1, 4)
            C = arr(o['C'], np.uint8).reshape(-1, 3)
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
            base = np.array([v.co[:] for v in bm.verts])
            tris = np.array([[v.index for v in f.verts] for f in bm.faces])
            bm.free()
            keep = X[:, 3] > 0
            Ps, Ts, Cs = [], [], []
            for i in np.nonzero(keep)[0]:
                Ts.append(tris + len(Ps) * len(base))
                Ps.append(base * X[i, 3] + X[i, :3])
                Cs.append(np.repeat(C[i:i + 1], len(base), 0))
            if Ps:
                ob = mesh_from('atoms', np.concatenate(Ps), np.concatenate(Ts), np.concatenate(Cs))
                kit.assign(ob, part_material('atoms', None))
            if o.get('bonds'):
                B = arr(o['bonds'], np.uint32).reshape(-1, 2)
                for p, q in B:
                    A, Bp = Vector(X[p, :3].tolist()), Vector(X[q, :3].tolist())
                    d = Bp - A
                    cy = primitive('cylinder', [o['bondR'], d.length / 2])
                    cy.matrix_world = kit.frame_matrix((A + Bp) / 2, d.orthogonal(), None, d)
                    kit.assign(cy, part_material('bond', (150, 154, 160)))
        elif o['kind'] == 'arrows':
            for ar in o['arrows']:
                v = Vector(ar['v'])
                L, r = v.length, ar['r']
                head = min(L * 0.45, r * 7)
                body = max(L - head, L * 0.01)
                sh = primitive('cylinder', [r, body / 2])
                cn = primitive('cone', [2.6 * r, head / 2])
                sh.matrix_world = kit.frame_matrix(Vector(ar['o']) + v.normalized() * body / 2, v.orthogonal(), None, v)
                cn.matrix_world = kit.frame_matrix(Vector(ar['o']) + v.normalized() * (body + head / 2), v.orthogonal(), None, v)
                for x in (sh, cn):
                    kit.assign(x, part_material('arrow', o['color'], emissive=True))
        elif o['kind'] == 'ground':
            s = o['size']
            g = primitive('plane', [s / 2, s / 2])
            g.location = (o['center'][0], o['center'][1], o['z'] - R * 2e-4)
            m = part_material('ground', o['color'], prop=True)
            m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.55
            kit.assign(g, m)
    for e in entry.get('add', []):
        if os.path.exists(os.path.join(kit.OUT, e['file'])):
            load_model(e['file'], Matrix.Identity(4))


def shadow_catcher(lay, view):
    """An invisible floor under everything that only takes shadows, when nothing else is under the subject."""
    if lay['follow'] or any(o['kind'] == 'ground' for o in view['objects']):
        return
    zmin = math.inf
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for c in ob.bound_box:
            zmin = min(zmin, (ob.matrix_world @ Vector(c)).z)
    R = lay['radius']
    p = primitive('plane', [R * 30, R * 30])
    c = [(a + b) / 2 for a, b in zip(lay['subject']['lo'], lay['subject']['hi'])]
    p.location = (c[0], c[1], zmin - R * 1e-3)
    p.is_shadow_catcher = True


def light(lay, cam):
    """A soft key from up and to the right of the camera, sized and powered for the scene's scale."""
    R = lay['radius']
    eye, tgt = Vector(cam['eye']), Vector(cam['target'])
    z = (eye - tgt).normalized()
    x = Vector((0, 0, 1)).cross(z).normalized()
    pos = tgt + (z * 0.8 + x * 0.9 + Vector((0, 0, 1)) * 0.9).normalized() * R * 5
    ld = bpy.data.lights.new('key', 'AREA')
    ld.shape = 'DISK'
    ld.size = R * 3
    ld.energy = 3.0 * (R * 5) ** 2 * 4 * math.pi / 4
    lo = bpy.data.objects.new('key', ld)
    bpy.context.scene.collection.objects.link(lo)
    d = (tgt - pos).normalized()
    lo.matrix_world = kit.frame_matrix(pos, d.orthogonal(), None, -d)


def hdri_rotation(path, cam):
    """Turn the HDRI so its brightest softbox sits up and to the right of the camera."""
    im = bpy.data.images.load(path, check_existing=True)
    w, h = im.size
    px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::8, ::8, :3].sum(-1)
    y, x = np.unravel_index(np.argmax(px), px.shape)
    u = (x * 8 + 4) / w
    # Blender's equirectangular lookup: u = atan2(dir.y, -dir.x) / 2pi + 0.5
    bright_az = (u - 0.5) * 2 * math.pi
    eye, tgt = Vector(cam['eye']), Vector(cam['target'])
    v = eye - tgt
    cam_az = math.atan2(v.y, -v.x)
    return cam_az - math.radians(50) - bright_az


def camera(lay, cam, dof=True):
    R = lay['radius']
    s = lay['subject']
    sr = math.dist(s['lo'], s['hi']) / 2
    near = (Vector(cam['eye']) - Vector(cam['target'])).length / 200
    ob = kit.camera(cam['eye'], cam['target'], cam['fov'], cam['aspect'], clip=(near, max(R * 400, near * 1e5)))
    if dof:
        cd = ob.data
        dist = (Vector(cam['eye']) - Vector(cam['target'])).length
        cd.dof.use_dof = True
        cd.dof.focus_distance = dist
        f = cd.lens
        sensor = cd.sensor_height if cd.sensor_fit == 'VERTICAL' else cd.sensor_width
        # blur of about a third of a percent of the frame height at the far edge of the subject
        dmm, smm, rmm = (dist + sr) * 1000, dist * 1000, sr * 1000
        A = 0.0035 * sensor * dmm * (smm - f) / (f * rmm)
        cd.dof.aperture_fstop = max(f / max(A, 1e-6), 0.5)
    return ob


def render_view(lay, view, blob, entry, out, w, h, samples, inset=False):
    kit.reset(1.0)
    _mats.clear()
    EYE[:] = view['camera']['eye']
    build(lay, view, blob, entry)
    if not inset:
        shadow_catcher(lay, view)
    hdr = sources.hdri('studio_small_08')
    kit.world_hdri(hdr, 0.9, background=lin(BG) if not inset else lin((20, 22, 26)), rot=hdri_rotation(hdr, view['camera']))
    light(lay, view['camera'])
    camera(lay, view['camera'], dof=not inset)
    sc = bpy.context.scene
    sc.view_settings.exposure = math.log2(1.3)
    # leave part of the machine free for whatever else runs beside the batch
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = max(2, (os.cpu_count() or 4) * 2 // 3)
    kit.render(out, w, h, samples=samples)


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    base, key, out = args[:3]
    samples = int(args[3]) if len(args) > 3 else 64
    lay = json.load(open(base + '.json'))
    blob = open(base + '.bin', 'rb').read()
    man = json.load(open(os.path.join(kit.OUT, 'manifest.json')))
    entry = man['examples'].get(key, {})
    W, H = lay['w'] * 3 // 2, lay['h'] * 3 // 2      # one and a half times the card, then scaled down
    render_view(lay, lay['main'], blob, entry, out, W, H, samples)
    if lay.get('inset'):
        x, y, w, h = [v * 3 // 2 for v in lay['inset']['rect']]
        tmp = out.replace('.png', '.inset.png')
        render_view(lay, lay['inset'], blob, entry, tmp, w, h, max(16, samples // 2), inset=True)
        a = bpy.data.images.load(out)
        b = bpy.data.images.load(tmp)
        A = np.array(a.pixels[:], dtype=np.float32).reshape(H, W, 4)
        B = np.array(b.pixels[:], dtype=np.float32).reshape(h, w, 4)
        y0 = H - y - h            # Blender's pixels start at the bottom row
        A[y0:y0 + h, x:x + w] = B
        line = np.array([0.24, 0.25, 0.28, 1.0], dtype=np.float32)   # the inset's hairline, as the page draws it
        A[y0:y0 + 2, x:x + w] = line
        A[y0 + h - 2:y0 + h, x:x + w] = line
        A[y0:y0 + h, x:x + 2] = line
        A[y0:y0 + h, x + w - 2:x + w] = line
        a.pixels[:] = A.reshape(-1)
        a.filepath_raw = out
        a.file_format = 'PNG'
        a.save()
        os.remove(tmp)
    if len(args) > 4:
        # the card: half size, as WebP, which loads fast in the gallery
        im = bpy.data.images.load(out)
        im.scale(lay['w'], lay['h'])
        im.file_format = 'WEBP'
        im.save(filepath=args[4], quality=82)


if __name__ == '__main__':
    main()
