"""How the context models look: tileable detail textures, glTF metallic roughness materials, the shared
material set, and ambient occlusion baked onto a second uv set.
"""
import math
import os

import bpy
import numpy as np

import sources
from shapes import select

TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache', 'tex')


# ---------------------------------------------------------------- textures
# Tileable detail maps made here from band limited noise (filtered in the frequency domain, so they wrap), seeded so a
# rebuild makes the same files. They are small and shared: a cast skin, paint orange peel, brushed and turned grain,
# rubber grain, and roughness maps that vary a material's sheen the way handling and dust do.

def _noise(n, lo, hi, seed, ax=1.0, ay=1.0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k)
    r = np.sqrt((kx * ax) ** 2 + (ky * ay) ** 2)
    mask = np.exp(-((np.log(np.maximum(r, 1e-6)) - math.log(math.sqrt(lo * hi))) / (0.5 * math.log(hi / lo) + 1e-6)) ** 2)
    mask[0, 0] = 0
    h = np.real(np.fft.ifft2(F * mask))
    return (h - h.mean()) / (h.std() + 1e-9)


def _save(name, rgb):
    os.makedirs(TEX, exist_ok=True)
    n = rgb.shape[0]
    im = bpy.data.images.new(name, n, n, alpha=False)
    a = np.ones((n, n, 4), dtype=np.float32)
    a[..., :3] = rgb
    im.pixels[:] = a[::-1].reshape(-1)
    im.filepath_raw = os.path.join(TEX, name)
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def _normal_from(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    nrm = np.stack([-dx, dy, np.ones_like(h)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    return nrm * 0.5 + 0.5


NORMALS = {
    # name: (size, band low, band high, x stretch, y stretch, strength, seed)
    'cast_n.png': (256, 14, 70, 1, 1, 0.9, 11),
    'peel_n.png': (256, 6, 22, 1, 1, 0.35, 12),
    'brushed_n.png': (256, 20, 110, 0.04, 1, 0.5, 13),
    'rubber_n.png': (256, 40, 120, 1, 1, 0.6, 14),
    'concrete_n.png': (512, 8, 200, 1, 1, 0.7, 15),
    'fine_n.png': (256, 50, 128, 1, 1, 0.25, 16),
}


def _tread(n=256):
    """Diamond tread plate: raised lozenges in two crossing directions on a grid, four to a tile."""
    y, x = np.mgrid[0:n, 0:n] / n
    h = np.zeros((n, n))
    for cx in range(4):
        for cy in range(4):
            px, py = (cx + 0.5) / 4, (cy + 0.5) / 4
            ang = math.radians(45 if (cx + cy) % 2 else -45)
            dx = (x - px + 0.5) % 1 - 0.5
            dy = (y - py + 0.5) % 1 - 0.5
            u = dx * math.cos(ang) + dy * math.sin(ang)
            v = -dx * math.sin(ang) + dy * math.cos(ang)
            d = (u / 0.085) ** 2 + (v / 0.018) ** 2
            h = np.maximum(h, np.clip(1 - d, 0, 1) ** 0.5)
    return h + 0.05 * _noise(n, 30, 100, 31)


def tex_normal(name):
    if name == 'tread_n.png' and not os.path.exists(os.path.join(TEX, name)):
        _save(name, _normal_from(_tread(), 6.0))
    if not os.path.exists(os.path.join(TEX, name)):
        n, lo, hi, ax, ay, st, seed = NORMALS[name]
        h = _noise(n, lo, hi, seed, ax, ay)
        if name == 'cast_n.png':
            h = np.sign(h) * np.abs(h) ** 0.7 + 0.35 * _noise(n, 4, 12, seed + 100)
        _save(name, _normal_from(h, st))
    return name


def tex_rough(lo, hi, metal, seed=21):
    """A metallic roughness map: roughness from lo to hi in large soft smudges and fine grain, metal in blue."""
    name = f'r_{lo:.2f}_{hi:.2f}_m{metal:.0f}.png'
    if not os.path.exists(os.path.join(TEX, name)):
        n = 256
        big = _noise(n, 1.5, 5, seed)
        fine = _noise(n, 30, 110, seed + 1)
        x = np.clip(0.5 + 0.12 * big + 0.16 * fine, 0, 1)
        r = lo + (hi - lo) * x
        rgb = np.stack([np.ones_like(r), r, np.full_like(r, metal)], -1)
        _save(name, rgb)
    return name


# ---------------------------------------------------------------- materials

def image(name, path=None, size=None, color=False):
    if path:
        im = bpy.data.images.load(path, check_existing=True)
    else:
        im = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    im.colorspace_settings.name = 'sRGB' if color else 'Non-Color'
    return im


def material(name, color=(0.5, 0.5, 0.5), metal=0.0, rough=0.5, normal=None, normal_strength=1.0, rough_var=None,
             uv_scale=1.0, coat=0.0, coat_rough=0.05, transmission=0.0, ior=1.5, alpha=1.0, base_map=None, emissive=None,
             pbr=None, tone=1.0):
    """A glTF metallic roughness material. normal names a tileable detail map and rough_var a roughness range,
    both made in looks.py and laid on uv0; uv_scale is how many tiles a model unit holds after the cube projection."""
    rough_tex = None
    if pbr:
        # a CC0 scanned material from ambientCG (sources.py): its color, normal and roughness
        c, normal, rough_tex = sources.ambientcg(pbr, metal=metal, tone=tone)
        base_map = os.path.join(TEX, c)
    m = bpy.data.materials.new(name)
    m['base'] = name
    if pbr:
        m['pbr'] = 1
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    b.inputs['IOR'].default_value = ior
    if coat:
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = coat_rough
    if transmission:
        b.inputs['Transmission Weight'].default_value = transmission
    if alpha < 1:
        b.inputs['Alpha'].default_value = alpha
        m.surface_render_method = 'BLENDED' if hasattr(m, 'surface_render_method') else None
    if emissive:
        b.inputs['Emission Color'].default_value = (*emissive[0], 1)
        b.inputs['Emission Strength'].default_value = emissive[1]
    uv = None

    def uvnode():
        nonlocal uv
        if uv is None:
            uv = nt.nodes.new('ShaderNodeUVMap')
            uv.uv_map = 'UVMap'
        return uv
    if normal:
        tex_normal(normal)
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = image(normal, os.path.join(TEX, normal))
        t.interpolation = 'Linear'
        nt.links.new(uvnode().outputs['UV'], t.inputs['Vector'])
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nm.uv_map = 'UVMap'
        nm.inputs['Strength'].default_value = normal_strength
        nt.links.new(t.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    if rough_var or rough_tex:
        # the glTF exporter reads roughness from the green channel of a metallic roughness texture, metal from blue
        rough_map = rough_tex or tex_rough(rough_var[0], rough_var[1], metal)
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = image(rough_map, os.path.join(TEX, rough_map))
        nt.links.new(uvnode().outputs['UV'], t.inputs['Vector'])
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(t.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], b.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'], b.inputs['Metallic'])
    if base_map:
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = image(os.path.basename(base_map), base_map, color=True)
        nt.links.new(uvnode().outputs['UV'], t.inputs['Vector'])
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    m['uv_scale'] = uv_scale
    return m


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    return ob


def _gltf_output(nt):
    """The node group the glTF exporter reads the occlusion texture from."""
    g = bpy.data.node_groups.get('glTF Material Output')
    if g is None:
        g = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
        g.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
        g.nodes.new('NodeGroupInput')
    n = nt.nodes.new('ShaderNodeGroup')
    n.node_tree = g
    return n


# ---------------------------------------------------------------- uvs, ambient occlusion

def uv_tiles(ob, density=1.0):
    """uv0 for the tileable detail textures: a cube projection, density tiles per model unit."""
    select(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.cube_project(cube_size=1.0 / max(density, 1e-9), correct_aspect=False, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def bake_ao(ob, size=512, samples=48, distance=None):
    """Bake ambient occlusion onto a second uv set and wire it in as the glTF occlusion texture."""
    me = ob.data
    if 'UVMap' not in me.uv_layers:
        me.uv_layers.new(name='UVMap')
    ao_uv = me.uv_layers.get('AO') or me.uv_layers.new(name='AO')
    me.uv_layers.active = ao_uv
    select(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(50), island_margin=0.02, area_weight=1.0, scale_to_bounds=True)
    bpy.ops.uv.pack_islands(margin=0.01, rotate=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    im = bpy.data.images.new(f'{ob.name}_ao', size, size, alpha=False)
    im.colorspace_settings.name = 'Non-Color'
    nodes = []
    for m in me.materials:
        nt = m.node_tree
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = im
        u = nt.nodes.new('ShaderNodeUVMap')
        u.uv_map = 'AO'
        nt.links.new(u.outputs['UV'], t.inputs['Vector'])
        nt.nodes.active = t
        nodes.append((m, t, u))
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.render.bake.margin = 6
    if distance:
        sc.world = sc.world or bpy.data.worlds.new('w')
        sc.world.light_settings.distance = distance
    bpy.ops.object.bake(type='AO')
    # soften the noise a little and lift the darkest creases, which a light probe would never make black
    a = np.array(im.pixels[:], dtype=np.float32).reshape(size, size, 4)
    g = a[..., 0]
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k /= k.sum()
    for ax in (0, 1):
        g = sum(np.roll(g, s - 2, axis=ax) * k[s] for s in range(5))
    g = 0.18 + 0.82 * np.clip(g, 0, 1)
    a[..., 0] = a[..., 1] = a[..., 2] = g
    im.pixels[:] = a.reshape(-1)
    im.filepath_raw = os.path.join(bpy.app.tempdir, f'{ob.name}_ao.png')
    im.file_format = 'PNG'
    im.save()
    for m, t, u in nodes:
        out = _gltf_output(m.node_tree)
        m.node_tree.links.new(t.outputs['Color'], out.inputs['Occlusion'])
    me.uv_layers.active = me.uv_layers['UVMap']
    me.uv_layers['UVMap'].active_render = True
    return im


# ---------------------------------------------------------------- the material set every model draws from
# Names are what runtime reads (site/public/models/manifest.json): "paint" is the surface it may tint with an
# object's live color, every other name keeps its own look.

def mats(unit=1.0):
    """The shared materials, with their detail textures sized for a scene in metres (unit 1) or mm (unit 1000)."""
    d = 1.0 / unit
    return {
        'paint': lambda: material('paint', (0.03, 0.03, 0.032), 0.0, 0.42, 'peel_n.png', 0.25, (0.32, 0.5), 40 * d),
        'paint_black': lambda: material('paint_black', (0.022, 0.022, 0.024), 0.0, 0.5, 'peel_n.png', 0.2, (0.42, 0.6), 40 * d),
        'steel_bare': lambda: material('steel_bare', (0.62, 0.62, 0.62), 1.0, 0.3, None, 0.4, (0.22, 0.38), 30 * d),
        'steel_zinc': lambda: material('steel_zinc', (0.74, 0.75, 0.77), 1.0, 0.34, None, 0.5, (0.26, 0.44), 60 * d),
        'steel_dark': lambda: material('steel_dark', (0.16, 0.16, 0.17), 1.0, 0.4, None, 0.5, (0.32, 0.5), 60 * d),
        'aluminum_cast': lambda: material('aluminum_cast', (0.40, 0.41, 0.42), 1.0, 0.6, 'cast_n.png', 1.0, (0.52, 0.7), 25 * d),
        'aluminum_machined': lambda: material('aluminum_machined', (0.8, 0.81, 0.82), 1.0, 0.24, None, 0.12, (0.2, 0.3), 90 * d),
        'plastic_black': lambda: material('plastic_black', (0.03, 0.03, 0.032), 0.0, 0.45, None, 0.6, (0.38, 0.55), 60 * d),
        'rubber': lambda: material('rubber', (0.018, 0.018, 0.018), 0.0, 0.72, None, 0.7, (0.62, 0.85), 80 * d),
        'paint_white': lambda: material('paint_white', (0.82, 0.82, 0.8), 0.0, 0.22, 'peel_n.png', 0.08, (0.16, 0.3), 30 * d, coat=1.0, coat_rough=0.04),
        'paint_red': lambda: material('paint_red', (0.62, 0.07, 0.04), 0.0, 0.22, 'peel_n.png', 0.08, (0.16, 0.3), 30 * d, coat=1.0, coat_rough=0.04),
        'paint_gloss_black': lambda: material('paint_gloss_black', (0.018, 0.018, 0.02), 0.0, 0.25, 'peel_n.png', 0.1, (0.18, 0.32), 30 * d, coat=1.0, coat_rough=0.05),
        'decal': lambda: material('decal', (0.015, 0.015, 0.016), 0.0, 0.32),
        'graphite': lambda: material('graphite', (0.05, 0.05, 0.055), 0.0, 0.62, 'cast_n.png', 0.4, (0.55, 0.72), 60 * d),
        'aluminum_anodized': lambda: material('aluminum_anodized', (0.05, 0.05, 0.055), 1.0, 0.35, None, 0.4, (0.3, 0.42), 30 * d),
        'nylon': lambda: material('nylon', (0.72, 0.71, 0.66), 0.0, 0.45, None, 0.4, (0.4, 0.55), 60 * d),
        'tread_steel': lambda: material('tread_steel', (0.5, 0.5, 0.5), 1.0, 0.45, 'tread_n.png', 1.0, (0.38, 0.6), 6 * d),
        'steel_mill': lambda: material('steel_mill', (0.2, 0.205, 0.215), 1.0, 0.55, 'fine_n.png', 0.4, (0.48, 0.62), 40 * d),
        'steel_stainless': lambda: material('steel_stainless', (0.66, 0.66, 0.67), 1.0, 0.26, 'brushed_n.png', 0.3, (0.2, 0.32), 30 * d),
        'steel_black_oxide': lambda: material('steel_black_oxide', (0.07, 0.07, 0.075), 1.0, 0.4, None, 0.4, (0.32, 0.48), 60 * d),
        'steel_chrome': lambda: material('steel_chrome', (0.85, 0.85, 0.86), 1.0, 0.08),
        'concrete': lambda: material('concrete', (1, 1, 1), 0.0, 0.85, None, 0.8, None, 1.6 * d, pbr='Concrete034', tone=0.55),
        'wood': lambda: material('wood', (1, 1, 1), 0.0, 0.6, None, 0.6, None, 5.0 * d, pbr='Wood049', tone=0.8),
        'rock': lambda: material('rock', (1, 1, 1), 0.0, 0.8, None, 1.0, None, 1.0 * d, pbr='Rock030', tone=1.5),
        'label': lambda: material('label', (0.78, 0.78, 0.76), 0.0, 0.42),
        'plastic_white': lambda: material('plastic_white', (0.8, 0.8, 0.78), 0.0, 0.38, None, 0.5, (0.32, 0.46), 60 * d),
        'aluminum_anodized_gold': lambda: material('aluminum_anodized_gold', (0.75, 0.6, 0.32), 1.0, 0.32, None, 0.4, (0.26, 0.4), 30 * d),
        'paint_wall': lambda: material('paint_wall', (0.36, 0.36, 0.35), 0.0, 0.82, 'fine_n.png', 0.3, (0.75, 0.9), 20 * d),
        'acrylic': lambda: material('acrylic', (0.92, 0.95, 0.96), 0.0, 0.04, transmission=1.0, ior=1.49),
        'cardboard': lambda: material('cardboard', (0.45, 0.32, 0.19), 0.0, 0.85, 'fine_n.png', 0.4, (0.78, 0.92), 30 * d),
        'tape': lambda: material('tape', (0.55, 0.42, 0.25), 0.0, 0.3),
        'cast_iron': lambda: material('cast_iron', (0.035, 0.035, 0.037), 0.0, 0.6, 'cast_n.png', 0.8, (0.5, 0.7), 40 * d),
        'pei': lambda: material('pei', (0.42, 0.27, 0.09), 0.25, 0.55, 'cast_n.png', 0.6, (0.45, 0.65), 50 * d),
        'brass': lambda: material('brass', (0.86, 0.66, 0.36), 1.0, 0.28, None, 0.4, (0.22, 0.38), 40 * d),
        'enamel_white': lambda: material('enamel_white', (0.85, 0.85, 0.83), 0.0, 0.4),
        'concrete_polished': lambda: material('concrete_polished', (1, 1, 1), 0.0, 0.35, None, 0.5, None, 1.6 * d, pbr='Concrete034', tone=0.7),
        'glass': lambda: material('glass', (0.86, 0.93, 0.9), 0.0, 0.02, transmission=1.0, ior=1.52),
        'frit': lambda: material('frit', (0.012, 0.012, 0.012), 0.0, 0.32),
    }


def finish(ob, unit=1.0, ao=512, ao_distance=None):
    """uv0 for the detail textures, then ambient occlusion baked on uv1. Each model gets its own copies of the
    materials, since the occlusion map is part of the material."""
    for i, m in enumerate(ob.data.materials):
        if m:
            ob.data.materials[i] = m.copy()
    dens = max((m.get('uv_scale', 1.0) for m in ob.data.materials if m), default=1.0)
    uv_tiles(ob, dens)
    if ao and any(m.get('pbr') for m in ob.data.materials if m):
        ao = 512   # the same size as the scanned maps, so the exporter can pack occlusion with their roughness
    if ao:
        bake_ao(ob, ao, distance=ao_distance)
    return ob
