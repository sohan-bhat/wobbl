"""Shared building blocks for the context models: shapes (shapes.py), looks (looks.py), export and previews.

Every model is built in metres or millimetres as its scene stores them, where the object stands at the frame
the example page opens on. to_local() moves it into the object's own frame and export() writes a compressed .glb.
"""
import json
import math
import os
import subprocess

import bpy
import numpy as np
from mathutils import Matrix, Vector

import shapes
from looks import *  # noqa: F401,F403
from shapes import *  # noqa: F401,F403
from shapes import bake_transform, frame_matrix, select

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.abspath(os.path.join(SITE, '..'))
OUT = os.path.join(SITE, 'public', 'models')


def reset(unit=1.0):
    """An empty scene, with model units per metre: 1 for a scene in metres, 1000 for one in millimetres."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shapes.UNIT = unit
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.unit_settings.system = 'METRIC'
    return sc


# ---------------------------------------------------------------- export

def to_local(ob, M_world):
    """Move a model built in the scene's world into the object's own frame (the inverse of its pose)."""
    bake_transform(ob)
    ob.data.transform(Matrix([list(r) for r in np.linalg.inv(np.asarray(M_world))]))
    ob.data.update()
    return ob


def export(ob, rel, webp_quality=82):
    """Write the model alone as site/public/models/<rel>, then compress it with gltfpack (meshopt)."""
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = os.path.join(bpy.app.tempdir, os.path.basename(rel).replace('.glb', '.raw.glb'))
    # materials keep their plain names (paint, steel_bare) in the file, whichever copy this model holds
    for m in ob.data.materials:
        base = m.get('base', m.name)
        other = bpy.data.materials.get(base)
        if other is not None and other != m:
            other.name = base + '_spare'
        m.name = base
    select(ob)
    bpy.ops.export_scene.gltf(filepath=raw, export_format='GLB', use_selection=True, export_apply=True,
                              export_yup=True, export_texcoords=True, export_normals=True, export_tangents=False,
                              export_materials='EXPORT', export_image_format='WEBP', export_image_quality=webp_quality,
                              export_cameras=False, export_lights=False, export_extras=False, export_animations=False)
    pack = os.environ.get('GLTFPACK', 'gltfpack')
    subprocess.run([*pack.split(), '-i', raw, '-o', path, '-cc', '-km', '-kn', '-kv', '-noq'] if os.environ.get('WOBBL_NOQ')
                   else [*pack.split(), '-i', raw, '-o', path, '-cc', '-km', '-kn', '-kv'], check=True,
                   stdout=subprocess.DEVNULL)
    kb = os.path.getsize(path) / 1024
    print(f'[models] {rel} {kb:.0f} KB')
    return path


# ---------------------------------------------------------------- previews

def world_hdri(path, strength=1.0, background=(0.0055, 0.006, 0.0068), rot=0.0):
    """Light from a studio HDRI while the camera sees the site's own background color."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new('studio')
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, rot)
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(path, check_existing=True)
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    flat = nt.nodes.new('ShaderNodeBackground')
    flat.inputs['Color'].default_value = (*background, 1)
    lp = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
    nt.links.new(env.outputs['Color'], bg.inputs['Color'])
    # what the camera sees straight or through glass is the page's background; everything else is lit by the studio
    mx = nt.nodes.new('ShaderNodeMath')
    mx.operation = 'MAXIMUM'
    nt.links.new(lp.outputs['Is Camera Ray'], mx.inputs[0])
    nt.links.new(lp.outputs['Is Transmission Ray'], mx.inputs[1])
    nt.links.new(mx.outputs['Value'], mix.inputs['Fac'])
    nt.links.new(bg.outputs['Background'], mix.inputs[1])
    nt.links.new(flat.outputs['Background'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return w


def camera(eye, target, fov_v=30.0, aspect=4 / 3, up=(0, 0, 1), dof=None, clip=(0.001, 1000)):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new('cam')
    cd.sensor_fit = 'VERTICAL'
    cd.angle_y = math.radians(fov_v)
    cd.clip_start, cd.clip_end = clip
    cam = bpy.data.objects.new('cam', cd)
    sc.collection.objects.link(cam)
    z = (Vector(eye) - Vector(target)).normalized()
    x = Vector(up).cross(z).normalized()
    y = z.cross(x)
    cam.matrix_world = frame_matrix(eye, x, y)
    sc.camera = cam
    if dof:
        cd.dof.use_dof = True
        cd.dof.focus_distance = (Vector(eye) - Vector(target)).length
        cd.dof.aperture_fstop = dof
    return cam


def render(path, w=960, h=720, samples=96, film_transparent=False):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 8
    sc.cycles.transmission_bounces = 8
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = film_transparent
    # the tone map the site draws with, so a stress color keeps its hue
    try:
        sc.view_settings.view_transform = 'Khronos PBR Neutral'
    except TypeError:
        sc.view_settings.view_transform = 'AgX'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def write_manifest(example, entry):
    """Merge one example's entry into site/public/models/manifest.json (the format runtime reads), one model to a line."""
    path = os.path.join(OUT, 'manifest.json')
    m = {'version': 1, 'examples': {}}
    if os.path.exists(path):
        m = json.load(open(path))
    m['examples'][example] = entry
    os.makedirs(OUT, exist_ok=True)
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
