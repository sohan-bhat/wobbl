"""How the CAD models look: the same material set the Blender models use (site/blender/lib/looks.py), as glTF
metallic roughness materials, with the same seeded tiling detail maps and the same CC0 scanned materials.

A material is named here and nowhere else, so a model's file carries the plain names the runtime reads ("paint"
is the surface it may tint with a part's live color). Textures are made once into site/cad/.cache and written into
each file as WebP.
"""
import hashlib
import io
import math
import os
import ssl
import urllib.request
import zipfile

import numpy as np
from PIL import Image

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')
TEX = os.path.join(CACHE, 'tex')

# ---------------------------------------------------------------- textures
# Band limited noise filtered in the frequency domain, so it wraps, seeded so a rebuild makes the same maps.


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


def _normal_from(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    nrm = np.stack([-dx, dy, np.ones_like(h)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    return nrm * 0.5 + 0.5


NORMALS = {
    # name: (size, band low, band high, x stretch, y stretch, strength, seed)
    'cast_n': (256, 14, 70, 1, 1, 0.9, 11),
    'peel_n': (256, 6, 22, 1, 1, 0.35, 12),
    'brushed_n': (256, 20, 110, 0.04, 1, 0.5, 13),
    'rubber_n': (256, 40, 120, 1, 1, 0.6, 14),
    'concrete_n': (512, 8, 200, 1, 1, 0.7, 15),
    'fine_n': (256, 50, 128, 1, 1, 0.25, 16),
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


def _png(name, rgb):
    os.makedirs(TEX, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(TEX, name + '.png'))


def tex_normal(name):
    if not os.path.exists(os.path.join(TEX, name + '.png')):
        if name == 'tread_n':
            _png(name, _normal_from(_tread(), 6.0))
        else:
            n, lo, hi, ax, ay, st, seed = NORMALS[name]
            h = _noise(n, lo, hi, seed, ax, ay)
            if name == 'cast_n':
                h = np.sign(h) * np.abs(h) ** 0.7 + 0.35 * _noise(n, 4, 12, seed + 100)
            _png(name, _normal_from(h, st))
    return name


def tex_rough(lo, hi, metal, seed=21):
    """A metallic roughness map: roughness from lo to hi in large soft smudges and fine grain, metal in blue."""
    name = f'r_{lo:.2f}_{hi:.2f}_m{metal:.0f}'
    if not os.path.exists(os.path.join(TEX, name + '.png')):
        n = 256
        x = np.clip(0.5 + 0.12 * _noise(n, 1.5, 5, seed) + 0.16 * _noise(n, 30, 110, seed + 1), 0, 1)
        r = lo + (hi - lo) * x
        _png(name, np.stack([np.ones_like(r), r, np.full_like(r, metal)], -1))
    return name


# ambientCG publishes these under CC0 1.0 (site/blender/SOURCES.md): material id -> sha256 of its 1K-JPG zip
AMBIENTCG = {
    'Concrete034': '5839d284d94ffb8d2a56df742ec522b13dd311c52dbd42b8fd33f0409ceedb81',
    'Wood049': '74ec51bcf5ddfc2e9af6205dc5cbcc90ad51c22117e013ff8b1c7258f8ecbc69',
    'Rock030': '2fd0bc2d214e2487c19062ed0798a85dd135b41cfc12bbaed42d5acae238f989',
}
UA = {'User-Agent': 'wobbl-site-build/1.0 (+https://github.com/sohan-bhat/wobbl)'}


def ambientcg(mid, size=512, metal=0.0, tone=1.0):
    """Color, OpenGL normal and metallic roughness map names for a scanned ambientCG material."""
    names = (f'{mid}_c{tone:.2f}', f'{mid}_n', f'{mid}_r{metal:.0f}')
    if all(os.path.exists(os.path.join(TEX, n + '.png')) for n in names):
        return names
    zp = os.path.join(CACHE, 'acg', f'{mid}.zip')
    if not os.path.exists(zp):
        os.makedirs(os.path.dirname(zp), exist_ok=True)
        url = f'https://ambientcg.com/get?file={mid}_1K-JPG.zip'
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = None
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120, context=ctx) as r:
            data = r.read()
        open(zp + '.part', 'wb').write(data)
        os.replace(zp + '.part', zp)
    if hashlib.sha256(open(zp, 'rb').read()).hexdigest() != AMBIENTCG[mid]:
        os.remove(zp)
        raise RuntimeError(f'{mid}.zip does not match its recorded sha256')
    with zipfile.ZipFile(zp) as z:
        def load(kind):
            return Image.open(io.BytesIO(z.read(f'{mid}_1K-JPG_{kind}.jpg'))).convert('RGB').resize((size, size), Image.LANCZOS)
        c = np.asarray(load('Color'), dtype=np.float64) / 255 * tone
        _png(names[0], c)
        _png(names[1], np.asarray(load('NormalGL'), dtype=np.float64) / 255)
        r = np.asarray(load('Roughness'), dtype=np.float64)[..., 0] / 255
        _png(names[2], np.stack([np.ones_like(r), r, np.full_like(r, metal)], -1))
    return names


def webp(name, quality=82):
    """A cached texture as WebP bytes, the way the Blender exporter wrote them."""
    buf = io.BytesIO()
    Image.open(os.path.join(TEX, name + '.png')).save(buf, 'WEBP', quality=quality, method=6)
    return buf.getvalue()


# ---------------------------------------------------------------- the material set
# color (linear), metal, rough, normal map, normal strength, roughness range, detail tiles per metre and extras,
# exactly as site/blender/lib/looks.py defines them

def M(color, metal=0.0, rough=0.5, normal=None, nstr=1.0, rvar=None, tiles=60.0, **kw):
    return dict(color=color, metal=metal, rough=rough, normal=normal, nstr=nstr, rvar=rvar, tiles=tiles, **kw)


MATERIALS = {
    'paint': M((0.03, 0.03, 0.032), 0.0, 0.42, 'peel_n', 0.25, (0.32, 0.5), 40),
    'paint_black': M((0.022, 0.022, 0.024), 0.0, 0.5, 'peel_n', 0.2, (0.42, 0.6), 40),
    'steel_bare': M((0.62, 0.62, 0.62), 1.0, 0.3, None, 0.4, (0.22, 0.38), 30),
    'steel_zinc': M((0.74, 0.75, 0.77), 1.0, 0.34, None, 0.5, (0.26, 0.44), 60),
    'steel_dark': M((0.16, 0.16, 0.17), 1.0, 0.4, None, 0.5, (0.32, 0.5), 60),
    'aluminum_cast': M((0.40, 0.41, 0.42), 1.0, 0.6, 'cast_n', 0.5, (0.54, 0.66), 25),
    'aluminum_machined': M((0.8, 0.81, 0.82), 1.0, 0.27, None, 0.12, (0.25, 0.3), 90),
    'plastic_black': M((0.03, 0.03, 0.032), 0.0, 0.45, None, 0.6, (0.38, 0.55), 60),
    'rubber': M((0.018, 0.018, 0.018), 0.0, 0.72, None, 0.7, (0.62, 0.85), 80),
    'paint_white': M((0.82, 0.82, 0.8), 0.0, 0.22, 'peel_n', 0.08, (0.16, 0.3), 30, coat=1.0, coat_rough=0.04),
    'paint_red': M((0.62, 0.07, 0.04), 0.0, 0.22, 'peel_n', 0.08, (0.16, 0.3), 30, coat=1.0, coat_rough=0.04),
    'paint_gloss_black': M((0.018, 0.018, 0.02), 0.0, 0.25, 'peel_n', 0.1, (0.18, 0.32), 30, coat=1.0, coat_rough=0.05),
    'decal': M((0.015, 0.015, 0.016), 0.0, 0.32),
    'graphite': M((0.05, 0.05, 0.055), 0.0, 0.62, 'cast_n', 0.4, (0.55, 0.72), 60),
    'aluminum_anodized': M((0.05, 0.05, 0.055), 1.0, 0.35, None, 0.4, (0.3, 0.42), 30),
    'nylon': M((0.72, 0.71, 0.66), 0.0, 0.45, None, 0.4, (0.4, 0.55), 60),
    'tread_steel': M((0.5, 0.5, 0.5), 1.0, 0.45, 'tread_n', 1.0, (0.38, 0.6), 6),
    'steel_mill': M((0.2, 0.205, 0.215), 1.0, 0.55, 'fine_n', 0.15, (0.48, 0.62), 40),
    'steel_stainless': M((0.66, 0.66, 0.67), 1.0, 0.26, 'brushed_n', 0.3, (0.2, 0.32), 30),
    'steel_black_oxide': M((0.07, 0.07, 0.075), 1.0, 0.4, None, 0.4, (0.32, 0.48), 60),
    'steel_chrome': M((0.85, 0.85, 0.86), 1.0, 0.08),
    'concrete': M((1, 1, 1), 0.0, 0.85, None, 0.8, None, 1.6, pbr='Concrete034', tone=0.55),
    'wood': M((1, 1, 1), 0.0, 0.6, None, 0.6, None, 5.0, pbr='Wood049', tone=0.8),
    'rock': M((1, 1, 1), 0.0, 0.8, None, 1.0, None, 1.0, pbr='Rock030', tone=1.5),
    'label': M((0.78, 0.78, 0.76), 0.0, 0.42),
    'plastic_white': M((0.8, 0.8, 0.78), 0.0, 0.38, None, 0.5, (0.32, 0.46), 60),
    'aluminum_anodized_gold': M((0.75, 0.6, 0.32), 1.0, 0.32, None, 0.4, (0.26, 0.4), 30),
    'paint_wall': M((0.36, 0.36, 0.35), 0.0, 0.82, 'fine_n', 0.3, (0.75, 0.9), 20),
    'acrylic': M((0.92, 0.95, 0.96), 0.0, 0.04, transmission=1.0, ior=1.49),
    'cardboard': M((0.45, 0.32, 0.19), 0.0, 0.85, 'fine_n', 0.4, (0.78, 0.92), 30),
    'tape': M((0.55, 0.42, 0.25), 0.0, 0.3),
    'cast_iron': M((0.035, 0.035, 0.037), 0.0, 0.6, 'cast_n', 0.8, (0.5, 0.7), 40),
    'pei': M((0.42, 0.27, 0.09), 0.25, 0.55, 'cast_n', 0.6, (0.45, 0.65), 50),
    'brass': M((0.86, 0.66, 0.36), 1.0, 0.28, None, 0.4, (0.22, 0.38), 40),
    'enamel_white': M((0.85, 0.85, 0.83), 0.0, 0.4),
    'concrete_polished': M((1, 1, 1), 0.0, 0.35, None, 0.5, None, 1.6, pbr='Concrete034', tone=0.7),
    'glass': M((0.86, 0.93, 0.9), 0.0, 0.02, transmission=1.0, ior=1.52),
    'frit': M((0.012, 0.012, 0.012), 0.0, 0.32),
}


def add(name, **kw):
    """Add a material to the set (a model's own colors, such as a resistor's bands), returning its name."""
    if name not in MATERIALS:
        MATERIALS[name] = M(**kw)
    return name


def textures(name):
    """The texture names a material uses: (base color, normal, metallic roughness), any of them None."""
    d = MATERIALS[name]
    if d.get('pbr'):
        c, n, r = ambientcg(d['pbr'], metal=d['metal'], tone=d.get('tone', 1.0))
        return c, n, r
    n = tex_normal(d['normal']) if d['normal'] else None
    r = tex_rough(d['rvar'][0], d['rvar'][1], d['metal']) if d['rvar'] else None
    return None, n, r


def gltf_material(name, tex_index):
    """The glTF material for a name. tex_index(texture name) gives the file's texture index for a map."""
    d = MATERIALS[name]
    c, n, r = textures(name)
    pbr = {'baseColorFactor': [*d['color'], 1.0]}
    if c:
        pbr['baseColorTexture'] = {'index': tex_index(c)}
    if r:
        pbr['metallicRoughnessTexture'] = {'index': tex_index(r)}
    else:
        pbr['metallicFactor'] = d['metal']
        pbr['roughnessFactor'] = d['rough']
    m = {'name': name, 'pbrMetallicRoughness': pbr, 'doubleSided': True}
    if n:
        m['normalTexture'] = {'index': tex_index(n), 'scale': d['nstr']}
    ext = {}
    if d.get('coat'):
        ext['KHR_materials_clearcoat'] = {'clearcoatFactor': d['coat'], 'clearcoatRoughnessFactor': d.get('coat_rough', 0.05)}
    if d.get('transmission'):
        ext['KHR_materials_transmission'] = {'transmissionFactor': d['transmission']}
        ext['KHR_materials_ior'] = {'ior': d.get('ior', 1.5)}
    if d.get('emissive'):
        m['emissiveFactor'] = list(d['emissive'])
    if ext:
        m['extensions'] = ext
    return m
