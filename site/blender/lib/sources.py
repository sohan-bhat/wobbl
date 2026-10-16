"""CC0 textures and the studio HDRI the models and stills use, fetched once into site/blender/.cache and checked
against the checksums recorded here and in site/blender/SOURCES.md.
"""
import hashlib
import os
import urllib.request
import zipfile

import bpy
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')
TEX = os.path.join(CACHE, 'tex')

# ambientCG material id: sha256 of its 1K-JPG zip
AMBIENTCG = {
    'Concrete034': '5839d284d94ffb8d2a56df742ec522b13dd311c52dbd42b8fd33f0409ceedb81',
    'Wood049': '74ec51bcf5ddfc2e9af6205dc5cbcc90ad51c22117e013ff8b1c7258f8ecbc69',
    'Rock030': '2fd0bc2d214e2487c19062ed0798a85dd135b41cfc12bbaed42d5acae238f989',
}
# Poly Haven HDRI: (url, md5 Poly Haven publishes for the file)
HDRI = {
    'studio_small_09': ('https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_09_2k.hdr', 'b056fea247bc84b81d2e0987d44c87d8'),
}
UA = {'User-Agent': 'wobbl-site-build/1.0 (+https://github.com/sohan-bhat/wobbl)'}


def _get(url, path, digest, algo):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(path + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(path + '.part', path)
    h = hashlib.new(algo, open(path, 'rb').read()).hexdigest()
    if h != digest:
        os.remove(path)
        raise RuntimeError(f'{os.path.basename(path)} does not match its recorded {algo}')
    return path


def hdri(name='studio_small_09'):
    url, md5 = HDRI[name]
    return _get(url, os.path.join(CACHE, 'hdri', os.path.basename(url)), md5, 'md5')


def _load(path, colour):
    im = bpy.data.images.load(path)
    im.colorspace_settings.name = 'sRGB' if colour else 'Non-Color'
    return im


def _write(name, im, size, pixels=None):
    """Save an image scaled to size by size as a PNG in the texture cache (square, so the exporter can pack the
    roughness map with a baked occlusion map of the same size)."""
    im.scale(size, size)
    if pixels is not None:
        im.pixels[:] = pixels(np.array(im.pixels[:], dtype=np.float32).reshape(-1, 4)).reshape(-1)
    im.filepath_raw = os.path.join(TEX, name)
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def ambientcg(mid, size=512, metal=0.0, rough_scale=1.0, tone=1.0):
    """Color, OpenGL normal and a metallic roughness map (roughness in green, metal in blue) for an ambientCG
    material, at size pixels wide. Returns their file names in the texture cache."""
    names = (f'{mid}_c{tone:.2f}.png', f'{mid}_n.png', f'{mid}_r{metal:.0f}.png')
    if all(os.path.exists(os.path.join(TEX, n)) for n in names):
        return names
    zp = _get(f'https://ambientcg.com/get?file={mid}_1K-JPG.zip', os.path.join(CACHE, 'acg', f'{mid}.zip'), AMBIENTCG[mid], 'sha256')
    out = os.path.join(CACHE, 'acg', mid)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith(('_Color.jpg', '_NormalGL.jpg', '_Roughness.jpg')) and '/' not in n:
                z.extract(n, out)
    os.makedirs(TEX, exist_ok=True)
    pick = {k: os.path.join(out, f'{mid}_1K-JPG_{k}.jpg') for k in ('Color', 'NormalGL', 'Roughness')}
    def dim(p):
        p[:, :3] *= tone
        return p
    _write(names[0], _load(pick['Color'], True), size, dim)
    _write(names[1], _load(pick['NormalGL'], False), size)

    def mr(p):
        r = np.clip(p[:, 0] * rough_scale, 0, 1)
        p[:, 0], p[:, 1], p[:, 2] = 1.0, r, metal
        return p
    _write(names[2], _load(pick['Roughness'], False), size, mr)
    return names
