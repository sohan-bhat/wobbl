"""Context models for the falling rock (examples/falling-rock): the ground at the foot of the cliff as packed soil,
a talus apron of angular boulders and scree along the cliff's foot, and pebbles round the landing spot, kept clear
of the rock's own path. The cliff itself is drawn by the scene, since its height follows the input as the page
plays, and a model only follows a pose.

    python site/cad/falling_rock.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import lab  # noqa: E402
import looks  # noqa: E402
from build123d import Cylinder, Pos  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'falling-rock', 'output', 'scene.wobbl')
EX = 'falling-rock'
MM = 1000.0
looks.add('stone', color=(0.2, 0.185, 0.16), metal=0.0, rough=0.86, normal='cast_n', nstr=1.0, rvar=(0.74, 0.92), tiles=1.5)
looks.add('soil', color=(0.12, 0.1, 0.075), metal=0.0, rough=0.92, normal='fine_n', nstr=1.5, rvar=(0.82, 0.95), tiles=2.0)


def main():
    sc = Scene(RUN)
    g = next(o for o in sc.objects if o['kind'] == 'ground')
    cx, cy = g['center']
    R = g['size'] / 2
    P, _ = sc.mesh(sc.find('cliff')[0])
    face_x = float(P[:, 0].max())
    y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
    rng = np.random.default_rng(7)
    rocks, pebbles = [], []
    # the talus apron: big blocks against the cliff's foot, thinning out away from it, none in the rock's path
    for k in range(70):
        y = rng.uniform(y0 + 2, y1 - 2)
        x = face_x + abs(rng.normal(0, 7)) + 0.6
        if abs(y) < 6 and x < 14:
            continue
        size = rng.uniform(0.8, 3.6) * np.exp(-(x - face_x) / 25)
        rocks.append(lab.boulder(size * MM, 100 + k, (x * MM, y * MM, 0)))
    for k in range(110):
        y = rng.uniform(y0, y1)
        x = face_x + abs(rng.normal(0, 12)) + 0.3
        if abs(y) < 3 and x < 6:
            continue
        rocks.append(lab.boulder(rng.uniform(0.15, 0.6) * MM, 400 + k, (x * MM, y * MM, 0)))
    # pebbles round the landing spot, half a metre clear of it
    for k in range(40):
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.55, 3.0)
        x, y = d * np.cos(a), d * np.sin(a)
        if x < face_x + 0.05:
            continue
        pebbles.append(lab.boulder(rng.uniform(0.04, 0.16) * MM, 900 + k, (x * MM, y * MM, 0)))
    ground = Pos(cx * MM, cy * MM, -0.25 * MM) * Cylinder(R * MM, 0.5 * MM)
    # each node is quantized over its own extent, so the pebbles keep their shape next to a 1.3 km ground
    export(f'{EX}/ground.glb', [Node('ground', [(ground, 'soil')]), Node('talus', [(r, 'stone') for r in rocks]),
                                Node('pebbles', [(r, 'stone') for r in pebbles])],
           unit=1 / MM, tol=50.0, ang=0.2, pack=('-vp', '14', '-vt', '12'))
    write_manifest(EX, {'replace': [{'object': 'rock', 'file': f'{EX}/rock.glb'}],
                        'add': [{'file': f'{EX}/ground.glb'}], 'hide': ['ground']})


if __name__ == '__main__':
    main()
