"""Context models for baking soda in vinegar (examples/baking-soda-vinegar): the run's beaker as a borosilicate
Griffin beaker with its rolled rim, pouring spout and printed graduations, figures and capacity, and a glass
stirring rod resting in it. The solution, the undissolved soda and the gathered CO2 keep their colors.

    python site/cad/baking_soda_vinegar.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import lab  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'baking-soda-vinegar', 'output', 'scene.wobbl')
EX = 'baking-soda-vinegar'


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('beaker')[0])
    r = np.hypot(P[:, 0], P[:, 1])
    r_out, h = float(r.max()), float(P[:, 2].max())
    wall = r_out - float(r[np.isclose(P[:, 2], h)].min())
    brim = math.pi * (r_out - wall) ** 2 * (h - wall) / 1000.0
    nominal = int(0.85 * brim // 100 * 100)
    parts = lab.beaker(r_out, h, wall, nominal, 50)
    parts += lab.stirring_rod((-r_out * 0.45, r_out * 0.25, wall + 3.5), (r_out * 1.05, -r_out * 0.35, h + 45))
    export(f'{EX}/beaker.glb', [Node('beaker', parts)], tol=0.05, ang=0.25)
    write_manifest(EX, {'replace': [{'object': 'beaker', 'file': f'{EX}/beaker.glb'}]})


if __name__ == '__main__':
    main()
