"""Context models for the propane flame (examples/propane-flame): a Bunsen burner on the run's own barrel, with its
cast iron base, felt pad, needle valve knob, brass barrel with air holes and the knurled air collar over them, the
brass hose barb of the gas inlet and the rubber gas hose off it. The flame keeps its colors and glow.

    python site/cad/propane_flame.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import lab  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'propane-flame', 'output', 'scene.wobbl')
EX = 'propane-flame'


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('burner')[0])
    r = np.hypot(P[:, 0], P[:, 1])
    top = float(P[:, 2].max())
    barrel_r = float(r[np.isclose(P[:, 2], top)].max())
    base_r = float(r.max())
    i, o, M = sc.named('air inlet')[0]
    rb, hl = o['size']
    z_in = M[2, 3]
    export(f'{EX}/burner.glb', [Node('burner', lab.bunsen(barrel_r, base_r, 6.0, top, (z_in + 6, z_in + 16)))], tol=0.02, ang=0.3)
    export(f'{EX}/air-inlet.glb', [Node('barb', lab.hose_barb(rb, hl))], tol=0.01, ang=0.3)
    end = M[:3, 3] - M[:3, 2] * hl
    hose = lab.gas_hose([end + np.array([1.5, 0, 0]), end + np.array([-10, 0, -3]), end + np.array([-26, -6, -z_in + 6]),
                         end + np.array([-60, -24, -z_in + 4.2]), end + np.array([-130, -40, -z_in + 4.2])])
    export(f'{EX}/gas-hose.glb', [Node('gas_hose', hose)], tol=0.03, ang=0.3)
    write_manifest(EX, {'replace': [{'object': 'burner', 'file': f'{EX}/burner.glb'},
                                    {'object': 'air inlet', 'file': f'{EX}/air-inlet.glb'}],
                        'add': [{'file': f'{EX}/gas-hose.glb'}]})


if __name__ == '__main__':
    main()
