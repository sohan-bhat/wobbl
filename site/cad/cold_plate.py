"""Context models for the liquid cold plate (examples/cold-plate): G1/4 barbed fittings in the two ports with O
rings, rubber hoses on the barbs held by worm drive clamps, and the 62 mm half bridge IGBT module bolted under the
plate whose 150 W it carries away. The plate keeps its temperature colors and the water its flow lines.

    python site/cad/cold_plate.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import thermal as th  # noqa: E402
from build123d import Pos  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech.util import place  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'cold-plate', 'output', 'scene.wobbl')
LENGTH, WIDTH, THICK, BORE, PITCH = 120.0, 70.0, 16.0, 8.0, 40.0     # examples/cold-plate/plate.py


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('part')[0])
    lo, hi = P.min(0), P.max(0)
    z = (lo[2] + hi[2]) / 2
    ys = [(WIDTH - PITCH) / 2 + lo[1], (WIDTH + PITCH) / 2 + lo[1]]
    nodes = []
    hose_r = 5.2 + 1.6
    for k, y in enumerate(ys):
        parts = place(th.barb_fitting(BORE / 2), Pos(lo[0], y, z))
        parts += place(th.hose_clamp(hose_r), Pos(lo[0] - 17.5, y, z))
        s = -1 if k == 0 else 1
        path = [(lo[0] - 10, y, z), (lo[0] - 40, y, z), (lo[0] - 62, y + s * 6, z - 6), (lo[0] - 74, y + s * 16, z - 24)]
        parts += th.hose(path, hose_r)
        nodes.append(Node(f'port_{k + 1}', parts))
    nodes.append(Node('igbt_module', th.igbt_module(106.0, 61.4, (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])))
    export('cold-plate/plumbing.glb', nodes, tol=0.03, ang=0.35)
    write_manifest('cold-plate', {'add': [{'file': 'cold-plate/plumbing.glb'}]})


if __name__ == '__main__':
    main()
