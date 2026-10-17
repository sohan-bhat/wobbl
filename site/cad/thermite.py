"""Context models for the thermite crucible (examples/thermite): the run's crucible as a clay graphite crucible with
a glazed rim and pouring lip, set in a mound of dry sand on a steel tray, crucible tongs on the ground beside it and
the magnesium ribbon fuse standing out of the charge. The burning charge keeps its colors and glow, seen through
the crucible's wall.

    python site/cad/thermite.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import lab  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'thermite', 'output', 'reaction', 'scene.wobbl')
EX = 'thermite'


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('crucible')[0])
    r, z = np.hypot(P[:, 0], P[:, 1]), np.round(P[:, 2], 2)
    outline = [(float(zz), float(r[z == zz].min()), float(r[z == zz].max())) for zz in sorted(set(z))]
    ground = next(o for o in sc.objects if o['kind'] == 'ground')['z']
    C, _ = sc.mesh(sc.find('burned charge')[0])
    top = float(C[:, 2].max())
    (zb, _, ro_b), _, (zt, _, ro_t) = outline
    r_at = ro_b + (ro_t - ro_b) * (18 - zb) / (zt - zb)
    export(f'{EX}/crucible.glb', [Node('crucible', lab.crucible(outline))], tol=0.05, ang=0.3)
    props = lab.sand_bed(r_at - 1.0, ro_b + 26, ground + 2.0, 18.0) + lab.tongs(260, (110, -70, ground), 28) + \
        lab.mg_ribbon(top)
    export(f'{EX}/setup.glb', [Node('setup', props)], tol=0.05, ang=0.3)
    write_manifest(EX, {'replace': [{'object': 'crucible', 'file': f'{EX}/crucible.glb'}],
                        'add': [{'file': f'{EX}/setup.glb'}]})


if __name__ == '__main__':
    main()
