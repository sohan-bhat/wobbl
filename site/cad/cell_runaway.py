"""Context models for the 18650 in its oven (examples/cell-runaway): the cell's wrap lips, positive cap with its
vent holes and insulator ring and the negative can face, an alumina stand under it, and the type K thermocouple
taped to its side with its leads out of the oven. The cell keeps its temperature colors and glow.

    python site/cad/cell_runaway.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import battery as bt  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'cell-runaway', 'output', 'scene.wobbl')


def main():
    sc = Scene(RUN)
    i, o, M = sc.named('cell')[0]
    r, hh = o['size']
    z0 = M[2, 3] - hh
    export('cell-runaway/cell-ends.glb', [Node('cell_ends', bt.cell_ends(r, hh))], tol=0.02, ang=0.3)
    oven = sc.objects[sc.find('oven')[0]]
    P = sc.arr(oven['pos']).reshape(-1, 3)
    corner = (P[:, 0].max(), P[:, 1].min() * 0.3, P[:, 2].max() * 0.82)
    fix = bt.cell_stand(r, z0, z0) + bt.thermocouple(r, M[2, 3] + hh * 0.2, -60.0, corner)
    export('cell-runaway/fixture.glb', [Node('fixture', fix)], tol=0.03, ang=0.3)
    write_manifest('cell-runaway', {'add': [{'file': 'cell-runaway/cell-ends.glb', 'parent': 'cell'},
                                            {'file': 'cell-runaway/fixture.glb'}]})


if __name__ == '__main__':
    main()
