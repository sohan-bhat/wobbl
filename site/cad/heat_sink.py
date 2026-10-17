"""Context model for the natural convection heat sink (examples/heat-sink): the chip under its base, a flip chip
BGA with a nickel plated lid against the aluminium, land side capacitors on the substrate and the ball grid below.
The heat sink itself is the simulated part and keeps its temperature colors.

    python site/cad/heat_sink.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import thermal as th  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'heat-sink', 'output', 'scene.wobbl')


def main():
    sc = Scene(RUN)
    i, o, M = sc.named('heat source')[0]
    hx, hy, hz = o['size']
    # the run draws this part in its own millimetre frame, which is the scene's, so the chip is placed there
    export('heat-sink/heat-source.glb', [Node('heat_source', th.fcbga(hx, hy, hz), loc=M)], tol=0.01, ang=0.3)
    write_manifest('heat-sink', {'add': [{'file': 'heat-sink/heat-source.glb', 'covers': ['heat source']}]})


if __name__ == '__main__':
    main()
