"""Context models for the closed loudspeaker cabinet (examples/speaker-box): the 18 mm veneered cabinet round the
run's air cavity, with a 6.5 inch woofer and a soft dome tweeter on the front baffle, the terminal cup on the back
and rubber feet, drawn as a cutaway with the side and top walls off so the run's air cavity and its standing wave
slice stay in view with their colors.

    python site/cad/speaker_box.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import acoustic as ak  # noqa: E402
from build123d import Pos  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech.util import place  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'speaker-box', 'output', 'scene.wobbl')
WALL = 18.0   # examples/speaker-box/make_box.py: 240 x 200 x 360 outside, 18 mm board, a brace at z 210 with a 90 mm hole


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('cavity 1')[0])
    lo, hi = P.min(0) - WALL, P.max(0) + WALL
    cx = (lo[0] + hi[0]) / 2
    woofer_z, tweeter_z = lo[2] + 110.0, lo[2] + 290.0
    cab = ak.cabinet(*lo, *hi, WALL, cutouts=[(cx, woofer_z, 70.0), (cx, tweeter_z, 19.0)], brace=(210.0, 228.0, 45.0))
    export('speaker-box/cabinet.glb', [
        Node('cabinet', cab),
        Node('woofer', place(ak.woofer(), Pos(cx, lo[1], woofer_z))),
        Node('tweeter', place(ak.tweeter(), Pos(cx, lo[1], tweeter_z))),
    ], tol=0.08, ang=0.3)
    write_manifest('speaker-box', {'add': [{'file': 'speaker-box/cabinet.glb'}]})


if __name__ == '__main__':
    main()
