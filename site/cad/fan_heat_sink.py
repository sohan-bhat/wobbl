"""Context models for the ducted fan heat sink (examples/fan-heat-sink): a 40 mm axial fan with twisted blades,
struts, screws and leads, on a moulded duct that funnels it onto the fin channels, the flip chip BGA under the base
and a patch of the board it sits on, with the fan's header. The heat sink keeps its temperature colors.

    python site/cad/fan_heat_sink.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import thermal as th  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech.util import frame  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'fan-heat-sink', 'output', 'scene.wobbl')
SIZE, BASE, HEIGHT = 60.0, 6.0, 36.0          # examples/fan-heat-sink/heatsink.py
FAN, DEPTH, DUCT = 40.0, 20.0, 12.0


def main():
    sc = Scene(RUN)
    P, _ = sc.mesh(sc.find('part')[0])
    lo, hi = P.min(0), P.max(0)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    # the duct from the heat sink's intake face (y = 0, air along +y) out to the fan
    duct_at = frame((cx, lo[1], cz), (1, 0, 0), (0, -1, 0))
    fan_at = frame((cx, lo[1] - DUCT - DEPTH / 2, cz), (1, 0, 0), (0, 1, 0))
    header = (cx + 24.0, lo[1] + 6.0, lo[2] - 2.4 + 2.6)
    hl = np.linalg.inv(fan_at) @ np.array([*header, 1.0])
    fan = th.fan(FAN, DEPTH, 7, 32.0, screws=True, leads=tuple(hl[:3]))
    chip = th.fcbga(12.5, 12.5, 1.2)
    board = th.board_patch(lo[0] - 12, lo[1] + 0.5, hi[0] + 12, hi[1] + 6, lo[2] - 2.4, parts=[
        (cx + 24.0, lo[1] + 6.0, 3.0, 1.3, 1.3, 'plastic_white'),
        (cx - 20, lo[1] + 8, 3.0, 3.0, 1.6, 'epoxy_black'), (cx - 20, lo[1] + 16, 3.0, 3.0, 1.6, 'epoxy_black'),
        (cx - 20, lo[1] + 24, 3.0, 3.0, 1.6, 'epoxy_black'), (cx + 22, hi[1] - 10, 2.6, 2.6, 2.2, 'ferrite'),
        (cx - 22, hi[1] - 10, 2.6, 2.6, 2.2, 'ferrite'), (cx + 22, hi[1] - 18, 2.6, 2.6, 2.2, 'ferrite')])
    chip_at = frame((cx, (lo[1] + hi[1]) / 2, lo[2] - 1.2), (1, 0, 0), (0, 0, 1))
    export('fan-heat-sink/fan.glb', [Node('fan', fan, loc=fan_at)], tol=0.04, ang=0.45)
    export('fan-heat-sink/duct.glb', [Node('duct', th.duct(SIZE, HEIGHT, FAN, FAN, DUCT), loc=duct_at)], tol=0.02, ang=0.3)
    export('fan-heat-sink/board.glb', [Node('chip', chip, loc=chip_at), Node('board', board)], tol=0.02, ang=0.3)
    write_manifest('fan-heat-sink', {'add': [
        {'file': 'fan-heat-sink/fan.glb'}, {'file': 'fan-heat-sink/duct.glb'}, {'file': 'fan-heat-sink/board.glb'}]})


if __name__ == '__main__':
    main()
