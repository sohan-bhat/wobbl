"""The printed lid's print bed in CAD (examples/printed-lid): a textured PEI spring steel sheet held by the magnetic
heatbed, its alignment pins in the sheet's notches, the heatbed's nine spacers on the Y carriage plate, and the thermistor and heater cable leaving at the back.

The run is drawn by the scene player, so the bed replaces the scene's print bed in that object's frame, built so the
sheet's top is the plane the lid prints on (z = 0 of the run).

    python site/cad/printed_lid.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from build123d import Axis, Circle, Pos, RectangleRounded, Solid, chamfer, extrude  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import cable  # noqa: E402
from mech import example as ex  # noqa: E402
from mech.util import plain  # noqa: E402

KEY = 'printed-lid'
CX, CY = 110.0, 30.0            # the lid's middle, which the bed is centred under
SW, SD = 254.0, 241.0           # the steel sheet
looks.add('heatbed_black', color=(0.025, 0.025, 0.028), metal=0.4, rough=0.45)
looks.add('cable_grey', color=(0.12, 0.12, 0.13), metal=0.0, rough=0.5)


def bed():
    out = []
    # the spring steel sheet: textured PEI on top, the tab at the front and two notches at the back for the pins
    f = Pos(CX, CY) * RectangleRounded(SW, SD, 4.0)
    f = f + Pos(CX, CY - SD / 2 - 6.0) * RectangleRounded(50.0, 16.0, 4.0)
    for s in (-1, 1):
        f = f - Pos(CX + s * 68.0, CY + SD / 2) * Circle(3.1)
    sheet = Pos(0, 0, -0.8) * extrude(f, 0.8)
    out.append((sheet, 'pei'))
    # the heatbed under it, black, on nine spacers
    hb = Pos(CX, CY, -0.8 - 3.0) * extrude(RectangleRounded(SW - 4.0, SD - 10.0, 6.0), 3.0)
    hb = chamfer(hb.edges().group_by(Axis.Z)[-1], 0.5)
    out.append((hb, 'heatbed_black'))
    for s in (-1, 1):
        out.append((Pos(CX + s * 68.0, CY + SD / 2, -0.8) * Solid.make_cylinder(2.8, 2.4), 'steel_bare'))
    zc = -0.8 - 3.0 - 22.0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            x, y = CX + i * 110.0, CY + j * 100.0
            # the spacer under each mounting point, down to the carriage
            sp = Pos(x, y, zc + 3.0) * (Solid.make_cylinder(3.5, 22.0 - 3.0) - Solid.make_cylinder(1.6, 22.0))
            out.append((sp, 'aluminum_machined'))
    # the Y carriage plate with its lightening holes
    car = Pos(CX, CY, zc) * extrude(RectangleRounded(236.0, 222.0, 10.0), 3.0)
    for i in (-1, 1):
        for j in (-1, 1):
            car = car - Pos(CX + i * 55.0, CY + j * 50.0, zc - 1) * extrude(RectangleRounded(70.0, 60.0, 12.0), 6.0)
    out.append((chamfer(car.edges().group_by(Axis.Z)[-1], 0.5), 'aluminum_machined'))
    # the cable bundle at the back, in its sleeve
    path = [(CX + 40.0, CY + SD / 2 - 12.0, -3.8), (CX + 40.0, CY + SD / 2 + 2.0, -7.0), (CX + 40.0, CY + SD / 2 + 14.0, -20.0),
            (CX + 30.0, CY + SD / 2 + 26.0, -45.0)]
    out.append((cable.tube(path, 3.2, seg=12), 'cable_grey'))
    return out


def main():
    sc = ex.scene(f'{KEY}/output')
    bi = sc.named('print bed')[0][0]
    files = {'print-bed': ([Node('print-bed', plain(ex.into(bed(), ex.pose_mm(sc, bi))), tol=0.1, ang=0.3)], {'object': 'print bed'})}
    ex.publish(KEY, files)


if __name__ == '__main__':
    main()
