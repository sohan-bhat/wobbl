"""The steel hook's test rig in CAD (examples/steel-hook): the hook hangs from an M8 x 35 bolt through its eye and a
10 mm steel bracket welded under a box section crossmember, and the 3 kN pull comes in through a galvanised bow
shackle seated in the bottom of the bend, a rod end eye bolt and an S beam load cell with its cable.

The run is drawn by the scene player: the bolt model replaces the scene's bolt in its frame (its head with it, so
the scene's head is hidden), the shackle and load cell replace the scene's weight, and the bracket is added.

    python site/cad/steel_hook.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Pos, Rot, Solid, fillet  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import cable  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'steel-hook'
EYE = np.array([0.0, 0.0, 57.25])     # the bolt's axis through the eye, along y
HOOK_Y = 5.0                          # the hook is 10 mm thick, y -5..5
BEND = np.array([22.0, 0.0, -16.0])   # the inside of the bend at the bottom, where the sling bears
looks.add('galvanised', color=(0.6, 0.62, 0.63), metal=1.0, rough=0.48)
looks.add('paint_blue', color=(0.04, 0.12, 0.3), metal=0.0, rough=0.4)
looks.add('cable_grey', color=(0.12, 0.12, 0.13), metal=0.0, rough=0.5)


def bolt_stack():
    """M8 x 35 hex bolt from the front, washers both sides, through the hook's eye and the bracket, nut behind."""
    w = fs.WASHER['M8'][2]
    M = frame(np.array([EYE[0], -HOOK_Y, EYE[2]]), [1, 0, 0], [0, -1, 0])        # z out of the front face
    parts = [(Pos(0, 0, w) * fs.hex_bolt('M8', 35), 'steel_zinc'), (fs.washer('M8'), 'steel_zinc')]
    back = 15.0
    Mb = frame(np.array([EYE[0], back, EYE[2]]), [1, 0, 0], [0, 1, 0])
    nut = [(fs.washer('M8'), 'steel_zinc'), (Pos(0, 0, w) * fs.hex_nut('M8'), 'steel_zinc')]
    return place(parts, M) + place(nut, Mb)


def bracket():
    """A 10 mm flat bar hanger, its top welded under an 80 x 40 box section crossmember painted blue."""
    plate = Pos(0, 10.0, 85.0) * Box(64, 10, 70)
    plate = fillet(plate.edges().filter_by(Axis.Y).group_by(Axis.Z)[0], 12.0)
    plate = plate - Pos(EYE[0], 0, EYE[2]) * Rot(90, 0, 0) * Pos(0, 0, -20) * Solid.make_cylinder(4.5, 40)
    beam = Pos(0, 10.0, 140.0) * Box(320, 40, 40)
    beam = beam - Pos(0, 10.0, 140.0) * Box(322, 32, 32)
    welds = []
    for s in (-1, 1):
        bead = Pos(0, 10.0 + s * 5.0, 120.0) * Rot(0, 90, 0) * Solid.make_cylinder(2.2, 64, Plane.XY.offset(-32))
        welds.append((bead, 'weld_bead'))
    caps = [(Pos(s * 160.4, 10.0, 140.0) * Box(0.8, 40, 40), 'paint_blue') for s in (-1, 1)]
    return [(plate, 'steel_mill'), (beam, 'paint_blue')] + welds + caps


def shackle_and_cell():
    """A bow shackle round the bottom of the bend (its bow square to the hook), its pin, a rod end eye bolt, the
    S beam load cell and its cable, and the pull rod below."""
    out = []
    d = 8.0                                  # the shackle's bar
    rc = 11.0                                # the bow's centre line radius, clear of the 10 mm hook section
    zc = BEND[2] + d / 2 - rc                # the bow's centre, its crown bearing on the bend
    # bow: a half torus in the yz plane over the top, and straight legs down to the eyes
    bow = Solid.make_torus(rc, d / 2, Plane(origin=(BEND[0], 0, zc), x_dir=(0, 1, 0), z_dir=(1, 0, 0)), major_angle=180)
    legs_bottom = zc - 18.0
    for s in (-1, 1):
        leg = Pos(BEND[0], s * rc, legs_bottom) * Solid.make_cylinder(d / 2, zc - legs_bottom)
        eye = Pos(BEND[0], s * rc, legs_bottom) * Rot(90, 0, 0) * (Solid.make_cylinder(d * 0.95, d, Plane.XY.offset(-d / 2))
                                                                     - Solid.make_cylinder(5.2, 3 * d, Plane.XY.offset(-1.5 * d)))
        bow = bow + leg + eye
    out.append((bow, 'galvanised'))
    pin_y0, pin_y1 = -rc - d / 2 - 1.0, rc + d / 2 + 6.0
    pin = Pos(BEND[0], pin_y0, legs_bottom) * Rot(-90, 0, 0) * Solid.make_cylinder(5.0, pin_y1 - pin_y0)
    head = Pos(BEND[0], pin_y0, legs_bottom) * Rot(90, 0, 0) * Solid.make_cylinder(8.5, 4.0)
    out += [(pin, 'galvanised'), (head, 'galvanised')]
    nut = Pos(BEND[0], rc + d / 2, legs_bottom) * Rot(-90, 0, 0) * fs.hex_nut('M10')
    out.append((nut, 'galvanised'))
    # the rod end eye bolt on the pin, between the legs
    ze = legs_bottom
    eye = Pos(BEND[0], 0, ze) * Rot(90, 0, 0) * (Solid.make_cylinder(9.0, 9.0, Plane.XY.offset(-4.5)) - Solid.make_cylinder(5.2, 20, Plane.XY.offset(-10)))
    shank = Pos(BEND[0], 0, ze - 8.0 - 18.0) * Solid.make_cylinder(5.0, 18.0)
    out += [(eye, 'steel_zinc'), (shank, 'steel_zinc')]
    out.append((Pos(BEND[0], 0, ze - 8.0 - 6.0 - 8.0) * fs.hex_nut('M10'), 'steel_zinc'))
    # the S beam load cell: 51 x 76 x 20 mm, the gauge pocket potted, M10 holes top and bottom
    top = ze - 8.0 - 12.0
    cz = top - 38.0
    s = Pos(BEND[0], 0, cz) * Box(51, 20, 76)
    s = fillet(s.edges().filter_by(Axis.Y), 3.0)
    for zz, side in ((cz + 13.0, 1), (cz - 13.0, -1)):
        slot = Pos(BEND[0] + side * 6.0, 0, zz) * Box(39, 22, 6.0)
        s = s - fillet(slot.edges().filter_by(Axis.Y), 2.9)
    s = s - Pos(BEND[0], 0, cz) * Rot(90, 0, 0) * Pos(0, 0, -11) * Solid.make_cylinder(9.0, 22)
    out.append((s, 'steel_stainless'))
    pot = Pos(BEND[0], 0, cz) * Rot(90, 0, 0) * Pos(0, 0, -6) * Solid.make_cylinder(9.0, 12)
    out.append((pot, 'rubber'))
    lab = Pos(BEND[0] + 4, -10.05, cz + 25) * Box(28, 0.2, 12)
    out.append((lab, 'label'))
    gland = Pos(BEND[0] + 25.5, 0, cz) * Rot(0, 90, 0) * fs.hex_prism(10, 6)
    out.append((gland, 'plastic_black'))
    out.append((cable.tube([(BEND[0] + 31, 0, cz), (BEND[0] + 46, 0, cz - 4), (BEND[0] + 56, 0, cz - 26), (BEND[0] + 58, 0, cz - 60)], 2.4, seg=10), 'cable_grey'))
    # the pull rod below, into the actuator's clevis out of the picture
    bot = cz - 38.0
    rod = Pos(BEND[0], 0, bot - 70.0) * Solid.make_cylinder(5.0, 70.0)
    out.append((rod, 'steel_zinc'))
    out.append((Pos(BEND[0], 0, bot - 8.0) * fs.hex_nut('M10'), 'steel_zinc'))
    return out


def main():
    sc = ex.scene(f'{KEY}/output')
    Mb = ex.pose_mm(sc, sc.named('bolt')[0][0])
    Mw = ex.pose_mm(sc, sc.named('weight')[0][0])
    looks.add('weld_bead', color=(0.18, 0.17, 0.16), metal=1.0, rough=0.55)
    files = {
        'bolt': ([Node('m8-bolt', plain(ex.into(bolt_stack(), Mb)), tol=0.05, ang=0.35)], {'object': 'bolt'}),
        'shackle-and-load-cell': ([Node('shackle-and-load-cell', plain(ex.into(shackle_and_cell(), Mw), names=('steel_zinc', 'galvanised', 'steel_stainless', 'rubber')), tol=0.06, ang=0.3)],
                                  {'object': 'weight'}),
        'bracket': ([Node('hanger-bracket', bracket(), tol=0.1, ang=0.3)], {}),
    }
    ex.publish(KEY, files, extra={'hide': ['bolt head']})


if __name__ == '__main__':
    main()
