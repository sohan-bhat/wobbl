"""The desktop arm as a built machine: a printed base screwed to the desk with an MG996R inside turning the turret on
a slewing ring, the shoulder MG996R lying in the turret's cradle, aluminium upper arm plates on the shoulder horn
and an idler bearing, the elbow MG996R riding in the printed forearm, an MG90S wrist in the hand and a parallel
gripper whose fingers are driven through a pair of spur gears by a second MG90S.

Every joint is built on the axis the MuJoCo run solved: the scene's shafts give the joint positions and axes at
the frame the page opens on, and each part is moved into the frame of the body that carries it, so it follows the
body's pose through every value of the slider. A servo's case is tinted with the torque its body's field shows:
the base and shoulder servos hang under their motor objects, the elbow servo rides the forearm and the wrist servo
the hand, and each of those carries its own joint's torque.

    python site/cad/desktop_arm.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import (Axis, Box, Circle, Cylinder, Plane, Polyline, Pos, Rectangle, RectangleRounded, Rot,  # noqa: E402
                       SlotCenterToCenter, Solid, chamfer, extrude, fillet, make_face, Sketch, Location)

import looks  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech import bearings as br  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech import gears  # noqa: E402
from mech import servos as sv  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'desktop-arm'
UNIT = 0.001
looks.add('printed', color=(0.46, 0.46, 0.45), metal=0.0, rough=0.52)
looks.add('pad_rubber', color=(0.02, 0.02, 0.02), metal=0.0, rough=0.8)

Y = np.array([0.0, 1.0, 0.0])

# joint layout across the arm (world y, mm): horn faces toward -y, the side the page's camera looks from
SH_HORN = -25.8                                # shoulder and elbow MG996R horn faces
SH_TOP = SH_HORN + sv.horn_top('MG996R')       # their case tops
SH_BOT = SH_TOP + sv.SPECS['MG996R']['H']      # their case bottoms
PLATE = 3.0                                    # upper arm plates
WR_HORN = -16.5                                # wrist MG90S horn face
WR_TOP = WR_HORN + sv.horn_top('MG90S')
WR_BOT = WR_TOP + sv.SPECS['MG90S']['H']


def plate_v(outline_face, v0, v1):
    """A flat part drawn in a link's (u, w) plane, thick from v0 to v1 along the joint axis."""
    return Pos(0, v1, 0) * Rot(90, 0, 0) * extrude(outline_face, v1 - v0)


def soft(op, part, edges, r):
    """A fillet or chamfer that falls back to the plain part (and smaller radii) where the kernel cannot make it."""
    for k in (1.0, 0.6, 0.35):
        try:
            return op(edges, r * k)
        except Exception:
            continue
    return part


def face_edges(p, v):
    """The edges of a plate's face at v along y (its outline and its holes)."""
    return p.edges().filter_by(Axis.Y, reverse=True).filter_by_position(Axis.Y, v - 0.01, v + 0.01)


def link_frame(o, x):
    """A link's frame at o: u along x, v along the joint axis (world y), w square to both in the arm's plane."""
    x = np.asarray(x, dtype=np.float64)
    x = x / np.linalg.norm(x)
    z = np.cross(x, Y)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, Y, z, o
    return M


def servo_frame(o_axis, y_top, case_dir):
    """A servo placed with its output on the joint axis through o_axis, horn toward -y, its case top at y_top and the
    case running along case_dir."""
    o = np.array([o_axis[0], y_top, o_axis[2]])
    return frame(o, -np.asarray(case_dir, dtype=np.float64), -Y)


def screws_on_circle(M, r, n, a0, size, length, z, flip=False, mat='steel_black_oxide', head='cap'):
    """n screws on a bolt circle of radius r in a frame M (axis z), heads standing at height z."""
    out = []
    for i in range(n):
        a = math.radians(a0 + 360 * i / n)
        s = fs.socket_cap(size, length, threads=False) if head == 'cap' else fs.button_head(size, length)
        loc = Pos(r * math.cos(a), r * math.sin(a), z) * (Rot(180, 0, 0) if flip else Rot(0, 0, 0))
        out.append((loc * s, mat))
    return place(out, M)


# ---------------------------------------------------------------- the base and its servo

def base_parts():
    """The base: a round printed housing on a 140 mm aluminium foot plate screwed to the desk, vents round its front,
    a cable gland at the back and the slewing ring's outer race in a recess on its top deck."""
    out = []
    R, H = 54.0, 41.0
    foot = Pos(0, 0, 2.5) * Box(140, 140, 5)
    foot = fillet(foot.edges().filter_by(Axis.Z), 12)
    foot = chamfer(foot.edges().group_by(Axis.Z)[-1], 0.8)
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = foot - Pos(sx * 58, sy * 58, -1) * Solid.make_cylinder(2.6, 8)
    out.append((foot, 'aluminum_machined'))
    shell = Pos(0, 0, 5) * Solid.make_cylinder(R, H - 5)
    shell = fillet(shell.edges().group_by(Axis.Z)[-1], 5.0)
    shell = shell - Pos(0, 0, H - 1.5) * Solid.make_cylinder(27.5, 2)
    shell = shell - Pos(0, 0, 30) * Solid.make_cylinder(15.5, 12)
    # a skirt groove just above the foot and the vent slots round the front
    shell = shell - (Pos(0, 0, 8.0) * Solid.make_cylinder(R + 1, 0.8) - Pos(0, 0, 7) * Solid.make_cylinder(R - 0.6, 3))
    for k in range(9):
        a = math.radians(-90 + (k - 4) * 9)
        slot = Box(5, 2.6, 14)
        slot = fillet(slot.edges().filter_by(Axis.X), 1.2)
        shell = shell - Pos(R * math.cos(a), R * math.sin(a), 22) * Rot(0, 0, math.degrees(a)) * slot
    shell = shell - Pos(-R, 0, 15) * Rot(0, 90, 0) * Pos(0, 0, -4) * Solid.make_cylinder(4.2, 8)
    out.append((shell, 'printed'))
    gland = Pos(-R - 3.0, 0, 15) * Rot(0, 90, 0) * Pos(0, 0, -2.0) * fs.hex_prism(11, 4.0)
    out.append((gland, 'plastic_black'))
    out.append((Pos(-R - 5.0, 0, 15) * Rot(0, -90, 0) * Solid.make_cylinder(2.8, 10), 'rubber'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((Pos(sx * 58, sy * 58, 5.0) * fs.washer('M5'), 'steel_zinc'))
            out.append((Pos(sx * 58, sy * 58, 6.0) * fs.pan_phillips(4.5, 20, head_d=9.0, wood=True), 'steel_zinc'))
    # the screws that hold the housing to the foot, through the skirt
    for a in (45, 135, 225, 315):
        ar = math.radians(a)
        out.append((Pos(42 * math.cos(ar), 42 * math.sin(ar), H) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    outer, inner = br.slewing_ring(30, 52, 6, bolts=6)
    out += place(outer, Pos(0, 0, H - 1.5))
    for i in range(6):
        a = math.radians(30 + 60 * i)
        r = (52 / 2 + 20.5) / 2 + 0.2
        out.append((Pos(r * math.cos(a), r * math.sin(a), H - 1.5 + 6 * 0.92) * fs.socket_cap('M2', 8, threads=False, lite=True), 'steel_black_oxide'))
    return out, inner


def base_servo():
    """The base MG996R standing in the base, its horn under the turret's floor (the horn turns with the turret)."""
    zc = 37.6
    M = frame(np.array([0, 0, zc]), [1, 0, 0], [0, 0, 1])
    s = sv.servo('MG996R', horn='disc', cable=False, horn_top=47.0 - zc)
    case = place(s['case'] + s['output'], M)
    horn = place(s['horn'], M)
    return case, horn


# ---------------------------------------------------------------- the turret and the shoulder

def turret_parts(S, inner_race):
    """The printed turret: the round floor on the slewing ring with a paddle under the shoulder servo, the posts the
    servo's tabs screw to and the wall carrying the shoulder's idler pivot."""
    out = []
    z0, z1 = 47.0, 50.0
    floor = Pos(0, 0, z0) * Solid.make_cylinder(23.5, z1 - z0)
    paddle = Pos(-12, 9.6, (z0 + z1) / 2) * Box(54, 30.4, z1 - z0)
    paddle = fillet(paddle.edges().filter_by(Axis.Z), 6)
    body = floor + paddle
    # the front and back posts behind the servo's tabs, and the idler wall
    tz = SH_TOP + 10.0                      # the face the tabs bear on (their underside faces +y)
    for x0, x1 in ((-38.6, -31.0), (10.8, 18.4)):
        post = Pos((x0 + x1) / 2, (tz + SH_BOT + 3) / 2, (z1 + 71) / 2) * Box(x1 - x0, SH_BOT + 3 - tz, 71 - z1)
        body = body + post
    wall = Pos(-10.1, SH_BOT + 0.3 + 1.25, (z1 + 66) / 2) * Box(57, 2.5, 66 - z1)
    boss = Pos(S[0], SH_BOT + 0.3, S[2]) * Rot(-90, 0, 0) * Solid.make_cylinder(8.5, 2.5)
    body = body + wall + boss
    body = body + Pos(-12, SH_BOT + 0.3 + 1.25, (z1 + 60) / 2) * Box(30, 2.5, 10)
    body = body - Pos(S[0], SH_BOT, S[2]) * Rot(-90, 0, 0) * Solid.make_cylinder(1.25, 8)
    out.append((body, 'printed'))
    # the inner race turns with the turret
    out += place(inner_race, Pos(0, 0, 41))
    return out


def shoulder_servo(S):
    """The shoulder MG996R lying in the turret, case toward the back; tab screws into the posts."""
    M = servo_frame(S, SH_TOP, [-1, 0, 0])
    s = sv.servo('MG996R', horn='disc')
    case = place(s['case'] + s['output'], M)
    horn = place(s['horn'], M)
    case += tab_screws(M, 'MG996R')
    return case, horn


def tab_screws(M, model):
    """Screws through a servo's tab holes into whatever carries it, heads on the tab's top face."""
    tzl, tzh, xs, ys = sv.tab_plane(model)
    size = 'M3' if model == 'MG996R' else 'M2'
    out = []
    for x in xs:
        for y in ys:
            out.append((Pos(x, y, tzh) * fs.socket_cap(size, 10, threads=False), 'steel_black_oxide'))
            out.append((Pos(x, y, tzh) * fs.washer(size), 'steel_zinc'))
    out = [(Pos(0, 0, fs.WASHER['M3' if model == 'MG996R' else 'M2'][2]) * s if m == 'steel_black_oxide' else s, m) for s, m in out]
    return place(out, M)


def idler(p, y_face, size='M3', bearing='F623'):
    """An idler pivot at the joint point p: a flanged bearing pressed into a plate from its outer face at y_face,
    the flange outside, and a shoulder screw with its washer through it into the boss behind the plate."""
    d, D, B = br.SIZES[br.FLANGE[bearing][0]]
    ft = br.FLANGE[bearing][2]
    out = place(br.flanged(bearing), frame(np.array([p[0], y_face - (B - ft), p[2]]), [1, 0, 0], [0, 1, 0]))
    w = fs.WASHER[size][2]
    out += place([(fs.washer(size), 'steel_zinc')], frame(np.array([p[0], y_face + ft, p[2]]), [1, 0, 0], [0, 1, 0]))
    out += place([(fs.socket_cap(size, 12, threads=False), 'steel_black_oxide')],
                 frame(np.array([p[0], y_face + ft + w, p[2]]), [1, 0, 0], [0, 1, 0]))
    return out


# ---------------------------------------------------------------- the upper arm

def upper_arm_parts(S, E):
    """Two 3 mm aluminium plates from the shoulder to the elbow with lightening slots, tied by hex standoffs: the -y
    plate bolts to both servo horns, the +y plate carries the idler bearings."""
    L = float(np.linalg.norm(E - S))
    M = link_frame(S, E - S)
    out = []
    prof = Pos(L / 2, 0) * SlotCenterToCenter(L, 26)
    holes_common = [Pos(L * 0.39, 0) * SlotCenterToCenter(16, 9), Pos(L * 0.61, 0) * SlotCenterToCenter(16, 9)]
    for side in (-1, 1):
        f = prof
        for h in holes_common:
            f = f - h
        for u in (0, L):
            if side < 0:
                f = f - Pos(u, 0) * Circle(3.3)
                for k in range(4):
                    a = math.radians(45 + 90 * k)
                    f = f - Pos(u + 7 * math.cos(a), 7 * math.sin(a)) * Circle(1.6)
            else:
                f = f - Pos(u, 0) * Circle(5.0)
        for u in (L * 0.2, L * 0.8):
            f = f - Pos(u, 0) * Circle(1.6)
        v0, v1 = (SH_HORN - PLATE, SH_HORN) if side < 0 else (SH_BOT + 3.0, SH_BOT + 3.0 + PLATE)
        p = plate_v(f, v0, v1)
        p = soft(chamfer, p, face_edges(p, v0) + face_edges(p, v1), 0.35)
        out.append((p, 'aluminum_machined'))
    # hex standoffs between the plates and their button heads outside
    span = (SH_BOT + 3.0) - SH_HORN
    for u in (L * 0.2, L * 0.8):
        st = Pos(u, SH_HORN, 0) * Rot(-90, 0, 0) * fs.hex_prism(5.5, span, chamfer_both=True)
        st = st - Pos(u, SH_HORN - 1, 0) * Rot(-90, 0, 0) * Solid.make_cylinder(1.25, span + 2)
        out.append((st, 'aluminum_machined'))
        out.append((Pos(u, SH_HORN - PLATE, 0) * Rot(90, 0, 0) * fs.button_head('M3', 6, lite=True), 'steel_black_oxide'))
        out.append((Pos(u, SH_BOT + 3.0 + PLATE, 0) * Rot(-90, 0, 0) * fs.button_head('M3', 6, lite=True), 'steel_black_oxide'))
    # the horn screws on the -y plate, at both ends
    for u in (0, L):
        for k in range(4):
            a = math.radians(45 + 90 * k)
            out.append((Pos(u + 7 * math.cos(a), SH_HORN - PLATE, 7 * math.sin(a)) * Rot(90, 0, 0) * fs.socket_cap('M3', 6, threads=False, lite=True), 'steel_black_oxide'))
    out = place(out, M)
    # idler bearings in the +y plate at both joints
    for P in (S, E):
        out += idler(P, SH_BOT + 3.0 + PLATE)
    return out


# ---------------------------------------------------------------- the forearm and the elbow

def forearm_parts(E, W):
    """The printed forearm: a cradle round the elbow servo running into a tapered, filleted channel beam with
    lightening windows, which ends in the wrist cheeks."""
    L = float(np.linalg.norm(W - E))
    M = link_frame(E, W - E)
    sp = sv.SPECS['MG996R']
    out = []
    tz = SH_TOP + 10.0
    c0, c1 = -sp['off'], sp['L'] - sp['off']          # the case runs from -10 to +30.3 along u
    # the cradle: a tub under the servo's lower half, open toward -y where the tabs sit on its rim
    tub = Pos((c0 + c1) / 2, (tz + SH_BOT + 2.5) / 2, 0) * Box(c1 - c0 + 17, SH_BOT + 2.5 - tz, 25.2)
    tub = fillet(tub.edges().filter_by(Axis.Y), 3.0)
    tub = fillet(tub.edges().group_by(Axis.Y)[-1], 2.0)
    pocket = Pos((c0 + c1) / 2, (tz - 1 + SH_BOT + 0.3) / 2, 0) * Box(c1 - c0 + 1.2, SH_BOT + 0.3 - tz + 1, 20.6)
    tub = tub - pocket
    # the tab pads are solid: the pocket is narrowed where the tabs bear on the rim
    # the beam: a tapered channel with filleted edges, walls at the wrist cheeks' spacing along v
    wv0, wv1 = WR_HORN - 3.5, WR_BOT + 2.5
    u0, u1 = c1 + 4.0, L
    prof = make_face(Polyline((u0, -12.6), (u1 - 10, -9.5), (u1 - 10, 9.5), (u0, 12.6), close=True))
    prof = prof + Pos(u1, 0) * Circle(9.5) + Pos(u1 - 5, 0) * Rectangle(10, 19)
    beam = plate_v(prof, wv0, wv1)
    beam = fillet(beam.edges().filter_by(Axis.Y, reverse=True).filter_by_position(Axis.Y, wv0 - 0.01, wv0 + 0.01).filter_by_position(Axis.X, u0 + 1, u1 - 9), 2.0)
    beam = fillet(beam.edges().filter_by(Axis.Y, reverse=True).filter_by_position(Axis.Y, wv1 - 0.01, wv1 + 0.01).filter_by_position(Axis.X, u0 + 1, u1 - 9), 2.0)
    # hollow it: a channel between the walls, open where the hand swings between the cheeks
    inner = make_face(Polyline((u0 + 6, -10.0), (u1 - 12, -7.2), (u1 - 12, 7.2), (u0 + 6, 10.0), close=True))
    beam = beam - plate_v(inner, WR_HORN + 0.01, WR_BOT + 2.5 - 3.9)
    beam = beam - Pos(u1 - 2, (WR_HORN + WR_BOT - 1.4) / 2, 0) * Box(24, WR_BOT - 1.4 - WR_HORN, 30)
    # lightening windows through both walls
    for k in range(3):
        uk = u0 + 14 + k * 22
        hk = 7.2 - k * 0.8
        win = Pos(uk, (wv0 + wv1) / 2, 0) * Box(15, wv1 - wv0 + 2, hk * 2)
        beam = beam - fillet(win.edges().filter_by(Axis.Y), 2.5)
    # the wrist holes: horn screws in the -y cheek, the idler bearing bore in the +y cheek
    beam = beam - Pos(u1, wv0 - 1, 0) * Rot(-90, 0, 0) * Solid.make_cylinder(2.6, 5)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        beam = beam - Pos(u1 + 5 * math.cos(a), wv0 - 1, 5 * math.sin(a)) * Rot(-90, 0, 0) * Solid.make_cylinder(1.1, 5)
    beam = beam - Pos(u1, wv1 + 1, 0) * Rot(90, 0, 0) * Solid.make_cylinder(5.0, 5)
    tie = Pos(c1 + 5.5, (wv0 + SH_BOT + 2.5) / 2, 0) * Box(5, SH_BOT + 2.5 - wv0, 25.2)
    body = tub + beam + tie
    body = body - Pos(0, SH_BOT - 2, 0) * Rot(-90, 0, 0) * Solid.make_cylinder(1.25, 8)
    out.append((body, 'printed'))
    out = place(out, M)
    out += idler(W, wv1)
    hm = link_frame(W, W - E)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        out += place([(Pos(5 * math.cos(a), wv0, 5 * math.sin(a)) * Rot(90, 0, 0) * fs.socket_cap('M2', 5, threads=False, lite=True), 'steel_black_oxide')], hm)
    return out


def elbow_servo(E, W):
    """The elbow MG996R in the forearm's cradle, case along the forearm."""
    M = servo_frame(E, SH_TOP, (W - E))
    s = sv.servo('MG996R', horn='disc')
    case = place(s['case'] + s['output'], M) + tab_screws(M, 'MG996R')
    horn = place(s['horn'], M)
    return case, horn


# ---------------------------------------------------------------- the hand, the wrist and the gripper

def hand_parts(W, G):
    """The printed hand: a cradle round the wrist MG90S with its idler wall, and the bar the gripper hangs from."""
    sp = sv.SPECS['MG90S']
    M = link_frame(W, G - W)
    tz = WR_TOP + 6.8
    c0, c1 = -sp['off'], sp['L'] - sp['off']
    body = Pos((c0 - 5.6 + c0 - 0.4) / 2, (tz + WR_BOT + 2.0) / 2, 0) * Box(5.2, WR_BOT + 2.0 - tz, 13)
    body = body + Pos((c1 + 0.4 + c1 + 5.6) / 2, (tz + WR_BOT + 2.0) / 2, 0) * Box(5.2, WR_BOT + 2.0 - tz, 13)
    body = body + Pos((c0 - 5.6 + c1 + 5.6) / 2, WR_BOT + 0.2 + 0.9, 0) * Box(c1 - c0 + 11.2, 1.8, 13)
    # the gripper's top bar, from the cradle's far post down to the gripper plates
    bar = Pos(c1 + 3.0 + 3.0, (-13.0 + 9.0) / 2, 0) * Box(6, 22.0, 34)
    bar = fillet(bar.edges().filter_by(Axis.Y), 2.0)
    body = body + bar
    body = body - Pos(0, WR_BOT - 2, 0) * Rot(-90, 0, 0) * Solid.make_cylinder(1.25, 8)
    out = [(body, 'printed')]
    return place(out, M)


def wrist_servo(W, G):
    M = servo_frame(W, WR_TOP, (G - W))
    s = sv.servo('MG90S', horn='disc')
    case = place(s['case'] + s['output'], M) + tab_screws(M, 'MG90S')
    horn = place(s['horn'], M)
    return case, horn


def gripper_parts(W, G):
    """A parallel gripper with its gear train on the front: a black anodised main plate, two meshing 16 tooth spur
    gears, each driving a finger through a parallelogram of links so the rubber pads stay square to each other, and
    an MG90S behind the plate turning the left gear."""
    M = link_frame(W, G - W)
    out = []
    u_top = 22.0
    ug = 32.0                          # gear axes
    mod, z = 1.0, 16
    cd = mod * z
    pv0, pv1 = 1.0, 4.0                # main plate
    gv0, gv1 = -3.0, 0.6               # gears
    dv0, dv1 = -5.2, -3.4              # driver links
    iv0, iv1 = -8.2, -6.4              # idler links, in front on standoffs
    fv0, fv1 = -10.8, -8.6             # fingers
    ip = 9.0                           # idler pivots sit this far below the gears
    f = make_face(Polyline((u_top, -20), (u_top, 20), (ug + 6, 23), (ug + ip + 6, 20), (ug + ip + 6, -20), (ug + 6, -23), close=True))
    for s in (-1, 1):
        f = f - Pos(ug, s * cd / 2) * Circle(1.6)
        f = f - Pos(ug + ip, s * 15.5) * Circle(1.6)
        f = f - Pos(u_top + 3.5, s * 15) * Circle(1.6)
    f = f - Pos(ug, -cd / 2) * Circle(3.2)
    for s in (-1, 1):
        f = f - Pos(ug + ip * 0.5, s * 2.5) * Circle(2.2)
    p = plate_v(f, pv0, pv1)
    p = soft(chamfer, p, face_edges(p, pv0), 0.35)
    out.append((p, 'aluminum_anodized'))
    for s in (-1, 1):
        out.append((Pos(u_top + 3.5, pv0, s * 15) * Rot(90, 0, 0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    # the gears, the right one turned half a tooth so they mesh, on shoulder screws
    for s in (-1, 1):
        g = gears.spur(mod, z, gv1 - gv0, bore=3.0, holes=4, hole_r=1.3)
        g = Rot(0, 0, 0 if s < 0 else 180.0 / z) * g
        out.append((Pos(ug, gv1, s * cd / 2) * Rot(90, 0, 0) * g, 'aluminum_machined'))
    # links and fingers
    th = math.radians(14)
    Ld = 21.0
    for s in (-1, 1):
        a = np.array([ug, s * cd / 2])
        b = np.array([ug + ip, s * 15.5])
        d = np.array([Ld * math.cos(th), s * Ld * math.sin(th)])
        da, db = a + d, b + d
        for (p0, p1), (v0, v1) in (((a, da), (dv0, dv1)), ((b, db), (iv0, iv1))):
            ln = float(np.linalg.norm(p1 - p0))
            ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            lf = Pos(*p0) * Rot(0, 0, ang) * Pos(ln / 2, 0) * SlotCenterToCenter(ln, 6.0)
            lf = lf - Pos(*p0) * Circle(1.6) - Pos(*p1) * Circle(1.6)
            out.append((plate_v(lf, v0, v1), 'aluminum_machined'))
        # the idler's spacer from the plate and the screws at every pivot
        sp_ = Pos(b[0], iv1, b[1]) * Rot(-90, 0, 0) * Solid.make_cylinder(2.2, pv0 - iv1)
        out.append((sp_, 'aluminum_machined'))
        out.append((Pos(a[0], dv0, a[1]) * Rot(90, 0, 0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
        out.append((Pos(b[0], iv0, b[1]) * Rot(90, 0, 0) * fs.button_head('M3', 12, lite=True), 'steel_black_oxide'))
        # the finger: a plate on both link ends running down to the pad
        tip_u = 67.0
        pad_w = s * 5.5
        ff = make_face(Polyline((da[0] - 4, da[1] + s * 3.5), (db[0] + 3, db[1] + s * 4.0), (tip_u - 4, pad_w + s * 6.5),
                                (tip_u, pad_w + s * 5.5), (tip_u, pad_w), (da[0] + 3, pad_w), (da[0] - 4, da[1] - s * 3.5), close=True))
        ff = ff - Pos(*da) * Circle(1.6) - Pos(*db) * Circle(1.6)
        fp = plate_v(ff, fv0, fv1)
        out.append((fp, 'aluminum_anodized'))
        # the finger's web back to the link plane, and its pad
        web = Pos((da[0] + 4 + tip_u) / 2, (fv0 + dv1) / 2, pad_w + s * 2.6) * Box(tip_u - da[0] - 5, dv1 - fv0, 3.4)
        out.append((web, 'aluminum_anodized'))
        pad = Pos((da[0] + 5 + tip_u) / 2, (fv0 + dv1) / 2, pad_w + s * 0.2) * Box(tip_u - da[0] - 7, dv1 - fv0 - 0.6, 1.6)
        out.append((pad, 'pad_rubber'))
        for p_ in (da, db):
            out.append((Pos(p_[0], fv0, p_[1]) * Rot(90, 0, 0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    out = place(out, M)
    # the gripper servo behind the plate, its spline through the plate into the left gear
    top = pv1 + 0.4
    Mg = frame((M @ np.array([ug, top, -cd / 2, 1]))[:3], -M[:3, 0], -Y)
    s = sv.servo('MG90S', case_material='plastic_black', horn=None, cable=True)
    gsv = place(s['case'] + s['output'], Mg)
    cpl = Pos(ug, gv1, -cd / 2) * Rot(-90, 0, 0) * Solid.make_cylinder(2.8, top - 5.8 - gv1 + 1.0)
    gsv += place([(cpl, 'aluminum_machined')], M)
    return out + gsv


# ---------------------------------------------------------------- build

HARDWARE = {'steel_black_oxide', 'steel_zinc', 'steel_bare', 'steel_chrome', 'brass', 'wire_brown', 'wire_red',
            'wire_orange', 'rubber', 'pad_rubber'}


def nodes_for(slug, parts):
    """A model's node tree: its body parts finely meshed, the hardware (screws, bearings, wires) coarser, since a
    screw head is a few pixels across at the page's zoom."""
    body = plain([(s, m) for s, m in parts if m not in HARDWARE])
    hw = plain([(s, m) for s, m in parts if m in HARDWARE])
    kids = [Node(f'{slug}-body', body, tol=0.05, ang=0.3)]
    if hw:
        kids.append(Node(f'{slug}-hardware', hw, tol=0.08, ang=0.55))
    return [Node(slug, children=kids)]


def build():
    sc = ex.scene(f'{KEY}/output')
    idx = ex.index(sc)
    home = sc.home()
    S = ex.at(sc, 'shoulder shaft', home)
    E = ex.at(sc, 'elbow shaft', home)
    W = ex.at(sc, 'wrist shaft', home)
    G = ex.at(sc, 'gripper', home)
    for P in (S, E, W, G):
        P[1] = 0.0

    def M(name):
        return ex.pose_mm(sc, idx[name], home)

    base, inner = base_parts()
    bcase, bhorn = base_servo()
    turret = turret_parts(S, inner) + bhorn
    scase, shorn = shoulder_servo(S)
    upper = upper_arm_parts(S, E) + shorn
    ecase, ehorn = elbow_servo(E, W)
    upper += ehorn
    fore = forearm_parts(E, W) + ecase
    wcase, whorn = wrist_servo(W, G)
    fore += whorn
    hand = hand_parts(W, G) + wcase
    grip = gripper_parts(W, G)

    models = [
        ('base', 'base', base),
        ('base motor', 'base-servo', bcase),
        ('turret', 'turret', turret),
        ('shoulder motor', 'shoulder-servo', scase),
        ('upper arm', 'upper-arm', upper),
        ('forearm', 'forearm', fore),
        ('hand', 'hand', hand),
        ('gripper', 'gripper', grip),
    ]
    entries = []
    sizes = {}
    for obj, slug, parts in models:
        local = ex.into(parts, M(obj))
        rel = f'{KEY}/{slug}.glb'
        sizes[rel] = export(rel, nodes_for(slug, local), unit=UNIT)
        e = {'object': obj, 'file': rel}
        if any(m == 'servo_case' for _, m in local):
            e['tint'] = ['servo_case']
        entries.append(e)
    entries.append({'object': 'desk', 'file': f'{KEY}/desk.glb'})
    return entries, sizes


def main():
    entries, sizes = build()
    write_manifest(KEY, {'replace': entries,
                         'hide': ['elbow servo', 'wrist servo', 'elbow motor', 'wrist motor', 'base shaft', 'shoulder shaft',
                                  'elbow shaft', 'wrist shaft']})
    print(f'[cad] {KEY} {sum(sizes.values()) / 1024:.0f} KB in {len(sizes)} files')


if __name__ == '__main__':
    main()
