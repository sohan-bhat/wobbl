"""A pull solenoid's parts as a half section, from open frame solenoid drawings: the zinc plated C frame with its pole piece, tapped
mounting holes and broken edges, the coil on a glass filled nylon bobbin with its winding turns, polyester tape and
leads, and the plated plunger with its clevis slot, cross pin hole and E clip. Builders follow the run's own STEP
dimensions (examples/solenoid-actuator/make_actuator.py) exactly, Z up in millimetres.
"""
import math

import numpy as np
from build123d import Align, Axis, Box, Compound, Cylinder, Pos, Rot, chamfer, fillet

import electronics  # noqa: F401  (registers the shared metals)
import battery  # noqa: F401  (silicone lead colors)
import looks

# the parts are drawn as a half section, cut on the plane the field slice lies in (y = 0) and kept behind it, so
# the slice, the flux lines and the force on the plunger stay in view the way a CAD section view shows them
CUT = 0.06


def section(shape):
    return shape & Pos(0, 500 + CUT, 0) * Box(1000, 1000, 1000)

looks.add('bobbin', color=(0.03, 0.03, 0.032), metal=0.0, rough=0.45)
looks.add('poly_tape', color=(0.75, 0.55, 0.08), metal=0.0, rough=0.3)
looks.add('enamel_copper', color=(0.75, 0.32, 0.12), metal=1.0, rough=0.25)
looks.add('steel_yellow_zinc', color=(0.78, 0.68, 0.38), metal=1.0, rough=0.3)


def frame(half=20.0, height=44.0, plate=4.0, bore=6.3, pole_r=6.0, pole_top=16.0):
    """The C frame: bottom and top plates and two side plates, open front and back, the plunger bore in the top
    plate with a chamfer, the pole piece standing on the bottom plate, and M3 tapped holes for mounting."""
    outer = Pos(0, 0, height / 2) * Box(2 * half, 2 * half, height)
    inner = Pos(0, 0, height / 2) * Box(2 * (half - plate), 2 * half + 2, height - 2 * plate)
    f = outer - inner
    f = chamfer(f.edges().filter_by(Axis.Y), 0.5)
    f = f - Pos(0, 0, height - plate / 2) * Cylinder(bore, plate + 2)
    holes = [Pos(sx * 11, sy * 11, 0) * Cylinder(1.25, 8.0) for sx in (-1, 1) for sy in (-1, 1)]
    holes += [Pos(sx * half, 0, h) * Cylinder(1.25, 8.0, rotation=(0, 90, 0)) for sx in (-1, 1) for h in (12, 32)]
    f = f - Compound(holes)
    pole = Pos(0, 0, plate) * Cylinder(pole_r, pole_top - plate, align=(Align.CENTER, Align.CENTER, Align.MIN))
    pole = chamfer(pole.edges().group_by(Axis.Z)[-1], 0.6)
    return [(section(f), 'steel_yellow_zinc'), (section(pole), 'steel_yellow_zinc')]


def coil(r_in=7.0, r_out=15.0, z0=5.0, z1=39.0, leads=None):
    """The coil: a bobbin with end flanges, the copper winding showing its turns at both ends, a band of polyester
    tape round the middle, and the two leads leaving toward leads [(x, y, z), (x, y, z)]."""
    fl = 1.0
    tube = Pos(0, 0, z0) * (Cylinder(r_in + 0.6, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN)) -
                            Cylinder(r_in, 3 * (z1 - z0)))
    flanges = [Pos(0, 0, z) * (Cylinder(r_out, fl, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(r_in, 4))
               for z in (z0, z1 - fl)]
    out = [(section(tube), 'bobbin')] + [(section(f), 'bobbin') for f in flanges]
    # the winding, its outer layer of turns as rings, under tape in the middle
    rw = r_out - 0.5
    wire = 0.3
    a, b = z0 + fl, z1 - fl
    body = Pos(0, 0, a) * (Cylinder(rw - wire, b - a, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(r_in + 0.6, 4 * (b - a)))
    out.append((section(body), 'enamel_copper'))
    from battery import torus_mesh
    tape0, tape1 = a + (b - a) * 0.2, b - (b - a) * 0.2
    z = a + wire
    while z < b - wire / 2:
        if not (tape0 - 0.1 < z < tape1 + 0.1):
            out.append((torus_mesh(rw - wire, wire, z, seg=26, ring=5, arc=(0.004, math.pi - 0.004)), 'enamel_copper'))
        z += 2 * wire
    tape = Pos(0, 0, tape0) * (Cylinder(rw + 0.08, tape1 - tape0, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(rw - 1, 99))
    out.append((section(tape), 'poly_tape'))
    if leads:
        from mech import cable
        for k, (end, mat) in enumerate(zip(leads, ('silicone_red', 'silicone_black'))):
            ang = math.radians(80 + 20 * k)
            p0 = (math.cos(ang) * (r_out - 2), math.sin(ang) * (r_out - 2), z1 + 0.2)
            p1 = (math.cos(ang) * (r_out + 1), math.sin(ang) * (r_out + 3), z1 - 1.5)
            out.append((cable.tube([p0, (p0[0], p0[1] + 1, p0[2] + 1), p1, end], 0.55, seg=8, n=48), mat))
    return out


def plunger(r=6.0, z0=19.0, z1=53.0, clip_z=47.0):
    """The plunger: plated steel with a chamfered nose, a groove and E clip above the frame, and a clevis slot with
    its cross pin hole at the working end."""
    p = Pos(0, 0, z0) * Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p = chamfer(p.edges().group_by(Axis.Z)[0], 0.8)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 0.5)
    groove = Pos(0, 0, clip_z) * (Cylinder(r + 1, 0.8) - Cylinder(r - 0.5, 2))
    slot = Pos(0, 0, z1) * Box(r * 0.55, 2 * r + 2, 8.0)
    pin = Pos(0, 0, z1 - 2.6) * Cylinder(1.3, 2 * r + 2, rotation=(0, 90, 0))
    p = p - groove - slot - pin
    out = [(section(p), 'steel_chrome')]
    clip = Pos(0, 0, clip_z) * (Cylinder(r + 2.2, 0.6) - Cylinder(r - 0.5, 2))
    clip = clip - Pos(0, -(r + 2.0), clip_z) * Box(r * 1.3, r * 1.8, 2)
    clip = clip - Compound([Rot(0, 0, a) * Pos(0, r + 1.5, clip_z) * Box(1.6, 1.4, 2) for a in (0, 120, 240)])
    out.append((section(clip), 'steel_black_oxide'))
    return out
