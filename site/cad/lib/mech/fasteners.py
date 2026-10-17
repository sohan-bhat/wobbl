"""ISO metric fasteners at their standard dimensions: socket head cap screws (ISO 4762), hex bolts (ISO 4017), button
heads (ISO 7380), countersunk screws (ISO 10642), set screws (ISO 4026), hex nuts (ISO 4032), washers (ISO 7089),
plus wood screws and nylon wall plugs for the props that hang on walls.

Every screw stands on +z with the face under its head at z = 0: the head rises above, the shank runs down to -length.
A nut or a washer sits on z = 0 and rises to its height. Threads are suggested by rolled rings at the real pitch,
which read as thread at any zoom the pages allow and cost a fraction of a true helix.

    from mech import fasteners as fs
    s = fs.socket_cap('M3', 10)                    # a build123d Part
    parts = fs.bolt_stack('M8', 40, grip=24)       # [(shape, material), ...] bolt, washers and nut
"""
import math

from build123d import (Axis, Box, Cylinder, Locations, Plane, Polyline, Pos, RegularPolygon, Rot, Solid, Sketch, Cone,
                       chamfer, extrude, fillet, make_face, revolve, Part)

# coarse pitch
PITCH = {'M2': 0.4, 'M2.5': 0.45, 'M3': 0.5, 'M4': 0.7, 'M5': 0.8, 'M6': 1.0, 'M8': 1.25, 'M10': 1.5, 'M12': 1.75,
         'M16': 2.0, 'M20': 2.5}
# ISO 4762: head diameter, head height, socket across flats, socket depth
SHCS = {'M2': (3.8, 2.0, 1.5, 1.0), 'M2.5': (4.5, 2.5, 2.0, 1.1), 'M3': (5.5, 3.0, 2.5, 1.3), 'M4': (7.0, 4.0, 3.0, 2.0),
        'M5': (8.5, 5.0, 4.0, 2.5), 'M6': (10.0, 6.0, 5.0, 3.0), 'M8': (13.0, 8.0, 6.0, 4.0), 'M10': (16.0, 10.0, 8.0, 5.0),
        'M12': (18.0, 12.0, 10.0, 6.0), 'M16': (24.0, 16.0, 14.0, 8.0)}
# ISO 4017 / 4032: across flats, bolt head height, nut height
HEX = {'M2': (4.0, 1.4, 1.6), 'M2.5': (5.0, 1.7, 2.0), 'M3': (5.5, 2.0, 2.4), 'M4': (7.0, 2.8, 3.2), 'M5': (8.0, 3.5, 4.7),
       'M6': (10.0, 4.0, 5.2), 'M8': (13.0, 5.3, 6.8), 'M10': (16.0, 6.4, 8.4), 'M12': (18.0, 7.5, 10.8),
       'M16': (24.0, 10.0, 14.8), 'M20': (30.0, 12.5, 18.0)}
# ISO 7089: inner diameter, outer diameter, thickness
WASHER = {'M2': (2.2, 5.0, 0.3), 'M2.5': (2.7, 6.0, 0.5), 'M3': (3.2, 7.0, 0.5), 'M4': (4.3, 9.0, 0.8), 'M5': (5.3, 10.0, 1.0),
          'M6': (6.4, 12.0, 1.6), 'M8': (8.4, 16.0, 1.6), 'M10': (10.5, 20.0, 2.0), 'M12': (13.0, 24.0, 2.5),
          'M16': (17.0, 30.0, 3.0), 'M20': (21.0, 37.0, 3.0)}
# ISO 7380: head diameter, head height, socket across flats
BUTTON = {'M3': (5.7, 1.65, 2.0), 'M4': (7.6, 2.2, 2.5), 'M5': (9.5, 2.75, 3.0), 'M6': (10.5, 3.3, 4.0), 'M8': (14.0, 4.4, 5.0)}
# ISO 10642: head diameter, socket across flats
CSK = {'M3': (6.72, 2.0), 'M4': (8.96, 2.5), 'M5': (11.2, 3.0), 'M6': (13.44, 4.0), 'M8': (17.92, 5.0)}
# ISO 4026: socket across flats
SET = {'M2': 0.9, 'M2.5': 1.3, 'M3': 1.5, 'M4': 2.0, 'M5': 2.5, 'M6': 3.0, 'M8': 4.0}


def dia(size):
    return float(size[1:])


def _revolve(profile):
    """A solid of revolution about z from a closed (r, z) outline in the xz plane."""
    pts = []
    for r, z in profile:
        if not pts or abs(pts[-1][0] - r) > 1e-6 or abs(pts[-1][1] - z) > 1e-6:
            pts.append((r, z))
    if abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6:
        pts.pop()
    face = make_face(Plane.XZ * Polyline(*pts, close=True))
    return revolve(face, Axis.Z)


def threaded_rod(d, length, pitch, z_top=0.0, threads=True, tip_chamfer=True, plain=0.0):
    """A shank of diameter d from z_top down to z_top - length. plain is the unthreaded length under the head; the
    rest carries rolled rings at the pitch, ending in a 45 degree chamfer."""
    r = d / 2
    r_min = r - 0.6134 * pitch          # the basic minor radius of an ISO thread
    zb = z_top - length
    c = min(pitch * 0.8, length * 0.2) if tip_chamfer else 0.0
    prof = [(0, z_top), (r, z_top)]
    if not threads or length - plain < 3 * pitch:
        prof += [(r, zb + c), (r - c, zb), (0, zb)]
        return _revolve(prof)
    zt = z_top - plain                  # threads start here going down
    prof.append((r, zt))
    z = zt
    n = int((zt - (zb + c + pitch)) / pitch)
    for _ in range(n):
        prof += [(r_min, z - pitch * 0.5), (r, z - pitch)]
        z -= pitch
    prof += [(r, zb + c), (r - c, zb), (0, zb)]
    return _revolve(prof)


def _socket(across, depth, z_face):
    """A hex socket cut into a face at z_face going down: the hexagon, then the drill point's cone."""
    hexa = extrude(Plane.XY.offset(z_face - depth) * RegularPolygon(across / math.sqrt(3), 6, major_radius=True), depth + 0.01)
    cone = Solid.make_cone(across / math.sqrt(3) * 0.95, 0.0, across * 0.35, Plane.XY.offset(z_face - depth).rotated((180, 0, 0)))
    return hexa + cone


def socket_cap(size, length, threads=None, socket=True):
    """ISO 4762 socket head cap screw: knurl free cylindrical head with chamfered top, hex socket, threads below."""
    dk, k, s, t = SHCS[size]
    d = dia(size)
    threads = d >= 4 if threads is None else threads
    head = Solid.make_cylinder(dk / 2, k)
    head = chamfer(head.edges().group_by(Axis.Z)[-1], dk * 0.07)
    head = fillet(head.edges().group_by(Axis.Z)[0], min(0.2 * d, 0.4) * 0.5)
    if socket:
        head = head - _socket(s, t, k)
    shank = threaded_rod(d, length, PITCH[size], 0.0, threads, plain=max(0.0, length - 2 * d - 6) if length > 2 * d + 6 else 0.0)
    return head + shank


def hex_bolt(size, length, threads=True, marks=True, thread_len=None):
    """ISO 4017 / 4014 hex bolt: 30 degree chamfered hex head on a washer face, threads to thread_len (all of it by default)."""
    s, k, _ = HEX[size]
    d = dia(size)
    head = hex_prism(s, k, chamfer_both=False)
    dw = s * 0.92
    face = Solid.make_cylinder(dw / 2, 0.06 * k * 2)
    head = head + face
    plain = length - thread_len if thread_len else 0.0
    shank = threaded_rod(d, length, PITCH[size], 0.0, threads, plain=plain)
    out = head + shank
    if marks:
        # the grade marks pressed into the top of a property class 8.8 head
        for a in (90, 210, 330):
            m = Pos(0, 0, k) * Rot(0, 0, a) * Pos(s * 0.22, 0, 0) * Box(s * 0.16, s * 0.035, 0.12)
            out = out - m
    return out


def hex_prism(across, height, chamfer_both=True, z0=0.0):
    """A hex prism across flats standing from z0, its corners turned off at 30 degrees like a cold formed head or nut."""
    rc = across / math.sqrt(3)
    prism = extrude(Plane.XY.offset(z0) * RegularPolygon(rc, 6, major_radius=True), height)
    # the 30 degree chamfer cone: a cone that clips the corners at both ends
    cut_h = (rc - across / 2 * 0.95) * math.tan(math.radians(30))
    top = Solid.make_cone(across / 2 * 0.95 + height / math.tan(math.radians(30)), across / 2 * 0.95, height,
                          Plane.XY.offset(z0))
    keep = top & prism
    if chamfer_both:
        bot = Solid.make_cone(across / 2 * 0.95, across / 2 * 0.95 + height / math.tan(math.radians(30)), height,
                              Plane.XY.offset(z0))
        keep = keep & bot
    return keep if cut_h > 0 else prism


def hex_nut(size, nylock=False):
    """ISO 4032 hex nut on z = 0, both faces chamfered, with its thread suggested inside by rings."""
    s, _, m = HEX[size]
    d = dia(size)
    nut = hex_prism(s, m)
    p = PITCH[size]
    r = d / 2
    r_min = r - 0.6134 * p
    prof = [(0, -0.01), (r_min + 0.25 * p, -0.01)]
    z = 0.0
    while z + p < m:
        prof += [(r_min, z + p * 0.5), (r_min + 0.5 * p, z + p)]
        z += p
    prof += [(r_min + 0.25 * p, m + 0.01), (0, m + 0.01)]
    nut = nut - _revolve(prof)
    nut = nut - Solid.make_cone(r + 0.3, r_min, r + 0.3 - r_min, Plane.XY.offset(m).rotated((180, 0, 0)))
    nut = nut - Solid.make_cone(r + 0.3, r_min, r + 0.3 - r_min, Plane.XY.offset(-0.0))
    if nylock:
        ring = Solid.make_cylinder(s * 0.47, m * 0.35, Plane.XY.offset(m)) - Solid.make_cylinder(r_min, m, Plane.XY.offset(m - 0.1))
        return nut + ring
    return nut


def washer(size, outer=None, thickness=None, inner=None):
    """ISO 7089 flat washer on z = 0 with the outer edge broken."""
    d1, d2, h = WASHER[size]
    d1, d2, h = inner or d1, outer or d2, thickness or h
    w = Solid.make_cylinder(d2 / 2, h) - Solid.make_cylinder(d1 / 2, h * 3, Plane.XY.offset(-h))
    try:
        w = chamfer(w.edges().filter_by_position(Axis.Z, h - 1e-6, h + 1e-6).sort_by(Axis.X)[-1:], h * 0.25)
    except Exception:
        pass
    return w


def button_head(size, length, threads=False):
    """ISO 7380 button head socket screw: a domed head with its hex socket."""
    dk, k, s = BUTTON[size]
    d = dia(size)
    n = 10
    prof = [(0, 0), (dk / 2, 0)]
    for i in range(1, n + 1):
        a = i / n * math.pi / 2
        prof.append((dk / 2 * math.cos(a) * (1 - 0.0) if i < n else 0.0, k * math.sin(a)))
    head = _revolve(prof[:-1] + [(0, k)])
    head = head - _socket(s, k * 0.6, k)
    return head + threaded_rod(d, length, PITCH[size], 0.0, threads)


def countersunk(size, length, threads=False):
    """ISO 10642 countersunk socket screw with its head flush at z = 0 and sunk below it (90 degree head)."""
    dk, s = CSK[size]
    d = dia(size)
    kh = (dk - d) / 2
    head = _revolve([(0, 0), (dk / 2, 0), (d / 2, -kh), (0, -kh)])
    head = head - _socket(s, kh * 0.9, 0.0)
    return head + threaded_rod(d, length - kh, PITCH[size], -kh, threads)


def set_screw(size, length):
    """ISO 4026 / 4029 set screw: hex socket on its top at z = 0 and a cup point at -length."""
    d = dia(size)
    p = PITCH[size]
    r = d / 2
    c = p * 0.5
    prof = [(0, 0), (r - c, 0), (r, -c), (r, -length + c), (r * 0.7, -length), (r * 0.35, -length), (r * 0.3, -length + 0.25 * r), (0, -length + 0.25 * r)]
    s = _revolve(prof)
    return s - _socket(SET[size], min(length * 0.5, d * 0.6), 0.0)


def pan_phillips(d, length, head_d=None, wood=False):
    """A pan head cross recessed screw (ISO 7045 for machine screws, or a wood screw with its coarse sharp thread)."""
    dk = head_d or d * 2.0
    k = d * 0.7
    prof = [(0, 0), (dk / 2, 0), (dk / 2, k * 0.55), (dk / 2 * 0.86, k * 0.92), (dk / 2 * 0.6, k), (0, k)]
    head = _revolve(prof)
    w = d * 0.22
    for a in (0, 90):
        head = head - Pos(0, 0, k) * Rot(0, 0, a) * Box(dk * 0.5, w, k * 1.1)
    head = head - Solid.make_cone(dk * 0.2, 0, dk * 0.25, Plane.XY.offset(k).rotated((180, 0, 0)))
    if wood:
        p = d * 0.45
        r = d / 2
        rc = r * 0.62
        prof = [(0, 0), (rc, 0)]
        z = -p * 0.5
        tip = -length + d * 1.2
        while z - p > tip:
            prof += [(r, z - p * 0.15), (rc, z - p)]
            z -= p
        prof += [(rc * 0.9, tip), (0.05, -length), (0, -length)]
        shank = _revolve(prof)
    else:
        shank = threaded_rod(d, length, 0.5 if d <= 3 else 0.7, 0.0, False)
    return head + shank


def wall_plug(d, length):
    """A ribbed nylon wall plug (an anchor for a screw of about d * 0.75) with its collar on z = 0 going into -z."""
    r = d / 2
    prof = [(0, 0.0), (r * 1.25, 0.0), (r * 1.25, 0.6), (r * 0.5, 0.6)]
    prof = [(r * 0.45, 0.6), (r * 1.3, 0.6), (r * 1.3, 0.0)]
    z = 0.0
    n = int(length / (d * 0.55))
    for i in range(n):
        prof += [(r, z - d * 0.12), (r * 1.08, z - d * 0.4), (r, z - d * 0.55)]
        z -= d * 0.55
    prof += [(r * 0.8, -length), (r * 0.45, -length)]
    return _revolve(prof)


def bolt_stack(size, length, grip, nut=True, washers=(True, True), head='hex', threads=True):
    """A through bolt clamping a joint grip mm thick, as [(shape, material)]: the head on a washer at z = 0 (the top
    face of the joint, washer included), a washer and a nut under the joint at -grip. Steel, zinc plated."""
    out = []
    d1, d2, h = WASHER[size]
    z_head = h if washers[0] else 0.0
    bolt = hex_bolt(size, length, threads) if head == 'hex' else socket_cap(size, length, threads)
    out.append((Pos(0, 0, z_head) * bolt, 'steel_zinc'))
    if washers[0]:
        out.append((washer(size), 'steel_zinc'))
    zb = -grip
    if washers[1]:
        out.append((Pos(0, 0, zb - h) * washer(size), 'steel_zinc'))
        zb -= h
    if nut:
        m = HEX[size][2]
        out.append((Pos(0, 0, zb - m) * hex_nut(size), 'steel_zinc'))
    return out
