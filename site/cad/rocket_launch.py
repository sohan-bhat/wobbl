"""Context models for the dual deploy rocket on its rail (examples/rocket-launch): the airframe with its coupler
joint, avionics switch band, vents, shear pins, roll pattern, two 1010 rail buttons and a launch lug, the ogive nose
with its aluminium tip, three fins with rounded edges and root fillets, the motor retainer, aft closure and graphite
nozzle, the launch pad with its blast deflector and hinged rail mount, and the 1010 rail. Every length comes from the
run's scene.wobbl (metres), built here in millimetres.

    python site/cad/rocket_launch.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import rocketry as rk  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'rocket-launch', 'output', 'scene.wobbl')
EX = 'rocket-launch'
MM = 1000.0


def mesh_mm(sc, name):
    return sc.mesh(sc.find(name)[0])[0] * MM


def main():
    sc = Scene(RUN)
    body = mesh_mm(sc, 'rocket')
    R = float(np.hypot(body[:, 0], body[:, 1]).max())
    z_aft, z_top = float(body[:, 2].min()), float(body[:, 2].max())
    nose = mesh_mm(sc, 'nose')
    r = np.hypot(nose[:, 0], nose[:, 1])
    z = np.round(nose[:, 2], 3)
    profile = [(float(r[z == zz].max()), float(zz)) for zz in sorted(set(z))]
    profile = profile[:1] + profile[1:-1:max(1, len(profile) // 24)] + profile[-1:]
    fin = mesh_mm(sc, 'fins')[:8]
    ys, zs = fin[:, 1], fin[:, 2]
    root = fin[np.isclose(ys, ys.min())]
    tip = fin[np.isclose(ys, ys.max())]
    half = float(np.abs(fin[:, 0]).max())
    noz = mesh_mm(sc, 'nozzle')
    files = {}

    air = rk.airframe(R, z_aft, z_top)
    for zb in (z_aft + 110, -600.0):
        air += rk.rail_button(R, zb, 270)
    air += rk.launch_lug(R, -900.0, -760.0, 30)
    files['rocket'] = ('airframe', air, 0.05)
    files['nose'] = ('nose-cone', rk.ogive_nose(profile), 0.12)
    files['fins'] = ('fins', rk.fins(R, (float(root[:, 2].min()), float(root[:, 2].max())),
                                     (float(ys.max()), float(tip[:, 2].min()), float(tip[:, 2].max())), half), 0.05)
    files['nozzle'] = ('motor-tail', rk.motor_tail(float(noz[:, 2].max()), float(noz[:, 2].min())), 0.03)

    pi, po, PM = sc.named('launch pad')[0]
    ri, ro, RM = sc.named('launch rail')[0]
    w, L = ro['size'][0] * MM, ro['size'][2] * MM
    foot = RM[:3, 3] - RM[:3, 2] * ro['size'][2]
    lf = (np.linalg.inv(PM) @ np.array([*foot, 1.0]))[:3] * MM
    hx, hy, hz = (s * MM for s in po['size'])
    files['launch pad'] = ('launch-pad', rk.pad(hx, hy, hz, (lf[0], lf[1], 0.0)), 0.1)
    files['launch rail'] = ('rail', rk.rail_1010(w, L), 0.05)

    entries = []
    for obj, (name, parts, tol) in files.items():
        rel = f'{EX}/{name}.glb'
        # a two metre rocket needs 14 bit positions to keep its paint off the tube
        export(rel, [Node(name.replace('-', '_'), parts)], unit=1 / MM, tol=tol, ang=0.25, pack=('-vp', '14', '-vt', '10'))
        entries.append({'object': obj, 'file': rel})
    write_manifest(EX, {'replace': entries})


if __name__ == '__main__':
    main()
