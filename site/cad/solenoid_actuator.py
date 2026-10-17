"""Context models for the pull solenoid (examples/solenoid-actuator): the run's three solids drawn as the real
parts, a zinc plated C frame with its pole piece and tapped holes, the coil on its bobbin with winding turns, tape
and leads, and the plated plunger with its clevis and E clip, all to the run's STEP dimensions. The field slice
keeps its colors and shows through them.

    python site/cad/solenoid_actuator.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import actuator as ac  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'solenoid-actuator', 'output', 'scene.wobbl')


def main():
    sc = Scene(RUN)
    for k in (1, 2, 3):
        assert sc.find(f'actuator {k}') and sc.matrix(sc.find(f'actuator {k}')[0])[0, 3] == 0
    export('solenoid-actuator/frame.glb', [Node('frame', ac.frame())], tol=0.02, ang=0.3)
    export('solenoid-actuator/coil.glb', [Node('coil', ac.coil(leads=[(-8, 36, 20), (-5, 36, 18.5)]))], tol=0.02, ang=0.3)
    export('solenoid-actuator/plunger.glb', [Node('plunger', ac.plunger())], tol=0.01, ang=0.3)
    write_manifest('solenoid-actuator', {'replace': [
        {'object': 'actuator 1', 'file': 'solenoid-actuator/frame.glb'},
        {'object': 'actuator 2', 'file': 'solenoid-actuator/coil.glb'},
        {'object': 'actuator 3', 'file': 'solenoid-actuator/plunger.glb'}]})


if __name__ == '__main__':
    main()
