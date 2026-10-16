"""An extruded aluminum heat sink: a flat base with parallel fins.

The base underside sits on z = 0 and the chip touches its center. Coordinates
in mm.
"""

PARAMS = {
    "base": 6.0,          # base thickness
    "size": 60.0,         # base is size x size
    "fins": 9,
    "fin": 1.5,           # fin thickness
    "fin_height": 30.0,
}


def build(occ, base, size, fins, fin, fin_height):
    parts = [(3, occ.addBox(0, 0, 0, size, size, base))]
    pitch = (size - fin) / (fins - 1)
    for i in range(int(fins)):
        parts.append((3, occ.addBox(i * pitch, 0, base, fin, size, fin_height)))
    occ.fuse(parts[:1], parts[1:])
