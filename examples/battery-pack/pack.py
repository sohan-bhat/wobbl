"""A 2S2P pack of 18650 cells treated as one block with effective properties.

Four cells side by side with their nickel strips and tape act thermally like a
solid block about 37 x 37 x 65 mm. Coordinates in mm.
"""

PARAMS = {"size": 37.0, "length": 65.0}


def build(occ, size, length):
    occ.addBox(0, 0, 0, size, size, length)
