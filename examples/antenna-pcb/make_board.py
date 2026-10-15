"""Write ifa.kicad_pcb, a 2.4 GHz inverted F trace antenna on a 1.6 mm FR4 board, as a KiCad 8 file.

The arm is a 0.8 mm trace along the board's top edge, shorted to the ground pour at one end and fed by a stub
that stops 0.35 mm short of the pour, where the feed pad sits. Both layers carry ground under the board's lower
part, and the antenna's end is left bare of copper on both, as a real module's keep out is.

    python make_board.py      writes ifa.kicad_pcb next to this file (the same as the file in the repository)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NETS = ["", "GND", "ANT"]
W, H = 40.0, 30.0            # board size, mm (y is down as in KiCad)
Y_GND = 12.0                 # the pour starts here, and the antenna end lies above it
ARM_Y, ARM_X0, ARM_X1 = 7.0, 6.0, 24.0
FEED_X = 11.0


def segment(a, b, w, net):
    return (f'  (segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "F.Cu") '
            f'(net {NETS.index(net)}))\n')


def zone(layer):
    pts = f"(xy 0 {Y_GND}) (xy {W} {Y_GND}) (xy {W} {H}) (xy 0 {H})"
    return (f'  (zone (net 1) (net_name "GND") (layer "{layer}") (hatch edge 0.5)\n'
            f'    (connect_pads yes (clearance 0.3)) (min_thickness 0.25)\n'
            f'    (polygon (pts {pts}))\n    (filled_polygon (layer "{layer}") (pts {pts}))\n  )\n')


def build() -> str:
    head = ['(kicad_pcb (version 20240108) (generator "wobbl example")\n', '  (general (thickness 1.6))\n',
            '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))\n',
            '  (setup (stackup\n'
            '    (layer "F.Cu" (type "copper") (thickness 0.035))\n'
            '    (layer "dielectric 1" (type "core") (thickness 1.53) (material "FR4") (epsilon_r 4.4) '
            '(loss_tangent 0.02))\n'
            '    (layer "B.Cu" (type "copper") (thickness 0.035))))\n']
    head += [f'  (net {i} "{n}")\n' for i, n in enumerate(NETS)]
    edge = [f'  (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))\n'
            for a, b in [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))]]
    feed_pad = (f'  (footprint "wobbl:FEED" (layer "F.Cu") (at {FEED_X} {Y_GND - 0.6})\n'
                f'    (property "Reference" "J1" (at 0 -2 0) (layer "F.SilkS"))\n'
                f'    (pad "1" smd rect (at 0 0) (size 0.8 0.5) (layers "F.Cu" "F.Paste" "F.Mask") (net 2 "ANT"))\n'
                f'  )\n')
    arm = segment((ARM_X0, ARM_Y), (ARM_X1, ARM_Y), 0.8, "ANT")
    short = segment((ARM_X0, ARM_Y), (ARM_X0, Y_GND + 0.4), 0.8, "ANT")
    feed = segment((FEED_X, ARM_Y), (FEED_X, Y_GND - 0.6), 0.8, "ANT")
    return "".join(head + edge + [zone("F.Cu"), zone("B.Cu"), feed_pad, arm, short, feed]) + ")\n"


if __name__ == "__main__":
    (HERE / "ifa.kicad_pcb").write_text(build())
    print("wrote", HERE / "ifa.kicad_pcb")
