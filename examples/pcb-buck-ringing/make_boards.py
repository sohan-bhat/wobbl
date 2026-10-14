"""Write the two buck converter layouts of this example as KiCad 8 board files.

Both boards carry the same parts and nets as buck.cir: the input connector J1, the input capacitor C1, the switch
S1, the catch diode D1, the inductor L1, the output capacitor C2 and the load R1. The loose layout spreads them
over a two layer board with the ground return routed as tracks on the bottom. The tight layout packs the hot loop
(C1, S1, D1) together over a solid ground pour on the bottom layer.

    python make_boards.py      writes buck_loose.kicad_pcb and buck_tight.kicad_pcb next to this file
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NETS = ["", "VIN", "SW", "VOUT", "GND", "G"]


def pad(num, net, x, y, w, h):
    n = NETS.index(net)
    return (f'    (pad "{num}" smd rect (at {x} {y}) (size {w} {h}) (layers "F.Cu" "F.Paste" "F.Mask") '
            f'(net {n} "{net}"))\n')


def footprint(ref, x, y, pads):
    body = "".join(pad(*p) for p in pads)
    return (f'  (footprint "wobbl:{ref}" (layer "F.Cu") (at {x} {y})\n'
            f'    (property "Reference" "{ref}" (at 0 -2 0) (layer "F.SilkS"))\n{body}  )\n')


def track(points, w, layer, net):
    """Tracks through a list of points, KiCad's y down as in the editor."""
    return "".join(f'  (segment (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (width {w}) (layer "{layer}") '
                   f'(net {NETS.index(net)}))\n' for a, b in zip(points, points[1:]))


def via(x, y, net):
    return f'  (via (at {x} {y}) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net {NETS.index(net)}))\n'


def zone(net, layer, x0, y0, x1, y1):
    pts = f"(xy {x0} {y0}) (xy {x1} {y0}) (xy {x1} {y1}) (xy {x0} {y1})"
    return (f'  (zone (net {NETS.index(net)}) (net_name "{net}") (layer "{layer}") (hatch edge 0.5)\n'
            f'    (connect_pads yes (clearance 0.3)) (min_thickness 0.25)\n'
            f'    (polygon (pts {pts}))\n    (filled_polygon (layer "{layer}") (pts {pts}))\n  )\n')


def board(name, items, w, h):
    head = ['(kicad_pcb (version 20240108) (generator "wobbl example")\n', '  (general (thickness 1.6))\n',
            '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))\n',
            '  (setup (stackup\n'
            '    (layer "F.Cu" (type "copper") (thickness 0.035))\n'
            '    (layer "dielectric 1" (type "core") (thickness 1.51) (material "FR4") (epsilon_r 4.5) '
            '(loss_tangent 0.02))\n'
            '    (layer "B.Cu" (type "copper") (thickness 0.035))))\n']
    head += [f'  (net {i} "{n}")\n' for i, n in enumerate(NETS)]
    edge = [f'  (gr_line (start {a[0]} {a[1]}) (end {b[0]} {b[1]}) (layer "Edge.Cuts") (width 0.1))\n'
            for a, b in [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))]]
    (HERE / name).write_text("".join(head + edge + items) + ")\n")


def two_pad(ref, x, y, a, b, pitch=3.2, size=(1.6, 2.6)):
    return footprint(ref, x, y, [("1", a, -pitch / 2, 0, *size), ("2", b, pitch / 2, 0, *size)])


def switch(x, y):
    """S1 pins in the order of the netlist's switch: VIN, SW, the gate and ground."""
    return footprint("S1", x, y, [("1", "VIN", -1.9, -1.0, 1.2, 0.8), ("2", "SW", 1.9, -1.0, 1.2, 0.8),
                                  ("3", "G", -1.9, 1.0, 1.2, 0.8), ("4", "GND", 1.9, 1.0, 1.2, 0.8)])


def loose():
    items = [two_pad("J1", 6, 10, "VIN", "GND"), two_pad("C1", 14, 10, "VIN", "GND"), switch(30, 10),
             two_pad("D1", 46, 22, "GND", "SW"), two_pad("L1", 60, 10, "SW", "VOUT", 5, (3, 4)),
             two_pad("C2", 74, 18, "VOUT", "GND"), two_pad("R1", 84, 18, "VOUT", "GND")]
    items += [track([(4.4, 10), (4.4, 6), (28.1, 6), (28.1, 9)], 1.0, "F.Cu", "VIN"),
              track([(12.4, 6), (12.4, 10)], 1.0, "F.Cu", "VIN"),
              track([(31.9, 9), (47.6, 9), (57.5, 9), (57.5, 10)], 1.0, "F.Cu", "SW"),
              track([(47.6, 9), (47.6, 22)], 1.0, "F.Cu", "SW"),
              track([(62.5, 10), (82.4, 10), (82.4, 18)], 1.0, "F.Cu", "VOUT"),
              track([(72.4, 10), (72.4, 18)], 1.0, "F.Cu", "VOUT"),
              track([(28.1, 11), (24, 11), (24, 20)], 0.4, "F.Cu", "G")]
    # ground runs on the bottom from every ground pad back to the connector, through vias
    for x, y in [(7.6, 10), (15.6, 10), (31.9, 11), (44.4, 22), (75.6, 18), (85.6, 18)]:
        yv = 13 if y < 13 else y + 3
        items += [track([(x, y), (x, yv)], 1.0, "F.Cu", "GND"), via(x, yv, "GND")]
        if yv != 13:
            items.append(track([(x, yv), (x, 13)], 1.0, "B.Cu", "GND"))
    items.append(track([(7.6, 13), (85.6, 13)], 1.0, "B.Cu", "GND"))
    board("buck_loose.kicad_pcb", items, 92, 30)


def tight():
    items = [two_pad("J1", 4, 8, "VIN", "GND"), two_pad("C1", 11, 8, "VIN", "GND"), switch(18, 8),
             two_pad("D1", 24, 13, "GND", "SW"), two_pad("L1", 32, 8, "SW", "VOUT", 5, (3, 4)),
             two_pad("C2", 40, 12, "VOUT", "GND"), two_pad("R1", 46, 12, "VOUT", "GND")]
    items += [track([(2.4, 8), (2.4, 5), (16.1, 5), (16.1, 7)], 2.0, "F.Cu", "VIN"),
              track([(9.4, 5), (9.4, 8)], 2.0, "F.Cu", "VIN"),
              track([(19.9, 7), (25.6, 7), (29.5, 7), (29.5, 8)], 2.0, "F.Cu", "SW"),
              track([(25.6, 7), (25.6, 13)], 2.0, "F.Cu", "SW"),
              track([(34.5, 8), (44.4, 8), (44.4, 12)], 2.0, "F.Cu", "VOUT"),
              track([(38.4, 8), (38.4, 12)], 2.0, "F.Cu", "VOUT"),
              track([(16.1, 9), (14, 9), (14, 16)], 0.4, "F.Cu", "G")]
    for x, y in [(5.6, 8), (12.6, 8), (19.9, 9), (22.4, 13), (41.6, 12), (47.6, 12)]:
        items += [track([(x, y), (x, y + 1.6)], 1.0, "F.Cu", "GND"), via(x, y + 1.6, "GND")]
    items.append(zone("GND", "B.Cu", 0.5, 0.5, 51.5, 19.5))
    board("buck_tight.kicad_pcb", items, 52, 20)


if __name__ == "__main__":
    loose()
    tight()
