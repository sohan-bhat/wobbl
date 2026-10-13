# Enclosure: a thumb pressing on a polycarbonate shell

The top shell of a handheld device, 90 x 60 x 28 mm with 3 mm walls and four
screw bosses, sitting on its rim. A thumb presses 120 N on a 16 mm patch near
one corner.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress away from sharp corners | 8.85 MPa on the top skin next to the press, safety factor 7.0 against PC's 62 MPa yield |
| Dent under the thumb | 0.22 mm |
| Mesh convergence | 8.57, 8.82, 8.85 MPa over three meshes (73k to 110k nodes) |
| Equilibrium | reactions balance the 120 N to 0.00003% |

Confidence is medium for a clear reason: the bosses meet the underside of the
top with no fillet, and Wobbl flags those sharp corners as singular (14.9, 18.2,
18.6 MPa and still rising as the mesh refines). The shell holds this press with
a wide margin either way, but a 0.5 to 1 mm fillet at the boss roots is the cheap
change that makes the corner stress finite and trustworthy.

This is the slowest example: thin walls on a box need around 100k nodes per
mesh, and the study takes about four minutes.
