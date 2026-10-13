# Enclosure

Does a polycarbonate shell survive a thumb press, and how far does it dent?

This is the top shell of a handheld device, 90 x 60 x 28 mm with 3 mm walls and four screw
bosses, held at its rim. A thumb presses 120 N on a 16 mm patch near one corner.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress away from sharp corners | 8.77 MPa on the top skin under the press, safety factor 7.1 against PC's 62 MPa yield |
| Dent under the thumb | 0.22 mm |
| Mesh convergence | 8.67, 8.82, 8.77 MPa over three meshes (114k to 166k nodes) |
| Equilibrium | reactions balance the 120 N to 0.00002% |

Confidence is medium because the bosses meet the underside of the top with no fillet, and Wobbl
flags those sharp corners as singular. One boss root reads 15.2, 20.8 and 20.6 MPa over the
three meshes, a value the mesh sets rather than the part. The shell holds this press with a wide
margin either way, but a 0.5 to 1 mm fillet at the boss roots is the cheap change that makes the
corner stress finite and trustworthy.

Thin walls on a box need over 100k nodes per mesh, so the study takes about four and a half
minutes on a 2018 six core laptop.
