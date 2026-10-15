# Enclosure

Does a polycarbonate shell survive a thumb press, and how far does it dent?

This is the top shell of a handheld device, 90 x 60 x 28 mm with 3 mm walls and four screw
bosses, held at its rim. A thumb presses 120 N on a 16 mm patch near one corner.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Peak stress | 9.77 MPa in the top wall under the press, safety factor 6.3 against PC's 62 MPa yield |
| Dent under the thumb | 0.23 mm |
| Mesh convergence | 9.70, 9.53, 9.77 MPa over three meshes |
| Equilibrium | reactions balance the 120 N to 4e-10% |

Wobbl solves the walls as shells, with the four bosses as beams, which carries all of the part's
material. A coarse solid model run beside it gives 0.208 mm of dent and 8.3 MPa near the same
spot, 6% and 17% from the shell, so confidence is medium and the numbers carry that margin. The
shell holds this press with a wide margin either way.

The study takes about a minute and a half on a 2018 six core laptop.
