# Shelf bracket: sharp corner versus fillet versus gusset

A PLA L bracket printed flat, screwed to a wall through two holes, with the shelf
pressing 40 N on the arm 65 mm from the wall. The inside corner is what a
printer leaves on a "sharp" corner: a 0.5 mm radius.

```sh
./run.sh
```

| Version | Peak stress | Safety factor (PLA, 50 MPa) | Verdict |
|---|---|---|---|
| As drawn, 0.5 mm corner | 52.8 MPa | 0.95 | fails |
| 3 mm fillet | 24.5 MPa | 2.04 | marginal (target 2.5) |
| Legs thickened to 10 mm | 21.5 MPa | 2.32 | marginal |
| Gusset across the corner | 10.3 MPa | 4.84 | holds |

What Wobbl checked on the original:

- mesh convergence over four meshes: 44.2, 51.1, 53.4, 52.8 MPa
- reactions balance the 40 N load to 0.0001%
- beam theory through the corner gives 15.2 MPa nominal, so the corner multiplies stress by about 3.5
- the screws are modeled as bolted holes and the back of the upright can push on the wall but not pull,
  solved as contact

The thicker legs barely help because the sharp corner still multiplies the stress;
rounding the corner or bracing it is what works. The gusset's own toe is a sharp
corner, which Wobbl excludes as a singularity and flags (medium confidence); give
it a small fillet in the real part. Files: `output/report.html`,
`output/images/stress.png`, and the re-simulated versions in `output/fix-*/`.
