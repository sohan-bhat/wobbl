# Steel hook

Will a mild steel hook bend for good under a 3 kN sling, and what load pulls it open?

This is a forged J hook with a 12 by 10 mm section, bolted up through the 8.5 mm eye in its shank. A
sling hangs in the bottom of the 22 mm bend. A strength study would stop at "the peak passes yield",
which says little about a ductile part: steel yields in a small spot long before the hook gives way.
The plasticity study answers the questions that matter instead. Where does it first yield, how far
does it stay bent once the sling comes off, and what load opens it up?

```sh
./run.sh
```

| Result | Value |
|---|---|
| First yield | 0.70 times the load, 2.1 kN, on the inside of the bend where it leaves the shank, where the curve concentrates the bending |
| Under the 3 kN sling | small patches yield there and beside the eye, with a peak plastic strain of 0.14% |
| Permanent set | 0.033 mm once the sling comes off, too small to see |
| Collapse | 1.34 times the load, 4.0 kN, where the whole shank section yields and the hook opens |
| Mesh convergence | collapse at 1.33 then 1.34 times the load over two meshes |
| Hand calculation | a fully plastic 12 by 10 mm section carrying 66 N m of bending and 3 kN of tension, (M / Mp) + (N / Np)^2 = 1, collapses at 1.34 times the load, within 1% of the finite element answer |
| Balance | reactions match the loads to 0.0001% at collapse, and nothing is left on the supports once the load is off |

So the hook does yield under 3 kN, in small patches, but it springs back to within a few hundredths of a
millimeter, and it takes 4 kN to pull it open. That is only 1.34 times the working load, under the 1.5
margin most steel lifting gear asks for, so the verdict is MARGINAL. A section about 12.7 mm deep instead
of 12, or a steel that yields above 279 MPa, reaches the margin, and so does keeping the sling under
2.7 kN.

The run writes `output/scene.wobbl`, the 3D view embedded in `output/report.html`, where the hook takes
the load and lets it go increment by increment with the plastic zone colored as it spreads, then is
ramped until it opens. `output/playback.json` holds the same frames as plain arrays.
