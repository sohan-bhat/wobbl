# Strut buckling

Does a slender aluminum strut carry 7 kN with a factor of 3 against buckling?

This is a 6061-T6 tube 600 mm long, 20 mm across with a 3 mm wall, the kind that braces a frame. Its
base is clamped and its top is held sideways by the part it pushes on, free to turn, and the top
carries 7 kN straight down. A strut that slender fails by buckling long before the aluminum yields,
and a real tube is never perfectly straight, so the factor that counts is the one after a knockdown
for crookedness.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Linear buckling | 3.25 times the load, 22.8 kN, in a bow that is the same in every direction across the tube |
| Mesh convergence | 3.253 then 3.252 over two meshes |
| Hand calculation | Euler's column clamped at one end and pinned at the other (K = 0.699) gives 22.6 kN with the section the mesh cuts, 0.5% under the finite element answer |
| Knockdown | the EN 1999-1-1 column curve for class A alloys, relative slenderness 1.39, which keeps 0.42 of the squash load of 44.3 kN |
| Design factor | 2.65 against the target of 3, so MARGINAL |
| Yield | the most stressed point would only reach yield at 5.4 times the load, so buckling governs |

The tube buckles at more than three times the load on paper, but a part with an ordinary amount of
crookedness carries only 2.65 times it, because the bow it starts with grows as the load approaches
the buckling load. Wobbl solves the column curve backwards for the fixes. About 1.2 times the bending
stiffness gets there, as does a strut 554 mm long, and a brace that holds it sideways at mid length
does far more, since it halves the buckling length.

The report opens with a 3D view of the tube with a load slider up to 95% of the buckling load, the stress
coloring the tube and its bow grown from a crookedness of a thousandth of its length, saved as
`output/scene.wobbl`. The run also writes `output/playback.json` with the tube's surface, its stress, its
first buckling shape and the bow at each load.
