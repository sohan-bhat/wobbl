# Drone arm resonance

Does a printed drone arm resonate with its propeller anywhere between hover and full throttle?

This is the PA12 arm from the [drone-arm](../drone-arm) example, bolted into the frame through the two
holes in its root block, now carrying what sits on its pad, an 80 g motor and propeller with their center
17 mm above the pad. The propeller turns from 5000 rpm at hover to 12000 rpm at full throttle, so any
imbalance pushes the arm once per turn, from 83 to 200 Hz, and the two blades push it each time one
passes, from 167 to 400 Hz. The usual rule keeps every natural frequency at least 20% clear of both.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Verdict | fails, because two modes sit inside the excitation bands |
| Mode 1, side to side bending | 30.4 Hz, below hover |
| Mode 2, up and down bending | 43.7 Hz, below hover |
| Mode 3, the arm twisting under the motor | 196 Hz, inside both bands |
| Mode 4, the motor rocking along the arm | 219 Hz, inside the blade pass band and 10% above once per turn, moving 57% of the mass |
| Mesh convergence | 221 then 219 Hz for mode 4, a change of 1.0% |
| Mass balance | 120 g of arm and motor, with every mode at unit modal mass to 0.0003% |
| Hand calculation | Rayleigh's quotient from the sag under 1 g gives 30.4 Hz, the same as mode 1 |

The two bending modes sit below hover, which is fine as long as the motor spins up through them
quickly. The trouble is the motor itself, which sits high on a narrow pad and twists and rocks the
thin tip of the arm near 200 Hz, right where both the imbalance and the blade pass push. Wobbl
suggests stiffening the tip enough to lift mode 3 above 480 Hz, which takes about six times the
stiffness, or mounting it softly enough to drop under hover. A wider and deeper pad with the motor
sitting lower is usually the cheaper way to stiffen that tip.

The report opens with a 3D view of the four mode shapes nearest the excitation, each oscillating in
turn with its frequency beside it, saved as `output/scene.wobbl`. The run also writes
`output/playback.json` with the arm's surface and those mode shapes, for the website to play back.
