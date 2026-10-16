# Circuits

Will this LED fry, what is the ripple, how hot does the MOSFET get, does anything on this board
overheat, and is this buffer stable?

The first three are template questions, each answered with ngspice and checked against the
textbook answer computed independently. The last two are a circuit of the user's own: a KiCad
schematic of a small board and a SPICE netlist. Wobbl reads them, gives every part its library
model and ratings, picks the analyses, judges every part, checks itself and solves its fixes
again before offering them.

```sh
./run.sh
```

| Question | Answer | Checks |
|---|---|---|
| Will a blue LED on 5 V with a 47 ohm resistor fry? | Yes, 36.9 mA goes through a 30 mA LED, and Wobbl suggests 91 ohm for about 20 mA | the diode equation by hand gives 36.9 mA and ngspice 36.9 mA, and the supply's power equals the resistor's plus the LED's |
| What is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz, 10 uH, 22 uF? | 8.56 mV peak to peak with 0.60 A of inductor ripple, within a 50 mV limit | the ideal triangular ripple math gives 7.95 mV, 8% apart because of the switch and diode drops, the output reached steady state, and the ripple is the same at half the time step |
| How hot does a MOSFET get switching 15 A at 20 kHz in a 50 C box with 40 C/W to air? | About 110 C at the junction from 1.51 W, under the 150 C limit with margin | the conduction loss `I_rms^2 R` matches the simulated V x I to 0.001%, and the junction temperature is iterated with the on resistance rising 0.6% per C |
| Will anything on [this board](kicad-board.kicad_sch) overheat with a 2 A LED strip on J2? (a 12 V jack, an L7805, an ATtiny85 lighting an LED and switching an IRLZ44N) | No. The hardest working part is C1 at 53% of its 25 V rating, the IRLZ44N runs at 33 C and the 7805 at 32 C | KCL, the power account, half the time step, four hand calculations (the gate divider, the LED's diode equation, the regulator's dissipation and the MOSFET's I_rms^2 R loss) and both ends of the supply range |
| Is [this LM358 buffer](opamp-cable.cir) stable driving a cable of about 1 uF? | No. Its loop gain crosses 1 at 47 kHz with 2 degrees of phase margin, so it oscillates. Wobbl adds 100 ohm between the output and the cable with the feedback taken before it, sweeps the loop again and gets 72 degrees | the loop gain is Bode's return ratio of the op amp's gain stage, cut for AC only, and the fix is solved again before it is offered |

Each folder holds `report.html`, `summary.txt`, `outcome.json`, the netlist Wobbl solved,
`playback.json` with every part's voltage, current and power over the run for the site to animate,
and the bench and schematic drawings in `images/`.

Each run also has a 3D bench you can work with (`wobbl view led`). A slider sets the one input its
question points at and the circuit is solved again at every step. The LED run moves the series
resistor from 15 to 470 ohm, the buck and MOSFET runs move the load current, the board moves the
current of the strip on J2 up to 6 A, and the buffer moves the cable's capacitance from 1 nF to
10 uF. Each part is shaded by its share of its limit, and the LED glows with its current.
