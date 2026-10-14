# Circuits

Will this LED fry, what is the ripple and how hot does the MOSFET get?

These are three everyday circuit questions, each answered with ngspice and checked against the
textbook answer computed independently.

```sh
./run.sh
```

| Question | Answer | Checks |
|---|---|---|
| Will a blue LED on 5 V with a 47 ohm resistor fry? | Yes, 36.9 mA goes through a 30 mA LED, and Wobbl suggests 91 ohm for about 20 mA | the diode equation by hand gives 36.9 mA and ngspice 36.9 mA, and the supply's power equals the resistor's plus the LED's |
| What is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz, 10 uH, 22 uF? | 8.56 mV peak to peak with 0.60 A of inductor ripple, within a 50 mV limit | the ideal triangular ripple math gives 7.95 mV, 8% apart because of the switch and diode drops, the output reached steady state, and the ripple is the same at half the time step |
| How hot does a MOSFET get switching 15 A at 20 kHz in a 50 C box with 40 C/W to air? | About 110 C at the junction from 1.51 W, under the 150 C limit with margin | the conduction loss `I_rms^2 R` matches the simulated V x I to 0.001%, and the junction temperature is iterated with the on resistance rising 0.6% per C |

Each folder (`led/`, `buck/`, `mosfet/`) holds `report.html`, `summary.txt`, `outcome.json` and the
netlist Wobbl generated. Its `images/` folder has `setup.svg`, the circuit wired up on a bench with
the run's own readings, `schematic.svg`, drawn from the netlist, and a waveform plot where the
analysis is in time.
