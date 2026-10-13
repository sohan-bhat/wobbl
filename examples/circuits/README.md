# Circuits: an LED, a buck converter and a MOSFET

Three everyday questions, each answered with ngspice and checked against the
textbook answer computed independently.

```sh
./run.sh
```

| Question | Answer | Checks |
|---|---|---|
| Will a blue LED on 5 V with a 47 ohm resistor fry? | Yes: 36.9 mA through a 30 mA LED. Wobbl suggests 91 ohm for about 20 mA. | diode equation by hand 36.9 mA, ngspice 36.9 mA; supply power equals resistor plus LED power |
| What is the ripple on a 12 V to 5 V buck at 2 A, 500 kHz, 10 uH, 22 uF? | 8.56 mV peak to peak, inductor ripple 0.60 A, within a 50 mV limit | ideal triangular ripple math 7.95 mV (8% apart, switch and diode drops explain it); steady state reached; same ripple at half the time step |
| How hot does a MOSFET get switching 15 A at 20 kHz in a 50 C box with 40 C/W to air? | About 110 C junction (1.51 W), under the 150 C limit with margin | conduction loss I_rms^2 R matches the simulated V x I to 0.001%; junction temperature iterated with on resistance rising 0.6% per C |

Each folder (`led/`, `buck/`, `mosfet/`) holds `report.html`, `summary.txt`,
`outcome.json`, the netlist Wobbl generated, and a waveform plot where the
analysis is in time.
