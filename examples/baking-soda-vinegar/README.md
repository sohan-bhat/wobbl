# Baking soda and vinegar

How much gas does baking soda in vinegar make, and does it get warm or cold?

Wobbl works out the products from the reactants (an acid on a carbonate), balances the reaction and takes the
thermodynamics from NASA Glenn and NBS data, with the species in water treated as ideal.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Reaction | CH3COOH(aq) + NaHCO3(s) -> CH3COONa(aq) + H2O(l) + CO2(g) |
| Reaction enthalpy | +31.1 kJ per mole, so it takes in heat |
| Gibbs energy | -13.3 kJ, K = 214, so it runs by itself |
| Reacted at equilibrium | 99.5% |
| Temperature change | -6.2 K in 1201 g of 5% vinegar |
| Gas | 24.5 L of CO2 per mole at 298 K |

`output/playback.json` holds the frames the viewer plays.
