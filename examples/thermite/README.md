# Thermite

How much heat does thermite release and how hot does it get?

Wobbl takes the reaction from its name, balances it and finds the heat from NASA Glenn data. It then holds the
energy fixed and finds the temperature the products reach, stopping where a product melts or boils.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Reaction | 2 Al + Fe2O3 -> Al2O3 + 2 Fe |
| Reaction enthalpy | -850.4 kJ per mole as written |
| Gibbs energy | -838.9 kJ, K = 10^147, so it runs to completion |
| Adiabatic temperature | 3139 K, held there by iron boiling |

Cantera's full multiphase equilibrium did not converge for this one, and the report says so and uses the
complete-reaction energy balance instead.

`output/combustion/playback.json` and `output/reaction/playback.json` hold the frames the viewer plays.
