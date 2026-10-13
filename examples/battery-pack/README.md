# Battery pack

When does a pack in a sealed box pass 60 C?

This 2S2P pack of 18650 cells is treated as one 37 x 37 x 65 mm block with effective cell
properties, and it makes 6 W during a one hour discharge inside a sealed plastic box at 35 C.
Still air gives about 5 W/(m2 K), and the wrap radiates with an emissivity of 0.9.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Hottest point after 60 min | 74.5 C, over the 60 C limit for Li-ion cells |
| Time to pass 60 C | about 21 minutes |
| Hand calculation | a lumped model with convection and radiation gives a mean of 72.2 C at 60 min, and the FEA mean is 72.8 C |

![Temperature over the discharge](output/images/history.png)

The pack heats almost uniformly (Biot number 0.012), so the answer is set by how fast the box
sheds heat. Wobbl suggests roughly doubling the heat transfer, with airflow or a thermal path to
a metal wall, or cutting the heat to about 3 W, which keeps the pack near 55 C with some margin
under the limit.
