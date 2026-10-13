# How Wobbl checks an answer

This guide now lives on the website, in two pages,
[checks](https://sohan-bhat.github.io/wobbl/guides/results/checks/) and
[confidence](https://sohan-bhat.github.io/wobbl/guides/results/confidence/).

Every strength answer runs mesh convergence, force and moment balance, a beam theory hand
calculation, singularity detection, bolt bearing, units and sanity, a large deflection re-solve
when the part bends a lot, and a load introduction check for loads that act off the part. Heat
answers check energy balance and a lumped hand calculation, and circuit answers are compared
with the textbook and run again at half the time step. The checks set a confidence of high,
medium or low, always with the reasons, and a result that failed a check is never high.
