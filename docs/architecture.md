# Architecture

This page now lives on the website, as part of
[contributing](https://sohan-bhat.github.io/wobbl/get-started/contributing/).

Wobbl is layered the way the work flows. A domain package for strength, heat or circuits turns
loads, supports and materials into a solver neutral model, a solver adapter writes the input
files, runs CalculiX or ngspice and reads the results back, the verification layer turns the
results into a confidence level, and the report layer draws the heatmaps and writes the
summary and the HTML page. A new solver plugs in as one adapter class in `wobbl/solvers/`, and
every CAD import, mesh attempt and geometry edit runs in a child process with a timeout.
