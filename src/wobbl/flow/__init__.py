"""Air and water flow: pipes and ducts, drag on bodies, fan cooled heat sinks, propellers and hulls.

CFD runs on OpenFOAM, which Wobbl installs on demand and keeps lean (see install.py). Propellers use blade element
momentum theory and hulls the Holtrop and Mennen method, which need no solver. Every answer is checked against a
correlation or a hand calculation, a grid convergence index, residuals and mass conservation.
"""
