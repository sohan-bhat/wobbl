"""Wobbl: simulate anything before you build it.

Wobbl turns a CAD file and a plain question ("will this motor mount crack under
200 N?") into a verified engineering answer: it meshes the part, applies loads
and materials, runs a real solver (CalculiX, ngspice), checks its own work and
reports the number, a confidence level, the assumptions and a suggested fix.

The package is layered the same way the work flows:

    domain (what you ask)  ->  solver adapter (who computes it)
        ->  verification (can we trust it)  ->  report (what you see)
"""

__version__ = "0.1.1"

__all__ = ["__version__"]
