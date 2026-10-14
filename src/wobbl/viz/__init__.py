"""wobbl.viz: every run's interactive 3D simulation, written through one API and played by one small viewer.

A domain builds a Scene from its own results and saves it next to the outcome (scene.py has the API and the file
format).
"""
from __future__ import annotations

from .scene import BUDGET, FILE, Scene, load

__all__ = ["BUDGET", "FILE", "Scene", "load"]
