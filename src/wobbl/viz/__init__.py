"""wobbl.viz: every run's interactive 3D simulation, written through one API and played by one small viewer.

A domain builds a Scene from its own results and saves it next to the outcome (scene.py has the API and the file
format). attach() then puts the viewer into report.html so the run opens as a single offline page, and runs
without a scene of their own get one from their own files through the adapters. docs/viz.md is the guide.
"""
from __future__ import annotations

from .embed import attach, open_view, view_path
from .scene import BUDGET, FILE, Scene, load


def simulation_view(path):
    """The page for a finished `wobbl simulate` run: the question, its setting and every step played in 3D.
    Takes the path of simulation.json and returns the page's path, or None when no step has anything to show."""
    from .simulation import simulation_view as build
    return build(path)


__all__ = ["BUDGET", "FILE", "Scene", "attach", "load", "open_view", "simulation_view", "view_path"]
