"""The cost model: the dissection that sets a direct solve's cost, the fitted CalculiX coefficients against the
measured runs they came from, the verdict and the options when a run does not fit, and the calibration."""
from __future__ import annotations

import numpy as np
import pytest

from wobbl import estimate as E
from wobbl.decks import Deck, Step

# CalculiX 2.23 on one thread of an Intel i7-8750H, gmsh tet10 meshes: nodes, dissection sums, cpu s, peak bytes
MEASURED = {
    "static": [(3830, 5.256e6, 1.245e5, 2.06, 52.4e6), (15367, 1.670e7, 4.276e5, 7.37, 182.8e6),
               (8049, 6.358e6, 2.239e5, 3.93, 101.9e6), (10054, 3.511e7, 4.468e5, 8.99, 163.5e6),
               (27088, 4.075e7, 8.013e5, 15.68, 335.1e6), (21729, 4.242e7, 8.098e5, 14.33, 299.3e6),
               (22313, 1.734e8, 1.257e6, 31.58, 454.7e6)],
    "heat-steady": [(10054, 3.511e7, 4.468e5, 3.50, 42.2e6), (27088, 4.075e7, 8.013e5, 7.10, 86.9e6),
                    (21729, 4.242e7, 8.098e5, 7.59, 76.5e6), (34142, 3.922e8, 2.201e6, 19.29, 173.6e6)],
}


def grid(nx, ny, nz, lx, ly, lz):
    g = np.stack(np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny), np.linspace(0, lz, nz),
                             indexing="ij"), -1)
    return g.reshape(-1, 3)


def test_a_compact_block_costs_more_than_a_plate_or_a_bar_with_as_many_nodes():
    block = grid(32, 20, 16, 100, 60, 50)
    plate = grid(80, 64, 2, 400, 320, 8)
    bar = grid(320, 4, 8, 1600, 20, 40)
    sums = {k: E.dissection(p, len(p)) for k, p in (("block", block), ("plate", plate), ("bar", bar))}
    assert sums["block"][0] > 3 * sums["plate"][0] > 4.5 * sums["bar"][0]
    # a spread out eighth of the nodes gives about the same sums as all of them
    full = E.dissection(block, len(block))
    thin = E.dissection(block[np.random.default_rng(1).choice(len(block), len(block) // 8, replace=False)],
                        len(block))
    assert 0.5 < thin[0] / full[0] < 2.0 and 0.5 < thin[1] / full[1] < 2.0


@pytest.mark.parametrize("procedure", list(MEASURED))
def test_the_fit_reproduces_the_measured_runs(procedure):
    for n, F, M, cpu, rss in MEASURED[procedure]:
        c = E.calculix_cost(n, sums=(F, M), dof=1 if "heat" in procedure else 3, procedure=procedure)
        assert 0.75 < cpu / c.core_seconds < 1.33, (procedure, n, cpu, c.core_seconds)
        assert 0.75 < rss / c.peak_bytes < 1.33, (procedure, n, rss, c.peak_bytes)


def test_threads_help_the_factorization_more_than_the_work_per_node():
    block = E.calculix_cost(22313, sums=(1.734e8, 1.257e6))
    plate = E.calculix_cost(27088, sums=(4.075e7, 8.013e5))
    assert block.parallel > plate.parallel
    assert E.amdahl(2, block.parallel) > 1.6 and E.amdahl(2, plate.parallel) < 1.5


def test_transient_increments_and_deck_cost(tmp_path):
    deck = Deck(tmp_path / "pack.inp", "calculix", "inp", nodes=60000, elements={"C3D10": 40000},
                steps=[Step("heat", period=3600.0, initial=1.0, max_inc=60.0)])
    inc, solves, note = E.step_increments(deck.steps[0])
    assert inc == pytest.approx((3600 * 60) ** 0.5) and solves == pytest.approx(0.85 * inc) and "increments" in note
    c = E.deck_cost(deck)
    assert c.solver == "calculix" and c.size["dof"] == 60000 and c.size["result_sets"] == int(inc)
    steady = E.calculix_cost(60000, 40000, dof=1, procedure="heat")
    assert c.core_seconds == pytest.approx(steady.core_seconds * solves, rel=1e-6)
    assert E.step_increments(Step("static"))[:2] == (1.0, 1.0)


def test_this_machine_reads_cores_memory_and_disk(tmp_path):
    m = E.this_machine(tmp_path / "not" / "made" / "yet")
    assert m.cores >= 1 and m.ram > 1e8 and m.disk_free > 0 and 1 <= m.threads <= m.cores
