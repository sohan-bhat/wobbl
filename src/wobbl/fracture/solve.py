"""A lean static solve for cracked parts: displacements and reactions only, and no files left behind."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..fe import StaticModel
from ..solvers.base import SolverError


def solve_displacements(adapter, model: StaticModel, workdir: Path,
                        timeout: float = 1800.0) -> tuple[np.ndarray, np.ndarray]:
    """A static solve that writes only displacements and reaction forces, so the result file stays small, and
    removes its files once they are read."""
    from ..solvers.calculix import _run, read_frd, write_static_inp

    ccx = adapter.require()
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "crack.inp"
    write_static_inp(inp, model)
    text = inp.read_text().replace("*EL FILE\nS\n", "")
    inp.write_text(text)
    log, _ = _run(ccx, workdir, "crack", timeout)
    blocks = read_frd(workdir / "crack.frd", model.mesh.n_nodes)
    for ext in (".frd", ".dat", ".sta", ".cvg", ".12d", ".inp"):
        (workdir / f"crack{ext}").unlink(missing_ok=True)
    if "DISP" not in blocks:
        raise SolverError("CalculiX finished without displacements", log=log[-3000:])
    U = blocks["DISP"][-1][1][:, :3]
    RF = blocks["FORC"][-1][1][:, :3] if "FORC" in blocks else np.zeros_like(U)
    if not np.isfinite(U).all():
        raise SolverError("the cracked part's solve did not converge")
    return U, RF
