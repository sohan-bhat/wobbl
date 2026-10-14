"""Agent facing operations for big problems, behind the MCP server: estimate, start, follow, stop and resume."""
from __future__ import annotations

from pathlib import Path


def _err(e: Exception) -> dict:
    hint = getattr(e, "hint", "")
    out = {"error": str(e), **({"hint": hint} if hint and hint not in str(e) else {})}
    if getattr(e, "estimate", None):
        out["estimate"] = e.estimate
    return out


def estimate_cost(command: list[str] | None = None, path: str = "", solver: str = "", nodes: float = 0,
                  elements: float = 0, end_time: float = 0.0, time_step: float = 0.0, iterations: int = 0,
                  procedure: str = "static", remote: str = "", threads: int = 0,
                  max_hours: float | None = None) -> dict:
    """Run time, peak memory and disk for any Wobbl command, a solver deck or case, or a run given by size, with
    options when it will not fit."""
    from .. import estimate as E

    try:
        machine = None
        if remote:
            from . import remote as R
            R.get(remote)
            machine = R.machines()[remote]
            if threads:
                machine.threads = threads
        elif threads:
            machine = E.this_machine(Path(path).expanduser().resolve().parent if path else ".", threads)
        if command:
            est = E.estimate_command([str(c) for c in command], machine, max_hours)
        elif path:
            est = E.estimate_deck(str(Path(path).expanduser()), machine, max_hours)
        elif solver:
            est = E.estimate_size(solver, int(nodes), int(elements), end_time, time_step, iterations, procedure,
                                  machine=machine, max_hours=max_hours)
        else:
            return {"error": "pass command (any Wobbl command line), path (a solver deck or case), or solver with "
                             "a size"}
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return est.to_dict()


def start_job(path: str = "", command: list[str] | None = None, remote: str = "", threads: int = 0,
              checkpoint: bool = True, cloud: str = "", instance: str = "", region: str = "us-east-1",
              key_name: str = "", security_group: str = "", confirm: str = "",
              max_hours: float | None = None) -> dict:
    """Start a CalculiX deck or a Wobbl command in the background. A cloud run returns its plan and starts
    nothing until called again with confirm set to the plan's token."""
    from . import manage

    try:
        if cloud:
            from .. import estimate as E
            from ..decks import read_inp
            from . import cloud as C
            from .store import new_id

            p = C.plan(E.deck_cost(read_inp(path), checkpoint), instance, region, key_name, security_group,
                       new_id(path), max_hours)
            if confirm != p.token:
                return {"started": False, "plan": p.to_dict(), "prices": C.PRICES, "confirm_token": p.token,
                        "note": "Nothing was started. Show this plan and its cost to the user, and call start_job "
                                "again with confirm set to the token only after they agree."}
            launched = C.launch(p, confirm)
            return manage.start(path, threads=threads, checkpoint=checkpoint, max_hours=max_hours,
                                cloud={**launched, "job": p.job, "cores": p.cores, "ram": p.ram_gb * 1e9})
        return manage.start(path, list(command or []), remote, threads, checkpoint, False, max_hours)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def job_status(job_id: str = "") -> dict:
    from . import manage

    try:
        return manage.status(job_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def job_logs(job_id: str = "", lines: int = 40) -> dict:
    from . import manage

    try:
        return {"job_id": job_id, "log": manage.logs(job_id, lines)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


def cancel_job(job_id: str = "") -> dict:
    from . import manage

    try:
        return manage.cancel(job_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def resume_job(job_id: str = "") -> dict:
    from . import manage

    try:
        return manage.resume(job_id)
    except Exception as e:  # noqa: BLE001
        return _err(e)
