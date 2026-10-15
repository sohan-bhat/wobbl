"""What every general analysis shares: the model as set up for the question, and the finding each one hands back
to be combined into one answer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..units import fmt
from .model import Info, Loaded
from .motors import Motor
from .spec import Mechanism

RANK = {"ok": 0, "info": 0, "marginal": 1, "fails": 2}


@dataclass
class Setup:
    mech: Mechanism
    ld: Loaded
    model: object                         # compiled MjModel, payload and floor included
    info: Info
    motors: dict[str, Motor]              # what drives each joint, from the file, the spec or the question
    question: str = ""
    payload: tuple | None = None          # (kg, body name, point in its frame)
    passive: bool = False                 # the question says nothing drives it
    quick: bool = False                   # fewer poses and a shorter settle, for a quick look
    img: Path = Path(".")
    say: object = None
    cache: dict = field(default_factory=dict)

    def log(self, stage: str, msg: str) -> None:
        if self.say:
            self.say(stage, msg)

    def n(self, full: int) -> int:
        """How many poses or samples to use: ``full``, or a third of it for a quick look."""
        return max(full // 3, 40) if self.quick else full

    @property
    def driven(self) -> list[dict]:
        """The joints something drives, or every joint when the model names no drives at all."""
        js = [j for j in self.info.joints if j["name"] in self.motors]
        return js or list(self.info.joints)


@dataclass
class Finding:
    name: str                             # torque, range, interference, tipping, contact, motion
    verdict: str                          # ok, info, marginal, fails
    sentence: str
    metrics: list[tuple[str, str]] = field(default_factory=list)
    checks: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    fixes: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    figures: list[list[str]] = field(default_factory=list)
    images: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    extra: list[tuple[str, str]] = field(default_factory=list)
    run: object = None
    kin: object = None                    # the slow ideal run and the mechanism it ran, for checks that reuse it


def motor_from_limits(joint: str, effort: float, speed: float, source: str) -> Motor:
    """A drive as a model file states it: a torque limit and a speed limit, full torque up to that speed."""
    return Motor("file", f"the limits {source} gives {joint}", effort, speed if speed > 0 else math.inf, curve="box")


def amount(j: dict, x: float, digits: int = 3) -> str:
    """A torque on a hinge or a force on a slide, with its unit."""
    return f"{fmt(x, digits)} N m" if j["kind"] == "hinge" else f"{fmt(x, digits)} N"


def worst_verdict(findings: list[Finding]) -> str:
    v = max((RANK[f.verdict] for f in findings), default=0)
    return {0: "ok", 1: "marginal", 2: "fails"}[v]
