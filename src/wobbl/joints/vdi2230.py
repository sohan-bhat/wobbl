"""The hand calculation for a bolted joint, after VDI 2230 Part 1 (2015).

Under preload the bolt and the parts it clamps are two springs in series: the bolt stretches by F_V delta_S and the
parts squeeze by F_V delta_P. An external load F_A pulling the joint apart then shares itself by stiffness. The bolt
takes Phi F_A more (Phi = n delta_P / (delta_S + delta_P)) and the clamp force at the interface falls by (1 - Phi) F_A,
so the joint opens at F_V / (1 - Phi). The bolt's compliance adds its head, the loaded thread and the nut; the clamped
parts carry the load through a cone that spreads from under the head (or a cone and a sleeve when the parts are
narrower than the cone). Sizes are ISO metric coarse threads, ISO 4014 hexagon heads, ISO 273 medium clearance holes
and ISO 4032 nuts, with strengths from ISO 898-1 and ISO 3506-1.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

# ISO 261 coarse pitch, ISO 4014 bearing diameter dw (min), ISO 273 medium clearance hole, ISO 4032 nut height (max)
PITCH = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0, 20: 2.5, 24: 3.0, 30: 3.5,
         36: 4.0}
HEAD_DW = {3: 4.57, 4: 5.88, 5: 6.88, 6: 8.88, 8: 11.63, 10: 14.63, 12: 16.63, 14: 19.64, 16: 22.49, 20: 28.19,
           24: 33.61, 30: 42.75, 36: 51.11}
HOLE = {3: 3.4, 4: 4.5, 5: 5.5, 6: 6.6, 8: 9.0, 10: 11.0, 12: 13.5, 14: 15.5, 16: 17.5, 20: 22.0, 24: 26.0, 30: 33.0,
        36: 39.0}
NUT_M = {3: 2.4, 4: 3.2, 5: 4.7, 6: 5.2, 8: 6.8, 10: 8.4, 12: 10.8, 14: 12.8, 16: 14.8, 20: 18.0, 24: 21.5, 30: 25.6,
         36: 31.0}
# property class: (yield Rp0.2 or ReL, proof stress Sp, tensile Rm), MPa; 8.8 above M16 is a little stronger
GRADES = {"4.6": (240, 225, 400), "4.8": (340, 310, 420), "5.6": (300, 280, 500), "5.8": (420, 380, 520),
          "6.8": (480, 440, 600), "8.8": (640, 580, 800), "9.8": (720, 650, 900), "10.9": (940, 830, 1040),
          "12.9": (1100, 970, 1220), "a2-70": (450, 450, 700), "a4-70": (450, 450, 700), "a2-80": (600, 600, 800),
          "a4-80": (600, 600, 800)}
GRADE_SOURCE = {"steel": "ISO 898-1", "stainless": "ISO 3506-1 (no proof stress is defined, so its 0.2% proof stress "
                "stands in)"}


@dataclass
class Bolt:
    d: float                 # nominal diameter, mm
    pitch: float
    grade: str
    dw: float                # bearing diameter under the head and nut, mm
    hole: float              # clearance hole, mm
    nut_height: float
    E: float = 205000.0      # MPa (VDI takes 205 GPa for steel bolts)
    yield_strength: float = 640.0
    proof: float = 580.0
    tensile: float = 800.0

    @property
    def d2(self) -> float:
        return self.d - 0.649519 * self.pitch

    @property
    def d3(self) -> float:
        return self.d - 1.226869 * self.pitch

    @property
    def stress_area(self) -> float:
        return math.pi / 4 * (0.5 * (self.d2 + self.d3)) ** 2

    @property
    def name(self) -> str:
        return f"M{self.d:g} {self.grade.upper()}"


def bolt_from(text: str, dw: float | None = None) -> Bolt:
    """'M8 8.8', 'M10x1.25 10.9', 'M6 a2-70'. The class defaults to 8.8."""
    m = re.match(r"^\s*m(\d+(?:\.\d+)?)(?:\s*x\s*(\d*\.?\d+))?(?:\s+(\d+\.\d+|a[24]-[78]0))?\s*$", text.strip(), re.I)
    if not m:
        raise ValueError(f"could not read the bolt '{text}'; write it like 'M8 8.8' or 'M10 10.9'")
    d = float(m.group(1))
    key = int(round(d))
    if key not in PITCH:
        raise ValueError(f"M{d:g} is not a standard metric size; use one of {', '.join(f'M{k}' for k in PITCH)}")
    grade = (m.group(3) or "8.8").lower()
    if grade not in GRADES:
        raise ValueError(f"unknown property class {grade}; use one of {', '.join(GRADES)}")
    ry, sp, rm = GRADES[grade]
    if grade == "8.8" and d > 16:
        ry, sp, rm = 660, 600, 830
    E = 193000.0 if grade.startswith("a") else 205000.0
    pitch = float(m.group(2)) if m.group(2) else PITCH[key]
    return Bolt(d, pitch, grade, dw or HEAD_DW[key], HOLE[key], NUT_M[key], E, ry, sp, rm)


def bolt_compliance(b: Bolt, clamp: float) -> float:
    """delta_S, mm/N: head (0.5 d of the nominal section), the loaded thread through the clamp length, the engaged
    thread (0.5 d at the core section) and the nut (0.4 d). A bolt threaded the whole way is taken, which is the
    softer and so the safer choice for separation."""
    an, a3 = math.pi / 4 * b.d ** 2, math.pi / 4 * b.d3 ** 2
    return (0.5 * b.d / an + clamp / a3 + 0.5 * b.d / a3 + 0.4 * b.d / an) / b.E


def cone_tan(clamp: float, dw: float, outer: float) -> float:
    """tan phi of the pressure cone in a through bolted joint (VDI 2230, Eq. 5.1/23)."""
    beta = clamp / dw
    y = max(outer, dw) / dw
    return 0.362 + 0.032 * math.log(beta / 2) + 0.153 * math.log(y)


def plate_compliance(dw: float, hole: float, clamp: float, outer: float, E: float) -> tuple[float, float, float]:
    """delta_P, mm/N, of the clamped parts of a through bolted joint, with tan phi and the limit diameter. Two cones
    from under the head and the nut, or cones that reach the outside diameter and a sleeve between them."""
    t = cone_tan(clamp, dw, outer)
    limit = dw + clamp * t
    if outer <= dw:      # narrower than the head: a plain sleeve
        return 4 * clamp / (E * math.pi * (outer ** 2 - hole ** 2)), t, limit
    if outer >= limit:
        x = (dw + hole) * (dw + clamp * t - hole) / ((dw - hole) * (dw + clamp * t + hole))
        return 2 * math.log(x) / (E * math.pi * hole * t), t, limit
    cones = 2 * math.log((dw + hole) * (outer - hole) / ((dw - hole) * (outer + hole))) / (E * math.pi * hole * t)
    sleeve = 4 / (outer ** 2 - hole ** 2) * (clamp - (outer - dw) / t) / (E * math.pi)
    return cones + sleeve, t, limit


def torque_for(b: Bolt, preload: float, mu_thread: float = 0.12, mu_head: float = 0.12) -> float:
    """VDI 2230 tightening torque, N mm: F (0.16 P + 0.58 d2 mu_G + D_Km / 2 mu_K)."""
    dkm = 0.5 * (b.dw + b.hole)
    return preload * (0.16 * b.pitch + 0.58 * b.d2 * mu_thread + 0.5 * dkm * mu_head)


@dataclass
class JointCheck:
    """The hand calculation of one bolt."""
    bolt: str
    clamp_length: float
    outer_diameter: float
    delta_S: float
    delta_P: float
    tan_phi: float
    phi: float
    preload: float
    axial: float             # external load along the bolt, N
    shear: float             # external load across the bolt, N
    bolt_force: float        # F_V + Phi F_A
    clamp_force: float       # F_V - (1 - Phi) F_A, at the interface
    separation_load: float   # F_V / (1 - Phi)
    slip_capacity: float     # mu q F_KR
    slip_safety: float
    bolt_stress: float       # bolt force over the stress area, MPa
    proof_use: float
    torsion_use: float       # tightening stress with the thread torque, over the yield stress
    head_pressure: float     # MPa
    strip_safety: float      # nut thread shear capacity over the bolt's breaking load

    def to_dict(self) -> dict:
        return asdict(self)


def check(b: Bolt, clamp: float, outer: float, E_parts: float, preload: float, axial: float = 0.0,
          shear: float = 0.0, mu: float = 0.15, interfaces: int = 1, n: float = 1.0,
          mu_thread: float = 0.12, engagement: float | None = None,
          thread_strength: float | None = None) -> JointCheck:
    """VDI 2230 for one bolt: load factor, bolt force, residual clamp force, separation and slip, the bolt stress
    against its proof stress, the stress while tightening, the pressure under the head and thread stripping."""
    ds = bolt_compliance(b, clamp)
    dp, t, _ = plate_compliance(b.dw, b.hole, clamp, outer, E_parts)
    phi = n * dp / (ds + dp)
    fa = max(axial, 0.0)
    fs = preload + phi * fa
    fkr = preload - (1 - phi) * fa
    sep = preload / (1 - phi)
    cap = mu * interfaces * max(fkr, 0.0)
    slip = cap / shear if shear > 0 else float("inf")
    stress = fs / b.stress_area
    # tightening: tension plus the torsion of the thread torque, judged at the yield stress (VDI 2230, Eq. R8)
    d0 = 0.5 * (b.d2 + b.d3)
    tau = preload * (0.16 * b.pitch + 0.58 * b.d2 * mu_thread) / (math.pi * d0 ** 3 / 16)
    sig = preload / b.stress_area
    red = math.sqrt(sig ** 2 + 3 * (0.5 * tau) ** 2)
    head = fs / (math.pi / 4 * (b.dw ** 2 - b.hole ** 2))
    # the internal thread shears over about 0.75 pi d m at 0.6 times its strength, against the bolt's breaking
    # load: a nut of the matching class is made so the bolt breaks first, a tapped hole in a softer part may not be
    m = engagement or b.nut_height
    strip = 0.6 * (thread_strength or b.tensile) * 0.75 * math.pi * b.d * m / (b.stress_area * b.tensile)
    return JointCheck(b.name, clamp, outer, ds, dp, t, phi, preload, fa, shear, fs, fkr, sep, cap, slip, stress,
                      stress / b.proof, red / b.yield_strength, head, strip)
