"""Materials library with real, sourced properties.

Values are typical room temperature properties. Printed plastics use values for
parts printed at 100 percent infill, loaded in the plane of the layers; the
``layer_factor`` is the fraction of that strength left when the load pulls the
layers apart. Ductile materials are judged by von Mises stress against yield,
brittle ones (printed PLA, wood, FR4) by the largest principal stress against
tensile strength.

Units in this table are the ones people read on datasheets (MPa, kg/m3,
W/(m K), J/(kg K), 1/K, degC). ``Material.internal()`` converts them to Wobbl's
solver units.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    category: str                 # metal, printed plastic, molded plastic, wood, composite, electronics
    E: float                      # Young's modulus, MPa
    nu: float                     # Poisson's ratio
    density: float                # kg/m3
    yield_strength: float         # MPa (for brittle materials, equal to tensile strength)
    tensile_strength: float       # MPa
    elongation: float             # percent at break
    conductivity: float           # W/(m K)
    specific_heat: float          # J/(kg K)
    cte: float                    # 1/K
    max_service_temp: float       # degC, where stiffness or strength falls off sharply
    ductile: bool = True
    layer_factor: float = 1.0     # strength fraction across print layers
    recommended_sf: float = 1.5   # sensible design safety factor for static loads
    aliases: tuple[str, ...] = ()
    source: str = ""
    notes: str = ""
    process: str = ""             # how parts are made: printed, molded, cast, wrought, natural, laminated

    @property
    def criterion(self) -> str:
        return "von Mises vs yield" if self.ductile else "max principal vs tensile strength"

    @property
    def allowable(self) -> float:
        """Stress at which this material fails by Wobbl's criterion, MPa."""
        return self.yield_strength if self.ductile else self.tensile_strength

    def internal(self) -> dict[str, float]:
        """Properties in solver units (mm, N, s, t, mW, degC)."""
        return {
            "E": self.E,
            "nu": self.nu,
            "density": self.density * 1e-12,          # t/mm3
            "conductivity": self.conductivity,         # mW/(mm K) == W/(m K)
            "specific_heat": self.specific_heat * 1e6,  # mJ/(t K)
            "cte": self.cte,
        }

    def with_overrides(self, **kw) -> "Material":
        return replace(self, **kw)

    def summary(self) -> str:
        return (f"{self.name}: E {self.E:g} MPa, nu {self.nu:g}, {self.criterion} "
                f"{self.allowable:g} MPa, density {self.density:g} kg/m3")


_M = Material

LIBRARY: dict[str, Material] = {m.key: m for m in [
    # printed plastics (100 percent infill, in plane of layers)
    _M("pla", "PLA (3D printed)", "printed plastic", 3500, 0.36, 1240, 50, 50, 4, 0.13, 1800, 68e-6, 50,
       ductile=False, layer_factor=0.6, recommended_sf=2.5, aliases=("pla+", "polylactic acid"),
       source="Prusament PLA technical data sheet; Ultimaker PLA TDS",
       notes="Brittle in bending, creeps under sustained load, softens near 55 C.", process="printed"),
    _M("petg", "PETG (3D printed)", "printed plastic", 2100, 0.38, 1270, 47, 50, 6, 0.20, 1200, 60e-6, 70,
       layer_factor=0.7, recommended_sf=2.5, aliases=("pet-g", "pet"),
       source="Prusament PETG technical data sheet", process="printed"),
    _M("abs", "ABS (3D printed)", "printed plastic", 2200, 0.35, 1040, 33, 37, 7, 0.17, 1400, 90e-6, 85,
       layer_factor=0.6, recommended_sf=2.5, source="Ultimaker ABS TDS; Prusament ASA/ABS data", process="printed"),
    _M("asa", "ASA (3D printed)", "printed plastic", 2000, 0.35, 1070, 35, 40, 8, 0.17, 1300, 95e-6, 90,
       layer_factor=0.6, recommended_sf=2.5, source="Prusament ASA technical data sheet", process="printed"),
    _M("nylon", "Nylon PA6 (conditioned)", "printed plastic", 1800, 0.39, 1140, 50, 60, 30, 0.25, 1700, 85e-6, 100,
       layer_factor=0.7, recommended_sf=2.5, aliases=("pa6", "pa", "polyamide", "nylon 6"),
       source="BASF Ultramid B3S conditioned; Polymaker PA6 TDS",
       notes="Absorbs moisture; dry values are stiffer and stronger.", process="printed"),
    _M("pa12", "Nylon PA12 (SLS or MJF)", "printed plastic", 1700, 0.39, 1010, 48, 48, 20, 0.22, 1700, 100e-6, 120,
       layer_factor=0.9, recommended_sf=2.0, aliases=("nylon 12", "mjf", "sls nylon"),
       source="HP 3D High Reusability PA 12 datasheet; EOS PA 2200", process="printed"),
    _M("pc", "Polycarbonate", "molded plastic", 2350, 0.37, 1200, 62, 65, 100, 0.20, 1200, 68e-6, 120,
       recommended_sf=2.0, aliases=("polycarbonate", "lexan", "makrolon"),
       source="Covestro Makrolon 2405 datasheet", process="molded"),
    _M("hdpe", "HDPE", "molded plastic", 1000, 0.42, 955, 26, 30, 500, 0.45, 1900, 120e-6, 80,
       recommended_sf=2.5, aliases=("polyethylene",), source="MatWeb HDPE injection molded average", process="molded"),
    # metals
    _M("6061-t6", "Aluminum 6061-T6", "metal", 68900, 0.33, 2700, 276, 310, 12, 167, 896, 23.6e-6, 150,
       aliases=("6061", "al6061", "aluminum 6061", "aluminium 6061", "aluminum", "aluminium", "al"),
       source="ASM Aerospace Specification Metals, Aluminum 6061-T6", process="wrought"),
    _M("7075-t6", "Aluminum 7075-T6", "metal", 71700, 0.33, 2810, 503, 572, 11, 130, 960, 23.4e-6, 120,
       aliases=("7075", "al7075", "aluminum 7075", "aluminium 7075"),
       source="ASM Aerospace Specification Metals, Aluminum 7075-T6", process="wrought"),
    _M("6063-t5", "Aluminum 6063-T5", "metal", 68900, 0.33, 2700, 145, 186, 12, 209, 900, 23.4e-6, 150,
       aliases=("6063", "extrusion"), source="ASM Aerospace Specification Metals, Aluminum 6063-T5",
       notes="Common for extrusions and heat sinks.", process="wrought"),
    _M("5052-h32", "Aluminum 5052-H32", "metal", 70300, 0.33, 2680, 193, 228, 12, 138, 880, 23.8e-6, 150,
       aliases=("5052", "sheet aluminum"), source="ASM Aerospace Specification Metals, Aluminum 5052-H32",
       process="wrought"),
    _M("a356-t6", "Aluminum A356-T6 (sand cast)", "metal", 72400, 0.33, 2670, 165, 228, 3.5, 151, 963, 21.4e-6,
       150, recommended_sf=2.0, aliases=("a356", "a356.0", "cast aluminum", "cast aluminium"),
       source="MatWeb Aluminum A356.0-T6, sand cast; ASM Handbook Vol. 2", process="cast",
       notes="Castings vary with porosity, so they carry a larger safety factor."),
    _M("mild-steel", "Mild steel (ASTM A36)", "metal", 200000, 0.29, 7850, 250, 400, 20, 51.9, 486, 11.7e-6, 400,
       aliases=("steel", "a36", "1018", "low carbon steel", "carbon steel", "mild steel"),
       source="ASTM A36 minimum yield; MatWeb A36 typical properties", process="wrought"),
    _M("4140", "Steel 4140 (quenched and tempered)", "metal", 205000, 0.29, 7850, 655, 1020, 18, 42.6, 473,
       12.3e-6, 400, aliases=("4140 steel", "chromoly", "4130"), source="MatWeb AISI 4140 Q&T 650 C",
       process="wrought"),
    _M("ss304", "Stainless steel 304 (annealed)", "metal", 193000, 0.29, 8000, 215, 505, 70, 16.2, 500, 17.3e-6,
       500, aliases=("304", "stainless", "stainless steel", "304 stainless", "ss"),
       source="ASM Aerospace Specification Metals, AISI 304", process="wrought"),
    _M("ti-6al-4v", "Titanium Ti-6Al-4V", "metal", 113800, 0.34, 4430, 880, 950, 14, 6.7, 526, 8.6e-6, 400,
       aliases=("titanium", "ti64", "grade 5 titanium"), source="ASM Aerospace Specification Metals, Ti-6Al-4V",
       process="wrought"),
    _M("copper", "Copper C110", "metal", 115000, 0.31, 8890, 69, 220, 50, 388, 385, 17.0e-6, 200,
       aliases=("cu", "c110", "c101"), source="Copper Development Association C11000 annealed", process="wrought"),
    _M("brass", "Brass C360", "metal", 97000, 0.31, 8500, 310, 385, 25, 115, 380, 20.5e-6, 200,
       aliases=("c360",), source="Copper Development Association C36000 half hard", process="wrought"),
    # wood and composites
    _M("wood", "Douglas fir (along grain)", "wood", 12400, 0.30, 530, 50, 50, 1, 0.12, 1700, 4e-6, 80,
       ductile=False, recommended_sf=3.0, aliases=("douglas fir", "fir", "pine", "softwood", "lumber"),
       source="USDA Wood Handbook FPL-GTR-282, clear wood at 12 percent moisture (strength reduced from MOR 85 MPa)",
       notes="Wood is orthotropic; Wobbl models it as isotropic along the grain. Knots and grain run out reduce "
             "strength.",
       process="natural"),
    _M("plywood", "Birch plywood", "wood", 8000, 0.30, 680, 35, 35, 1, 0.15, 1700, 6e-6, 80,
       ductile=False, recommended_sf=3.0, aliases=("baltic birch",), source="Finnish birch plywood handbook, typical",
       process="natural"),
    _M("fr4", "FR4 (PCB laminate, in plane)", "electronics", 22000, 0.15, 1850, 300, 300, 2, 0.3, 1100, 14e-6, 130,
       ductile=False, recommended_sf=2.0, aliases=("pcb", "g10"), source="Isola 370HR datasheet", process="laminated"),
    _M("li-ion-cell", "Li-ion cell (lumped, effective)", "electronics", 5000, 0.3, 2500, 10, 10, 1, 3.0, 950,
       10e-6, 60, ductile=False, aliases=("battery", "18650", "21700", "cell"),
       source="Effective radial properties, Drake et al. J. Power Sources 252 (2014)",
       notes="For thermal studies. The 60 C service limit is the usual safe operating ceiling for Li-ion."),
]}


def all_materials() -> list[Material]:
    return list(LIBRARY.values())


def _index() -> dict[str, str]:
    idx = {}
    for m in LIBRARY.values():
        for name in (m.key, m.name.lower(), *m.aliases):
            idx[_clean(name)] = m.key
    return idx


def _clean(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def get_material(name: str) -> Material:
    """Find a material by key, name or alias. Raises KeyError with suggestions."""
    idx = _index()
    k = _clean(name)
    if k in idx:
        return LIBRARY[idx[k]]
    # tolerate extra words: "printed PETG", "7075-T6 aluminum" (a grade number beats a generic word like aluminum)
    for alias, key in sorted(idx.items(), key=lambda kv: (not any(c.isdigit() for c in kv[0]), -len(kv[0]))):
        if len(alias) >= 3 and alias in k:
            return LIBRARY[key]
    close = difflib.get_close_matches(k, list(idx), n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(LIBRARY[idx[c]].key for c in close)}?" if close else ""
    raise KeyError(f"unknown material '{name}'.{hint} Run 'wobbl materials' for the list, or pass "
                   f"custom properties like --material 'custom:E=3000,nu=0.35,yield=40'.")


PROCESSES = {"printed": "printed", "3d printed": "printed", "molded": "molded", "moulded": "molded",
             "injection molded": "molded", "cast": "cast", "casting": "cast", "wrought": "wrought",
             "machined": "wrought", "rolled": "wrought", "extruded": "wrought", "forged": "wrought",
             "natural": "natural", "wood": "natural", "laminated": "laminated", "laminate": "laminated"}


def parse_custom(spec: str, base: Material | None = None) -> Material:
    """custom:E=72400,nu=0.33,yield=165,density=2670,process=cast (MPa, kg/m3).

    ``process`` says how the part is made (printed, molded, cast, wrought, natural or laminated), which sets the
    caveats a study lists and, for printed parts, the layer strength. Unspecified values come from ``base``, or
    from the library material closest in stiffness among those made the same way (ordinary stock when no process
    is given), so a custom aluminum gets an aluminum's density and a custom plastic a plastic's. Custom materials
    are judged as ductile unless ductile=0.
    """
    body = spec.split(":", 1)[1] if ":" in spec else spec
    vals: dict = {}
    process = None
    keymap = {"e": "E", "nu": "nu", "yield": "yield_strength", "sy": "yield_strength", "uts": "tensile_strength",
              "density": "density", "rho": "density", "k": "conductivity", "cp": "specific_heat",
              "cte": "cte", "tmax": "max_service_temp"}
    for part in body.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        k = k.strip().lower()
        if k == "ductile":
            vals["ductile"] = v.strip().lower() in ("1", "true", "yes")
            continue
        if k in ("process", "class", "made"):
            process = PROCESSES.get(v.strip().lower())
            if process is None:
                raise KeyError(f"unknown process '{v.strip()}' (use printed, molded, cast, wrought, natural or "
                               "laminated)")
            continue
        if k not in keymap:
            raise KeyError(f"unknown custom material property '{k}' (use {', '.join(keymap)}, process, ductile)")
        vals[keymap[k]] = float(v)
    if "yield_strength" in vals and "tensile_strength" not in vals:
        vals["tensile_strength"] = vals["yield_strength"]
    if base is not None:
        return replace(base, key="custom", name="Custom material", aliases=(), source="user supplied",
                       **({"process": process} if process else {}), **vals)
    E = vals.get("E", LIBRARY["pla"].E)
    # with no process given, nothing special is assumed: the base is ordinary stock, not a print or a casting
    made = (process,) if process else ("wrought", "molded", "natural", "laminated")
    pool = [m for m in LIBRARY.values() if m.process in made] or [m for m in LIBRARY.values() if m.process]
    base = min(pool, key=lambda m: abs(math.log(m.E / E)))
    vals.setdefault("ductile", True)
    if process != "printed":
        vals.setdefault("layer_factor", 1.0)
    return replace(base, key="custom", name="Custom material" + (f" ({process})" if process else ""), aliases=(),
                   source=f"user supplied; the rest from {base.name}", process=process or "", notes="", **vals)


def resolve_material(spec: str) -> Material:
    if spec.lower().startswith("custom"):
        return parse_custom(spec)
    if ":" in spec and "=" in spec:  # pla:E=3000 overrides on a library material
        name, body = spec.split(":", 1)
        return parse_custom("custom:" + body, get_material(name)).with_overrides(
            key=get_material(name).key + "*", name=get_material(name).name + " (modified)")
    return get_material(spec)


SWAP_ORDER = ["pla", "petg", "abs", "asa", "pc", "nylon", "pa12", "6063-t5", "5052-h32", "6061-t6", "mild-steel",
              "ss304", "7075-t6", "4140", "ti-6al-4v"]
PRINTABLE = {"pla", "petg", "abs", "asa", "nylon", "pa12", "pc"}


@dataclass(frozen=True)
class FatigueData:
    """High cycle fatigue strength of smooth, polished specimens in fully reversed loading (R = -1).

    ``strength`` is the fatigue strength at ``cycles``. Steels and titanium flatten into an endurance limit there
    (``knee``); aluminum, plastics and wood keep weakening, so beyond ``cycles`` their curve is an extrapolation.
    ``estimated`` marks a rule of thumb rather than a test of this material.
    """
    strength: float               # MPa
    cycles: float
    knee: bool
    family: str                   # steel, aluminum, titanium, plastic, wood: sets the notch sensitivity
    source: str
    estimated: bool = False


_SHIGLEY = "estimate: S'e = 0.5 Sut for steels up to 1400 MPa (Shigley's Mechanical Engineering Design, 10th ed., Eq. 6-8)"
_RR_MOORE = "R.R. Moore rotating beam, completely reversed, 5e8 cycles"
_PLASTIC = ("estimate: unfilled thermoplastics reach about 20 to 30% of their tensile strength at 1e7 cycles "
            "(for example Crawford, Plastics Engineering); Wobbl takes 25%")

FATIGUE: dict[str, FatigueData] = {
    "6061-t6": FatigueData(96.5, 5e8, False, "aluminum", f"ASM Aerospace Specification Metals, 6061-T6, {_RR_MOORE}"),
    "7075-t6": FatigueData(159.0, 5e8, False, "aluminum", f"ASM Aerospace Specification Metals, 7075-T6, {_RR_MOORE}"),
    "6063-t5": FatigueData(68.9, 5e8, False, "aluminum", f"ASM Aerospace Specification Metals, 6063-T5, {_RR_MOORE}"),
    "5052-h32": FatigueData(117.0, 5e8, False, "aluminum",
                            f"ASM Aerospace Specification Metals, 5052-H32, {_RR_MOORE}"),
    "a356-t6": FatigueData(59.0, 5e8, False, "aluminum", f"MatWeb Aluminum A356.0-T6, sand cast, {_RR_MOORE}"),
    "mild-steel": FatigueData(200.0, 1e6, True, "steel", _SHIGLEY, estimated=True),
    "4140": FatigueData(510.0, 1e6, True, "steel", _SHIGLEY, estimated=True),
    "ss304": FatigueData(252.0, 1e6, True, "steel", _SHIGLEY + ", applied to an austenitic stainless steel",
                         estimated=True),
    "ti-6al-4v": FatigueData(510.0, 1e7, True, "titanium",
                             "ASM Aerospace Specification Metals, Ti-6Al-4V, unnotched (Kt = 1), 1e7 cycles"),
    "pla": FatigueData(12.5, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "petg": FatigueData(12.5, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "abs": FatigueData(9.25, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "asa": FatigueData(10.0, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "nylon": FatigueData(15.0, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "pa12": FatigueData(12.0, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "pc": FatigueData(16.25, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "hdpe": FatigueData(7.5, 1e7, False, "plastic", _PLASTIC, estimated=True),
    "wood": FatigueData(23.8, 3e7, False, "wood",
                        "USDA Wood Handbook FPL-GTR-282: clear wood in fully reversed bending keeps about 28% of its "
                        "85 MPa modulus of rupture at 3e7 cycles"),
}


def fatigue_data(mat: Material) -> FatigueData | None:
    """The S-N anchor of a material: its own entry (scaled with the tensile strength when a library material is
    modified), or for a custom metal a rule of thumb picked by stiffness, steel near 200 GPa and aluminum near
    70 GPa. None when nothing applies."""
    key = mat.key.rstrip("*")
    if key in FATIGUE and mat.key != "custom":
        base = FATIGUE[key]
        scale = mat.tensile_strength / LIBRARY[key].tensile_strength
        return base if abs(scale - 1) < 1e-9 else replace(
            base, strength=base.strength * scale, source=base.source + ", scaled with the modified tensile strength",
            estimated=True)
    if mat.category == "metal" and 150000 <= mat.E <= 230000:
        return FatigueData(min(0.5 * mat.tensile_strength, 700.0), 1e6, True, "steel", _SHIGLEY, estimated=True)
    if mat.category == "metal" and 60000 <= mat.E <= 80000:
        return FatigueData(0.3 * mat.tensile_strength, 5e8, False, "aluminum",
                           "estimate: 0.3 Sut at 5e8 cycles, close to the wrought aluminum alloys in this table",
                           estimated=True)
    return None


# --- past yield ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hardening:
    """True stress against plastic strain past yield, for elastic plastic studies. Beyond the last point the stress
    stays flat, which is where the metal would start to neck."""
    points: tuple[tuple[float, float], ...]   # (true stress MPa, true plastic strain), the first at plastic strain 0
    kind: str                                 # multilinear or bilinear
    source: str


YUN_GARDNER = ("Yun and Gardner (2017), Stress-strain curves for hot-rolled steels, J. Constr. Steel Res. 133, "
               "quad-linear model through the yield and tensile strength")
ARRAYAGO = ("Arrayago, Real and Gardner (2015), Description of stress-strain curves for stainless steel alloys, "
            "Materials and Design 87, two stage Ramberg-Osgood with n = 7 for austenitic grades")
STEINHARDT = ("Ramberg-Osgood knee up to the 0.2% proof stress with Steinhardt's exponent n = f0.2 / 10, as given "
              "in Mazzolani (1995), Aluminium Alloy Structures, 2nd ed., then straight to the tensile strength at "
              "half the elongation at break")
HARDENING_MODELS = {"mild-steel": "hot rolled", "ss304": "austenitic", "6061-t6": "aluminum", "7075-t6": "aluminum",
                    "6063-t5": "aluminum", "5052-h32": "aluminum", "a356-t6": "aluminum"}


def hardening_curve(m: Material) -> Hardening:
    """The true stress strain curve past yield: a published model for the metals that have one, else a bilinear
    curve from the datasheet yield and tensile strength."""
    model = HARDENING_MODELS.get(m.key.rstrip("*"))
    fy, fu, E = m.yield_strength, max(m.tensile_strength, m.yield_strength), m.E
    if model == "hot rolled" and fu > fy:
        r = fy / fu
        e_sh = min(max(0.1 * r - 0.055, 0.015), 0.03)
        e_u = max(0.6 * (1.0 - r), 0.06)
        c1 = (e_sh + 0.25 * (e_u - e_sh)) / e_u
        f_c1 = fy + (fu - fy) / (0.4 * (e_u - e_sh)) * (c1 * e_u - e_sh)
        eng = [(fy / E, fy), (e_sh, fy), (c1 * e_u, f_c1), (e_u, fu)]
        return Hardening(_true_plastic(eng, E), "multilinear", YUN_GARDNER)
    if model == "austenitic" and fu > fy:
        n, e_u, m2 = 7.0, 1.0 - fy / fu, 1.0 + 2.8 * fy / fu
        e_y = E / (1.0 + 0.002 * n * E / fy)
        eng = [(s / E + 0.002 * (s / fy) ** n, s) for s in (0.6 * fy, 0.7 * fy, 0.8 * fy, 0.9 * fy, fy)]
        eng += [(0.002 + fy / E + (s - fy) / e_y + e_u * ((s - fy) / (fu - fy)) ** m2, s)
                for s in (fy + (fu - fy) * k for k in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0))]
        return Hardening(_true_plastic(eng, E, start=1e-4), "multilinear", ARRAYAGO)
    if model == "aluminum" and fu > fy:
        # the exponent describes the knee up to the proof stress; past it the curve runs straight to the tensile
        # strength at half the elongation at break, an estimate of where necking starts
        n = fy / 10.0
        eng = [(fy * (p / 0.002) ** (1.0 / n) / E + p, fy * (p / 0.002) ** (1.0 / n))
               for p in (1e-4, 3e-4, 6e-4, 1e-3, 1.5e-3, 2e-3)]
        eng.append((max(0.5 * m.elongation / 100.0, 0.004 + fu / E), fu))
        return Hardening(_true_plastic(eng, E, start=1e-4), "multilinear", STEINHARDT)
    # bilinear: yield, then straight to the tensile strength at half the elongation at break, an estimate of the
    # uniform elongation where necking starts
    e_u = max(0.5 * m.elongation / 100.0, 2.0 * fy / E)
    eng = [(fy / E, fy), (e_u, fu)] if fu > fy else [(fy / E, fy)]
    src = ("bilinear from the datasheet yield and tensile strength, reached at half the elongation at break"
           if fu > fy else "elastic perfectly plastic at the datasheet yield, since no tensile strength is given")
    return Hardening(_true_plastic(eng, E), "bilinear", src)


def _true_plastic(eng: list[tuple[float, float]], E: float, start: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Engineering (strain, stress) points to CalculiX's (true stress, true plastic strain), the first one at plastic
    strain zero. ``start`` drops points below that plastic strain, so a rounded Ramberg-Osgood knee begins at its
    0.01% proof stress."""
    out = []
    for e, s in eng:
        st, et = s * (1.0 + e), math.log1p(e)
        ep = max(et - st / E, 0.0)
        if ep < 0.99 * start:
            continue
        if not out:
            ep = 0.0
        elif ep <= out[-1][1] or st < out[-1][0]:
            continue
        out.append((round(float(st), 4), round(float(ep), 6)))
    return tuple(out)


# --- magnetic and dielectric materials for the field studies ------------------------------
# DC magnetization curves are B in tesla against H in A/m. The ASM Metals Handbook curves were digitized for the
# FEMM 4.2 material library (D. Meeker, femm.info), which is where these numbers come from.
@dataclass(frozen=True)
class MagneticMaterial:
    key: str
    name: str
    mur: float = 1.0                 # relative permeability (linear materials, magnet recoil)
    bh: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    br: float = 0.0                  # remanence of a magnet, T
    hcj: float = 0.0                 # intrinsic coercivity of a magnet, A/m
    aliases: tuple[str, ...] = ()
    source: str = ""

    def knee(self, mur_diff: float = 100.0) -> float:
        """Flux density where the differential relative permeability falls below mur_diff, T (saturation)."""
        if not self.bh:
            return float("inf")
        b, h = self.bh
        for i in range(1, len(b)):
            if (b[i] - b[i - 1]) / max(h[i] - h[i - 1], 1e-12) / (4e-7 * math.pi) < mur_diff:
                return b[i - 1]
        return b[-1]


_MH = "ASM Metals Handbook DC magnetization curve, as digitized in the FEMM 4.2 material library"
MAGNETIC: dict[str, MagneticMaterial] = {m.key: m for m in [
    MagneticMaterial("air", "Air, copper, aluminum or plastic", aliases=("none", "copper", "aluminum", "plastic",
                                                                      "ss304", "304", "brass", "vacuum")),
    MagneticMaterial("1010-steel", "Low carbon steel 1010", bh=(
        (0.0, 0.2119, 0.2657, 0.3324, 0.4144, 0.5138, 0.6319, 0.7678, 0.917, 1.0704, 1.2143, 1.3346, 1.423, 1.4806,
         1.5172, 1.5445, 1.5713, 1.602, 1.6384, 1.6805, 1.7273, 1.7767, 1.8254, 1.8706, 1.9108, 1.9472, 1.9823,
         2.0183, 2.0554, 2.0925, 2.1281, 2.1616, 2.1946, 2.2303, 2.2724, 2.3243, 2.3894, 2.4712),
        (0.0, 79.6, 100.2, 126.1, 158.8, 199.9, 251.6, 316.8, 398.8, 502.1, 632.1, 795.8, 1001.8, 1261.2, 1587.8,
         1998.9, 2516.5, 3168.0, 3988.3, 5021.0, 6321.1, 7957.7, 10018.2, 12612.2, 15877.8, 19989.0, 25164.6,
         31680.4, 39883.2, 50210.0, 63210.6, 79577.5, 100182.1, 126121.8, 158777.9, 199889.6, 251646.1, 316803.6)),
        aliases=("1010", "1008", "low carbon steel", "steel", "iron", "1018", "12l14"),
        source="FEMM 4.2 material library, 1010 Steel"),
    MagneticMaterial("mild-steel", "Hot rolled low carbon steel", bh=(
        (0.0, 0.0554, 0.0826, 0.1375, 0.1701, 0.277, 0.407, 0.9834, 1.2472, 1.3003, 1.3439, 1.3681, 1.548, 1.5714,
         1.627, 1.634, 1.7342, 1.8664, 1.9876, 2.0274, 2.1014, 2.1252, 2.414, 2.4898),
        (0.0, 54.4, 60.3, 73.4, 81.4, 104.4, 129.6, 266.5, 470.4, 555.2, 655.3, 719.3, 1700.4, 1945.9, 2826.9,
         2977.4, 6091.7, 12988.3, 24965.0, 32025.0, 54377.3, 64205.3, 260993.3, 321280.0)),
        aliases=("a36", "hot rolled steel", "mild steel", "structural steel"), source=_MH),
    MagneticMaterial("cold-drawn-steel", "Cold drawn carbon steel, annealed", bh=(
        (0.0, 0.0909, 0.1214, 0.159, 0.1989, 0.3212, 0.3661, 0.6386, 0.8061, 0.9969, 1.0545, 1.2389, 1.3106, 1.4476,
         1.5272, 1.5696, 1.6139, 1.6637, 1.7641, 1.7814, 1.9319, 2.0894, 2.1336, 2.1967, 2.2435, 2.4709),
        (0.0, 82.4, 96.2, 112.4, 126.0, 154.8, 164.7, 233.8, 296.3, 399.6, 438.4, 610.1, 719.8, 1159.1, 1718.7,
         2204.4, 2856.9, 3819.5, 6827.1, 7495.0, 17912.7, 50567.0, 71237.0, 107930.0, 142883.3, 321350.0)),
        aliases=("cold drawn steel", "cold rolled steel", "bar stock"), source=_MH),
    MagneticMaterial("pure-iron", "Pure iron, annealed", bh=(
        (0.0, 0.0588, 0.1213, 0.229, 0.4727, 0.9517, 1.1915, 1.3111, 1.3628, 1.443, 1.5193, 1.5976, 1.6691, 1.6919,
         1.7425, 1.7815, 1.82, 1.8702, 1.9361, 2.0256, 2.1086, 2.1723, 2.2847, 2.4118, 2.5628),
        (0.0, 16.3, 18.4, 22.4, 31.9, 62.2, 92.1, 118.0, 136.4, 176.7, 243.6, 409.3, 999.5, 1437.5, 2571.0, 3976.1,
         5600.2, 7969.0, 11696.8, 17893.1, 27661.2, 41894.6, 106704.4, 203216.7, 320990.0)),
        aliases=("armco", "soft iron", "electrical iron"), source=_MH),
    MagneticMaterial("m19", "Silicon steel M-19 laminations", bh=(
        (0.0, 0.0562, 0.093, 0.3607, 0.6568, 0.9245, 1.1448, 1.2407, 1.3226, 1.3564, 1.4314, 1.5568, 1.5995, 1.7207,
         1.8445, 1.882, 1.9773, 2.0382, 2.0859, 2.1646, 2.3428),
        (0.0, 22.7, 26.5, 48.3, 82.7, 141.5, 260.5, 394.3, 615.8, 765.7, 1518.9, 5010.8, 6494.2, 11605.3, 19895.7,
         23730.7, 44232.0, 83364.0, 121186.7, 183646.7, 325220.0)),
        aliases=("m-19", "silicon steel", "electrical steel", "lamination", "laminations"), source=_MH),
    MagneticMaterial("ss416", "Stainless steel 416, annealed", bh=(
        (0.0, 0.2078, 0.2617, 0.2889, 0.4122, 0.6257, 0.7825, 0.9582, 1.0768, 1.2056, 1.3726, 1.4276, 1.5423,
         1.6334, 1.6932, 1.764, 1.8508, 2.0216, 2.0765),
        (0.0, 320.9, 355.8, 374.7, 470.2, 660.9, 837.8, 1165.8, 1509.6, 2284.1, 4970.0, 6783.8, 12124.3, 21451.7,
         36049.7, 76959.0, 143556.7, 279173.3, 322880.0)),
        aliases=("416", "magnetic stainless", "430"), source=_MH),
    # sintered magnets: middle of each grade's remanence range in makers' grade tables, recoil permeability 1.05
    MagneticMaterial("n35", "NdFeB N35", mur=1.05, br=1.19, hcj=955e3, aliases=("neodymium", "ndfeb"),
                     source="sintered NdFeB grade tables (Br 1.17 to 1.21 T, Hcj at least 955 kA/m)"),
    MagneticMaterial("n42", "NdFeB N42", mur=1.05, br=1.30, hcj=955e3, aliases=("n40", "n45"),
                     source="sintered NdFeB grade tables (Br 1.28 to 1.32 T, Hcj at least 955 kA/m)"),
    MagneticMaterial("n52", "NdFeB N52", mur=1.05, br=1.44, hcj=876e3, aliases=("n50", "n48"),
                     source="sintered NdFeB grade tables (Br 1.42 to 1.47 T, Hcj at least 876 kA/m)"),
    MagneticMaterial("ferrite", "Ferrite magnet, grade C8", mur=1.1, br=0.39, hcj=250e3,
                     aliases=("ceramic", "c8", "ceramic 8", "y30", "hard ferrite"),
                     source="ceramic grade C8 tables (Br about 0.39 T, Hcj about 250 kA/m)"),
]}


@dataclass(frozen=True)
class Dielectric:
    key: str
    name: str
    epsr: float                      # relative permittivity
    strength: float                  # dielectric strength, kV/mm (short term, for the thickness tested)
    loss_tangent: float = 0.0
    aliases: tuple[str, ...] = ()
    source: str = ""


DIELECTRICS: dict[str, Dielectric] = {d.key: d for d in [
    Dielectric("air", "Air at sea level", 1.0006, 3.0, aliases=("vacuum", "none"),
               source="about 3 kV/mm in uniform fields across gaps of a few mm (Paschen); less across wider gaps"),
    Dielectric("fr4", "FR4 laminate", 4.4, 20.0, 0.02, aliases=("pcb", "g10"),
               source="typical FR4 at 1 MHz to 1 GHz (IPC-4101/21 allows up to 5.4); strength through 1.6 mm"),
    Dielectric("ro4350b", "Rogers RO4350B", 3.66, 31.2, 0.0037, aliases=("rogers", "ro4350"),
               source="Rogers RO4350B datasheet, design Dk 3.66 at 8 to 40 GHz, 31.2 kV/mm"),
    Dielectric("ptfe", "PTFE", 2.1, 60.0, 0.0002, aliases=("teflon",), source="Chemours Teflon PTFE data, thin film"),
    Dielectric("polyimide", "Polyimide film (Kapton HN)", 3.4, 300.0, 0.002, aliases=("kapton", "flex"),
               source="DuPont Kapton HN datasheet, 25 um film"),
    Dielectric("alumina", "Alumina 96%", 9.4, 15.0, 0.0003, aliases=("ceramic substrate", "al2o3"),
               source="CoorsTek AD-96 data"),
    Dielectric("glass", "Soda lime glass", 7.0, 10.0, 0.01, source="typical soda lime glass"),
    Dielectric("mineral-oil", "Transformer oil", 2.2, 12.0, 0.0005, aliases=("oil",),
               source="IEC 60296 oils, about 12 kV/mm in a 2.5 mm gap test"),
    Dielectric("pe", "Polyethylene", 2.25, 20.0, 0.0003, aliases=("polyethylene", "coax dielectric"),
               source="typical solid polyethylene cable insulation"),
]}


def get_magnetic(name: str) -> MagneticMaterial:
    return _lookup(name, MAGNETIC, "magnetic material")


def get_dielectric(name: str) -> Dielectric:
    if name.lower().startswith("er="):
        v = float(name.split("=", 1)[1])
        return Dielectric("custom", f"Dielectric with relative permittivity {v:g}", v, 0.0, source="user supplied")
    return _lookup(name, DIELECTRICS, "dielectric")


def _lookup(name: str, table: dict, what: str):
    k = _clean(name)
    for m in table.values():
        if k in (_clean(m.key), _clean(m.name), *(_clean(a) for a in m.aliases)):
            return m
    close = difflib.get_close_matches(k, [_clean(m.key) for m in table.values()], n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise KeyError(f"unknown {what} '{name}'.{hint} Known: {', '.join(table)}")

