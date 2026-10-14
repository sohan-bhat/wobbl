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

