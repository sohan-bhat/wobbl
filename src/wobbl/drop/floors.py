"""What a dropped part lands on.

A floor is either hard (concrete, steel, stone, tile), far stiffer than any part landing on it, or a soft layer
over a hard base (wood flooring, vinyl, carpet, packaging foam). A soft layer acts as a bed of springs with its
compressive stiffness over its thickness; foams crush at a plateau stress and keep the dent; every layer bottoms
out on the hard base under it once it is squeezed nearly flat.

Each value names where it comes from. Floors vary a lot, so the moduli of the soft layers are typical values,
and the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloorMaterial:
    key: str
    name: str
    hard: bool
    modulus: float = 0.0        # MPa, compressive modulus of the soft layer
    thickness: float = 0.0      # mm
    plateau: float = 0.0        # MPa, crush stress of a foam (0 for an elastic layer)
    bottom: float = 0.8         # share of the thickness at which the layer is squeezed flat onto its base
    source: str = ""

    def describe(self) -> str:
        if self.hard:
            return f"{self.name}, far stiffer than the part ({self.source})"
        s = f"{self.name}: a {self.thickness:g} mm layer, compressive modulus {self.modulus:g} MPa"
        if self.plateau:
            s += f", crushing at {self.plateau:g} MPa"
        return s + f", over a hard base ({self.source})"


FLOORS = {f.key: f for f in [
    FloorMaterial("concrete", "Concrete slab", True,
                  source="E about 30 GPa, ACI 318-19 19.2.2.1, Ec = 4700 sqrt(f'c) for 40 MPa concrete"),
    FloorMaterial("steel", "Steel plate", True, source="E 200 GPa"),
    FloorMaterial("tile", "Ceramic tile on concrete", True, source="porcelain tile E about 50 GPa"),
    FloorMaterial("stone", "Stone or terrazzo", True, source="granite E about 50 GPa"),
    FloorMaterial("wood", "Hardwood flooring on concrete", False, 1300.0, 19.0,
                  source="oak across the grain, E_R about 0.1 E_L with E_L 12.6 GPa, USDA Wood Handbook "
                         "FPL-GTR-282 tables 5-1 and 5-2; 19 mm boards"),
    FloorMaterial("vinyl", "Vinyl or linoleum on concrete", False, 20.0, 2.5,
                  source="typical resilient sheet flooring, not from a single source"),
    FloorMaterial("carpet", "Carpet with underlay on concrete", False, 0.4, 12.0,
                  source="typical cut pile carpet on foam underlay, not from a single source"),
    FloorMaterial("foam", "Expanded polystyrene packaging foam", False, 4.0, 25.0, plateau=0.1, bottom=0.7,
                  source="EPS 100 grade of EN 13163, 100 kPa at 10% strain; modulus typical"),
    FloorMaterial("grass", "Grass on soil", False, 5.0, 100.0, plateau=0.3, bottom=0.6,
                  source="typical soft soil, not from a single source"),
]}
ALIASES = {"floor": "concrete", "pavement": "concrete", "asphalt": "concrete", "hard": "concrete",
           "laminate": "wood", "hardwood": "wood", "parquet": "wood", "plywood": "wood", "linoleum": "vinyl",
           "eps": "foam", "styrofoam": "foam", "packaging": "foam", "rug": "carpet", "soil": "grass",
           "ground": "grass", "granite": "stone", "terrazzo": "stone", "ceramic": "tile", "metal": "steel"}


def get_floor(text: str) -> FloorMaterial:
    """A floor by name, or a custom soft layer as 'layer:E=5,t=10' (MPa, mm) with plateau=... for a foam."""
    t = (text or "concrete").strip().lower()
    if t.startswith("layer") or t.startswith("custom"):
        kv = dict(p.split("=", 1) for p in t.split(":", 1)[-1].replace(" ", "").split(",") if "=" in p)
        try:
            return FloorMaterial("custom", "Custom layer", False, float(kv["e"]), float(kv["t"]),
                                 float(kv.get("plateau", 0.0)), float(kv.get("bottom", 0.8)), "given")
        except (KeyError, ValueError):
            raise KeyError(f"a custom floor needs E and t, e.g. 'layer:E=5,t=10' (MPa, mm), not '{text}'") from None
    t = ALIASES.get(t, t)
    if t not in FLOORS:
        raise KeyError(f"unknown floor '{text}'. Floors: {', '.join(FLOORS)}, or 'layer:E=5,t=10'")
    return FLOORS[t]


__all__ = ["FloorMaterial", "FLOORS", "get_floor"]
