"""Pieces shared by the 3D GetDP problems built on a general scene: groups, the infinite shell, slices."""
from __future__ import annotations

import numpy as np

from .forms import rname


def num(x) -> str:
    """A number as GetDP reads it, never a numpy repr."""
    return repr(float(x))


def groups(tags: dict, extra: dict[str, list[str]]) -> str:
    lines = [f"  {rname(n)} = Region[{t}];" for n, t in tags.items()]
    for g, names in extra.items():
        ts = [str(tags[n]) for n in names if n in tags]
        lines.append(f"  {g} = Region[{{{', '.join(ts)}}}];")
    return "Group {\n" + "\n".join(lines) + "\n}\n"


def jacobian(msh, shell: bool) -> str:
    sh = ""
    if shell and "shell" in msh.tags:
        # Gauss points of the shell's first layer can sit a little inside the true radius, by a hair on curved
        # elements and by the sagitta of a facet on straight ones. A shell twice as wide as its inner radius keeps
        # the mapping continuous to first order when its start moves in a little.
        rin = msh.radius * (1 - 5e-4 if getattr(msh, "order", 1) == 2 else 0.97)
        nums = ", ".join(num(x) for x in (rin, msh.outer, *msh.center))
        sh = f"      {{ Region {rname('shell')}; Jacobian VolSphShell{{{nums}}}; }}\n"
    return ("Jacobian {\n"
            f"  {{ Name Vol; Case {{\n{sh}      {{ Region All; Jacobian Vol; }} }} }}\n"
            "  { Name Sur; Case { { Region All; Jacobian Sur; } } }\n}\n")


INTEGRATION = """Integration {
  { Name I1; Case { { Type Gauss; Case {
      { GeoElement Point; NumberOfPoints 1; } { GeoElement Line; NumberOfPoints 3; }
      { GeoElement Triangle; NumberOfPoints 3; } { GeoElement Tetrahedron; NumberOfPoints 4; }
      { GeoElement Line2; NumberOfPoints 3; } { GeoElement Triangle2; NumberOfPoints 6; }
      { GeoElement Tetrahedron2; NumberOfPoints 15; } } } } }
  { Name I2; Case { { Type Gauss; Case {
      { GeoElement Point; NumberOfPoints 1; } { GeoElement Line; NumberOfPoints 3; } { GeoElement Line2; NumberOfPoints 4; }
      { GeoElement Triangle; NumberOfPoints 6; } { GeoElement Tetrahedron; NumberOfPoints 15; }
      { GeoElement Triangle2; NumberOfPoints 12; } { GeoElement Tetrahedron2; NumberOfPoints 15; } } } } }
}
"""


def newton(tol: float = 1e-8, max_iter: int = 40) -> str:
    """Operations for a Newton loop on the residual, as GetDP's own nonlinear templates do it."""
    return (" Generate[A]; GetResidual[A, $res0]; Evaluate[$res = $res0, $iter = 0];"
            ' Print[{$iter, $res, $res / $res0}, Format "Residual %03g: abs %14.12e rel %14.12e"];'
            f" While[$res > 1e-30 && $res / $res0 > {tol} && $res / $res0 <= 1 && $iter < {max_iter}] {{"
            " Solve[A]; Generate[A]; GetResidual[A, $res]; Evaluate[$iter = $iter + 1];"
            ' Print[{$iter, $res, $res / $res0}, Format "Residual %03g: abs %14.12e rel %14.12e"]; }')


def slice_plane(bbox: tuple, through: tuple | None = None, normal: str | None = None, n: int = 80,
                margin: float = 0.35) -> dict:
    """A plane for pictures and playback: it holds the scene's thinnest and longest directions, so plates,
    gaps and coils are cut across, and passes through a point of interest (the scene's center by default)."""
    lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
    ext = hi - lo
    order = np.argsort(ext)
    ax = "xyz"
    k = ax.index(normal) if normal else int(order[1])
    u, v = sorted(i for i in range(3) if i != k)
    c = 0.5 * (lo + hi) if through is None else np.array(through, float)
    pad = margin * max(ext[u], ext[v])
    u0, u1, v0, v1 = lo[u] - pad, hi[u] + pad, lo[v] - pad, hi[v] + pad
    nu = int(n if ext[u] + 2 * pad >= ext[v] + 2 * pad else max(16, round(n * (u1 - u0) / (v1 - v0))))
    nv = int(max(16, round(nu * (v1 - v0) / (u1 - u0))))
    if nv > n:
        nv, nu = n, int(max(16, round(n * (u1 - u0) / (v1 - v0))))
    return {"normal": ax[k], "at": float(c[k]), "u": ax[u], "v": ax[v], "u0": float(u0), "u1": float(u1),
            "v0": float(v0), "v1": float(v1), "nu": nu, "nv": nv, "iu": u, "iv": v, "ik": k}


def print_plane(quantity: str, pl: dict, fname: str, step: str = "") -> str:
    """A GetDP Print of a quantity on the plane's grid (it writes x y z then the values per point)."""
    def pt(a, b):
        p = [0.0, 0.0, 0.0]
        p[pl["ik"]], p[pl["iu"]], p[pl["iv"]] = pl["at"], a, b
        return "{" + ", ".join(repr(float(x)) for x in p) + "}"
    corners = pt(pl["u0"], pl["v0"]) + pt(pl["u1"], pl["v0"]) + pt(pl["u0"], pl["v1"])
    return (f'Print[ {quantity}, OnPlane {{{corners}}} {{{pl["nu"] - 1}, {pl["nv"] - 1}}},{step} '
            f'Format SimpleTable, File "{fname}" ];')


def read_plane(path, pl: dict, comps: int = 1) -> np.ndarray:
    """The plane's grid (nv, nu, comps) with NaN where the point is outside the mesh (inside a conductor)."""
    out = np.full((pl["nv"], pl["nu"], comps), np.nan)
    rows = []
    try:
        with open(path) as fh:
            for ln in fh:              # a point on a conductor's surface can come back short, and is skipped
                t = ln.split()
                if len(t) >= 3 + comps:
                    rows.append([float(x) for x in t[:3]] + [float(x) for x in t[-comps:]])
    except (OSError, ValueError):
        return out[..., 0] if comps == 1 else out
    if not rows:
        return out[..., 0] if comps == 1 else out
    rows = np.array(rows)
    xyz, vals = rows[:, :3], rows[:, -comps:]
    du = (pl["u1"] - pl["u0"]) / (pl["nu"] - 1)
    dv = (pl["v1"] - pl["v0"]) / (pl["nv"] - 1)
    iu = np.rint((xyz[:, pl["iu"]] - pl["u0"]) / du).astype(int)
    iv = np.rint((xyz[:, pl["iv"]] - pl["v0"]) / dv).astype(int)
    ok = (iu >= 0) & (iu < pl["nu"]) & (iv >= 0) & (iv < pl["nv"])
    out[iv[ok], iu[ok]] = vals[ok]
    return out[..., 0] if comps == 1 else out
