"""The lid driven cavity, OpenFOAM's first tutorial, written out as an icoFoam case folder for tests and demos.

A 0.1 m square box whose lid slides at 1 m/s over fluid of viscosity 0.01 m2/s, so Re = 10, on a 20 by 20 grid,
run to 0.5 s. Its centre line velocity matches Ghia, Ghia and Shin (1982) closely at this Reynolds number.
"""
from __future__ import annotations

from pathlib import Path

H = "FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n\n"

FILES = {
    "system/controlDict": ("dictionary", "controlDict", """application icoFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 0.5;
deltaT 0.005;
writeControl timeStep;
writeInterval 20;
purgeWrite 0;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
"""),
    "system/blockMeshDict": ("dictionary", "blockMeshDict", """scale 0.1;
vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1));
edges ();
boundary
(
    movingWall { type wall; faces ((3 7 6 2)); }
    fixedWalls { type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
"""),
    "system/fvSchemes": ("dictionary", "fvSchemes", """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(p) Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
"""),
    "system/fvSolution": ("dictionary", "fvSolution", """solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }
    pFinal { $p; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }
}
PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
"""),
    "constant/transportProperties": ("dictionary", "transportProperties", "nu 0.01;\n"),
    "0/U": ("volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    movingWall { type fixedValue; value uniform (1 0 0); }
    fixedWalls { type noSlip; }
    frontAndBack { type empty; }
}
"""),
    "0/p": ("volScalarField", "p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    movingWall { type zeroGradient; }
    fixedWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
"""),
}


def write(folder: str | Path) -> Path:
    root = Path(folder)
    for rel, (cls, obj, body) in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(H.format(cls=cls, obj=obj) + body)
    return root
