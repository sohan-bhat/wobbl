"""Write VTK XML unstructured grids (.vtu) without VTK, for ParaView or any VTK reader.

The file is VTK's own XML layout with each array zlib compressed into an
appended binary block, as ParaView writes it. Coordinates and fields are 32
bit floats, plenty for viewing and half the size of doubles.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

TETRA, QUADRATIC_TETRA = 10, 24      # VTK cell types; a quadratic tetra's node order is CalculiX's C3D10 order
_TYPES = {"f4": "Float32", "f8": "Float64", "i4": "Int32", "i8": "Int64", "u1": "UInt8"}


def _block(a: np.ndarray) -> bytes:
    """One array as an appended block: the UInt64 header VTK expects (one block, its size, its compressed size),
    then the compressed little endian bytes."""
    raw = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes()
    comp = zlib.compress(raw, 6)
    return np.array([1, len(raw), len(raw), len(comp)], "<u8").tobytes() + comp


def write_vtu(path: str | Path, points, cells, point_data: dict) -> str:
    """Points (n, 3), linear (m, 4) or quadratic (m, 10) tetrahedra, and nodal fields of 1, 3 or 6 components.

    A 6 component field is a symmetric tensor in VTK's order xx, yy, zz, xy, yz, xz.
    """
    cells = np.asarray(cells)
    blocks, arrays, offset = [], [], 0

    def add(a: np.ndarray, name: str = "") -> str:
        nonlocal offset
        b = _block(a)
        blocks.append(b)
        comps = "" if a.ndim == 1 else f' NumberOfComponents="{a.shape[1]}"'
        tag = (f'<DataArray type="{_TYPES[a.dtype.str[1:]]}"' + (f' Name="{name}"' if name else "") + comps +
               f' format="appended" offset="{offset}"/>')
        offset += len(b)
        return tag

    for name, a in point_data.items():
        arrays.append(add(np.asarray(a, np.float32), name))
    idx = np.int32 if cells.size == 0 or cells.max() < 2 ** 31 - 1 else np.int64
    pts = add(np.asarray(points, np.float32))
    conn = add(cells.astype(idx).ravel(), "connectivity")
    offs = add((np.arange(1, len(cells) + 1) * cells.shape[1]).astype(idx), "offsets")
    kinds = add(np.full(len(cells), QUADRATIC_TETRA if cells.shape[1] == 10 else TETRA, np.uint8), "types")
    head = ('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64" compressor="vtkZLibDataCompressor">\n<UnstructuredGrid>\n'
            f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">\n'
            '<PointData>\n' + "\n".join(arrays) + '\n</PointData>\n'
            f'<Points>\n{pts}\n</Points>\n<Cells>\n{conn}\n{offs}\n{kinds}\n</Cells>\n'
            '</Piece>\n</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(head.encode())
        for b in blocks:
            f.write(b)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")
    return str(path)
