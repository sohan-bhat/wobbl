"""Reads a run's scene.wobbl the way site/src/scripts/scene-file.js reads it (a port of site/blender/lib/wobscene.py).

The models are built where each object stands at the frame the example page opens on, then moved into the
object's own frame for export, so a part lands exactly where the simulation put it.
"""
import gzip
import json
import math
import struct

import numpy as np

_T = {'u8': np.uint8, 'u16': np.uint16, 'u32': np.uint32, 'f32': np.float32}


class Scene:
    def __init__(self, path):
        raw = open(path, 'rb').read()
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        if raw[:4] != b'WBSC':
            raise ValueError(f'{path} is not a Wobbl scene')
        n = struct.unpack_from('<I', raw, 8)[0]
        self.h = json.loads(raw[12:12 + n].decode('utf-8'))
        at = 12 + n
        at += (8 - at % 8) % 8
        self.blob = raw[at:]
        self.path = path

    @property
    def objects(self):
        return self.h['objects']

    def arr(self, d):
        """The numbers behind a buffer reference, dequantized, or the value itself when it is inline."""
        if d is None:
            return None
        if not isinstance(d, dict) or 'b' not in d:
            return np.asarray(d, dtype=np.float64)
        b = self.h['buffers'][d['b']]
        q = np.frombuffer(self.blob, dtype=_T[b['t']], count=b['n'] // np.dtype(_T[b['t']]).itemsize, offset=b['o'])
        if not d.get('lo'):
            return q.astype(np.float64)
        top_all = 255 if b['t'] == 'u8' else 65535
        top = top_all - 1 if d.get('nan') else top_all
        c = d['c']
        lo = np.array(d['lo'], dtype=np.float64)
        hi = np.array(d['hi'], dtype=np.float64)
        qq = q.astype(np.float64).reshape(-1, c)
        out = lo + (qq / top) * (hi - lo)
        if d.get('nan'):
            out[q.reshape(-1, c) == top_all] = np.nan
        return out.reshape(-1)

    def find(self, name):
        """Indexes of every object with this name."""
        return [i for i, o in enumerate(self.objects) if o.get('name') == name]

    def home(self):
        """The saved frame the example page opens on: the value the question asked about (scenes.mjs)."""
        tl = self.h['timeline']
        t = list(self.arr(tl['t'])) if isinstance(tl['t'], dict) else list(tl['t'])
        F = tl['frames']
        if tl.get('design') is None:
            return F - 1
        x = tl['design']
        pos = F - 1
        for i in range(F - 1):
            if x <= t[i + 1]:
                pos = min(max(i + (x - t[i]) / ((t[i + 1] - t[i]) or 1), i), i + 1)
                break
        return int(round(pos))

    def pose(self, i, frame=None):
        """(position, quaternion w x y z) of object i at a saved frame, or None when it has no pose."""
        o = self.objects[i]
        if frame is None:
            frame = self.home()
        if o['kind'] == 'mesh':
            if not o.get('pose'):
                return None
            pp, pq = self.arr(o['pose']['pos']), self.arr(o['pose']['quat'])
        elif o['kind'] == 'body':
            pp, pq = self.arr(o['pos']), self.arr(o['quat'])
        else:
            return None
        p = pp[frame * 3:frame * 3 + 3] if len(pp) > 3 else pp[:3]
        q = pq[frame * 4:frame * 4 + 4] if len(pq) > 4 else pq[:4]
        q = q / (np.linalg.norm(q) or 1)
        return [float(x) for x in p], [float(x) for x in q]

    def matrix(self, i, frame=None):
        """The object's 4x4 local to world matrix at a frame, as a numpy array (identity without a pose)."""
        ps = self.pose(i, frame)
        M = np.eye(4)
        if not ps:
            return M
        p, (w, x, y, z) = ps
        M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
        M[:3, 3] = p
        return M

    def mesh(self, i):
        """Vertices (n, 3) and triangles (m, 3) of a mesh object in its own frame."""
        o = self.objects[i]
        P = self.arr(o['pos']).reshape(-1, 3)
        T = self.arr(o['tri']).astype(np.int64).reshape(-1, 3)
        return P, T


    def inverse(self, i, frame=None):
        """The world to local matrix of object i at a frame."""
        return np.linalg.inv(self.matrix(i, frame))

    def named(self, name, frame=None):
        """[(index, object, 4x4 matrix)] for every object with this name."""
        return [(i, self.objects[i], self.matrix(i, frame)) for i in self.find(name)]


def world(M, p):
    """A point through a 4x4 matrix."""
    return (M @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def angle_z(q):
    """The turn about z of a quaternion that only turns about z, in radians."""
    w, x, y, z = q
    return 2 * math.atan2(z, w)
