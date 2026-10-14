"""Small pieces the adapters share: reading a result folder, the verdict in plain words, a revolved mesh."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

WORDS = {"ok": ("OK", "pass"), "holds": ("HOLDS", "pass"), "pass": ("PASSES", "pass"), "marginal": ("MARGINAL", "warn"),
         "warm": ("WARM", "warn"), "fails": ("FAILS", "fail"), "fail": ("FAILS", "fail"), "hot": ("TOO HOT", "fail"),
         "stalls": ("STALLS", "fail"), "slips": ("SLIPS", "fail"), "tips": ("TIPS", "fail"),
         "runaway": ("RUNAWAY", "fail"), "safe": ("SAFE", "pass"), "stable": ("STABLE", "pass")}


def read(out: Path, name: str):
    p = Path(out) / name
    return json.loads(p.read_text()) if p.exists() else None


def plain(text: str, limit: int = 170) -> str:
    """A headline as one plain sentence: the first sentence, without bracketed asides."""
    t = re.sub(r"\s*\([^)]*\)", "", (text or "").strip())
    first = re.split(r"(?<=[.!?])\s", t)[0].strip()
    return first if len(first) <= limit else first[: limit - 1].rsplit(" ", 1)[0] + "."


def verdict(scene, o: dict, line: str | None = None):
    v = str(o.get("verdict", "")).lower()
    word, tone = WORDS.get(v, (v.upper() or "DONE", "info"))
    scene.verdict(word, line or plain(o.get("headline", "")), tone)


def words(o: dict) -> set[str]:
    return set(re.findall(r"[a-z]+", (o.get("question") or "").lower()))


def lathe(profile, seg: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A surface of revolution about z from (radius, z) points, closed at either end where the radius is zero."""
    pr = np.asarray(profile, float)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    X = np.concatenate([np.c_[r * np.cos(a), r * np.sin(a), np.full(seg, z)] for r, z in pr])
    T = []
    for i in range(len(pr) - 1):
        for j in range(seg):
            p, q = i * seg + j, i * seg + (j + 1) % seg
            T += [(p, q, q + seg), (p, q + seg, p + seg)]
    return X, np.asarray(T)


def join(*parts) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ts, n = [], [], 0
    for X, T in parts:
        Xs.append(X)
        Ts.append(np.asarray(T) + n)
        n += len(X)
    return np.concatenate(Xs), np.concatenate(Ts)


def box_mesh(center, half, rot=np.eye(3)) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    X = c @ np.asarray(rot).T + center
    T = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3)]
    return X, np.asarray(T)


def quats_along(d: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z) turning +z onto each direction, frame by frame."""
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    c = d[:, 2]
    w = np.sqrt(np.clip((1 + c) / 2, 0, 1))
    s = 1 / np.maximum(2 * w, 1e-9)
    q = np.c_[w, -d[:, 1] * s, d[:, 0] * s, np.zeros(len(d))]
    q[c < -0.999999] = [0, 1, 0, 0]
    return q
