"""Modèle synthétique du tissage 3D ferroélectrique (KTN:Li).

Les domaines entrelacés du KTN:Li (Light: Science & Applications, 2026,
doi:10.1038/s41377-026-02374-7) sont modélisés comme des spirales
entrelacées (brins hélicoïdaux) — le motif "woven" — contre des domaines
parallèles alignés — le motif classique. La réécriture optique est une
perturbation gaussienne locale; la régénération thermique est un nouveau
motif à chaque cycle.
"""

from __future__ import annotations

import numpy as np


def make_woven(n_points: int = 900, n_strands: int = 9, turn: float = 2.0,
               noise: float = 0.02, seed: int = 42) -> np.ndarray:
    """Tissage 3D : brins hélicoïdaux entrelacés (analogie KTN:Li woven)."""
    rng = np.random.default_rng(seed)
    pts = []
    per = n_points // n_strands
    for s in range(n_strands):
        t = np.linspace(0.0, np.pi * turn, per)
        phase = 2.0 * np.pi * s / n_strands
        x = np.cos(t + phase)
        y = np.sin(t + phase)
        z = t * 3.0
        e = rng.normal(0.0, noise, (len(t), 3))
        pts.append(np.column_stack([x + e[:, 0], y + e[:, 1], z + e[:, 2]]))
    return np.vstack(pts)


def make_aligned(n_points: int = 900, n_strands: int = 9, noise: float = 0.02,
                 seed: int = 42) -> np.ndarray:
    """Domaines parallèles classiques (non tissé) : lignes droites."""
    rng = np.random.default_rng(seed)
    pts = []
    per = n_points // n_strands
    for s in range(n_strands):
        t = np.linspace(0.0, 5.0, per)
        x = np.full_like(t, s * 2.0)
        y = t.copy()
        z = np.zeros(len(t))
        e = rng.normal(0.0, noise, (len(t), 3))
        pts.append(np.column_stack([x + e[:, 0], y + e[:, 1], z + e[:, 2]]))
    return np.vstack(pts)


def subsample(pts: np.ndarray, n: int = 80, seed: int = 0) -> np.ndarray:
    """Sous-échantillonne pour rester sous la limite de l'engine Rips."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(n, len(pts)), replace=False)
    return pts[idx]


def distance_matrix(pts: np.ndarray) -> np.ndarray:
    """Matrice de distance euclidienne."""
    dm = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dm, 0.0)
    return dm


def optical_unweave(pts: np.ndarray, strength: float = 2.0,
                    focus: np.ndarray | None = None) -> np.ndarray:
    """Réécriture optique (laser 514 nm) : perturbation gaussienne locale qui
    redresse localement les brins vers leur moyenne (démêlage partiel)."""
    out = pts.copy()
    if focus is None:
        focus = pts.mean(axis=0)
    sigma = pts.std(axis=0).mean()
    for i in range(len(out)):
        d = np.linalg.norm(out[i] - focus)
        w = np.exp(-d * d / (2.0 * sigma * sigma))
        target = np.array([out[i, 0], 0.0, out[i, 2]])  # projette sur l'axe
        out[i] = out[i] + strength * w * (target - out[i])
    return out


def thermal_regenerate(n_points: int = 900, n_strands: int = 9,
                       seed: int = 42) -> np.ndarray:
    """Régénération thermique : nouveau motif tissé à chaque cycle (seed neuf)."""
    return make_woven(n_points=n_points, n_strands=n_strands,
                      turn=2.0, noise=0.02, seed=seed)
