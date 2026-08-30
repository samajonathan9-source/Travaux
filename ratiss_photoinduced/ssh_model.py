"""Chaîne SSH (Su-Schrieffer-Heeger) — cristal topologique jouet à vérité analytique.

H = sum_i [ t1 c+_{2i} c_{2i+1} + t2 c+_{2i+1} c_{2i+2} + h.c. ]
t1 = t*(1+delta), t2 = t*(1-delta), t = 1.

Convention : delta > 0 -> phase triviale (winding 0)
             delta < 0 -> phase topologique (winding 1, états de bord)
Transition analytique exacte à delta = 0 (limite thermodynamique).
Le "laser" est modélisé par un pilotage temporel delta(t).
"""

from __future__ import annotations

import numpy as np


def ssh_hamiltonian(n_sites: int, delta: float, t: float = 1.0) -> np.ndarray:
    """Hamiltonien SSH à bords ouverts (n_sites pair)."""
    if n_sites % 2 != 0:
        raise ValueError("n_sites doit être pair (cellules de 2 sites)")
    t1 = t * (1.0 + delta)
    t2 = t * (1.0 - delta)
    h = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        hop = t1 if i % 2 == 0 else t2
        h[i, i + 1] = hop
        h[i + 1, i] = hop
    return h


def ground_state_orbitals(h: np.ndarray) -> np.ndarray:
    """Orbitales occupées à mi-remplissage (n/2 plus basses), spinless."""
    n = h.shape[0]
    _, vecs = np.linalg.eigh(h)
    return vecs[:, : n // 2]


def correlation_matrix(orbitals: np.ndarray) -> np.ndarray:
    """Matrice de corrélation C_ij = <c+_i c_j> (projecteur sur états occupés)."""
    return orbitals @ orbitals.conj().T


def spectral_gap(h: np.ndarray) -> float:
    """Gap à mi-remplissage de la chaîne finie."""
    n = h.shape[0]
    eigs = np.linalg.eigvalsh(h)
    return float(eigs[n // 2] - eigs[n // 2 - 1])


def winding_number(delta: float, t: float = 1.0, n_k: int = 4001) -> float:
    """Nombre d'enroulement de h(k) = t1 + t2 e^{-ik} (zone de Brillouin)."""
    t1 = t * (1.0 + delta)
    t2 = t * (1.0 - delta)
    k = np.linspace(0.0, 2.0 * np.pi, n_k, endpoint=False)
    hk = t1 + t2 * np.exp(-1j * k)
    phase = np.unwrap(np.angle(hk))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def edge_weight(corr: np.ndarray) -> float:
    """Corrélation bord-bord |C_{0,N-1}| — signature des états de bord."""
    return float(abs(corr[0, -1]))


def evolve_orbitals(orbitals0: np.ndarray, delta_of_t, dt: float, n_steps: int, t0: float = 0.0) -> np.ndarray:
    """Propagation temps réel i dpsi/dt = H(t) psi par RK4, orbite par orbite.

    Retourne un tableau (n_steps+1, n_sites, n_occ).
    """
    out = np.empty((n_steps + 1,) + orbitals0.shape, dtype=complex)
    out[0] = orbitals0

    def rhs(tt: float, psi: np.ndarray) -> np.ndarray:
        return -1j * (ssh_hamiltonian(psi.shape[0], delta_of_t(tt)) @ psi)

    psi = orbitals0.copy()
    t = t0
    for s in range(1, n_steps + 1):
        k1 = rhs(t, psi)
        k2 = rhs(t + dt / 2, psi + dt * k1 / 2)
        k3 = rhs(t + dt / 2, psi + dt * k2 / 2)
        k4 = rhs(t + dt, psi + dt * k3)
        psi = psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        out[s] = psi
    return out
