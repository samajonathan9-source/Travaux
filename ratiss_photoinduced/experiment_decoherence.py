"""Decoherence : robustesse du signal topologique au bruit (Lindblad).

Equation maitresse de Lindblad pour la matrice densite rho :
  d rho/dt = -i [H(t), rho] + gamma * sum_k D[L_k](rho)
avec D[L](rho) = L rho L+ - (1/2){L+ L, rho}.

Canaux : dephasage local L_k = n_k = c+_k c_k (projecteur occupation site k).
C est le bruit dominant dans les cristaux photo-induits (couplage phonons)
et sur les QPU supraconducteurs (T2).

Question : le signal P_sig survit-il a la decoherence ? Jusqu a quel taux
gamma ? C est la question cle pour la faisabilite QPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.ssh_model import (
    correlation_matrix,
    ground_state_orbitals,
    spectral_gap,
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_SITES = 16
DELTA_START = 0.4
DELTA_END = -0.4
T_RAMP = 100.0
DT = 0.5  # pas plus gros : Lindblad est plus couteux que RK4


def delta_of_t(t: float) -> float:
    frac = min(t / T_RAMP, 1.0)
    return DELTA_START + (DELTA_END - DELTA_START) * frac


def number_operators(n_sites: int) -> list[np.ndarray]:
    """Projecteurs d occupation n_k = |k><k| (canaux de dephasage)."""
    ops = []
    for k in range(n_sites):
        n = np.zeros((n_sites, n_sites))
        n[k, k] = 1.0
        ops.append(n)
    return ops


def lindblad_step(rho: np.ndarray, h: np.ndarray, l_ops: list[np.ndarray], gamma: float, dt: float) -> np.ndarray:
    """Un pas de Lindblad par splitting d operateurs (stable).

    1) Evolution unitaire exacte : rho -> U rho U+  avec U = expm(-i H dt).
    2) Dissipation exacte canal par canal : pour L_k = |k><k| (dephasage),
       la solution exacte de d rho/dt = gamma D[n_k] rho est
       rho_ij -> rho_ij * exp(-gamma dt) si exactement l un de {i,j} == k.
    Chaque etape preserve trace, hermiticite et positivite.
    """
    u = expm(-1j * h * dt)
    rho = u @ rho @ u.conj().T
    if gamma > 0.0:
        decay = np.exp(-gamma * dt)
        n = rho.shape[0]
        mask = np.ones((n, n))
        for k in range(n):
            # coherences impliquant le site k (hors diagonale) decroissent
            for i in range(n):
                if i != k:
                    mask[i, k] *= decay
                    mask[k, i] *= decay
        rho = rho * mask
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    return rho


def run_lindblad(gamma: float, out_dir: Path | None = None) -> dict:
    """Propage rho sous rampe + dephasage, mesure P_sig a chaque pas."""
    n_steps = int(T_RAMP / DT)
    times = np.linspace(0.0, T_RAMP, n_steps + 1)
    l_ops = number_operators(N_SITES)

    # etat initial : etat fondamental pur (matrice de correlation trace = N/2)
    orb0 = ground_state_orbitals(ssh_hamiltonian(N_SITES, DELTA_START))
    rho = orb0 @ orb0.conj().T
    n_occ = N_SITES // 2  # pour ramener a la meme echelle que le cas pur

    psig = np.empty(n_steps + 1)
    purity = np.empty(n_steps + 1)
    psig[0] = psig_from_correlation(rho)
    purity[0] = float(np.real(np.trace(rho @ rho)))

    for s in range(1, n_steps + 1):
        t = times[s - 1]
        h = ssh_hamiltonian(N_SITES, delta_of_t(t))
        rho = lindblad_step(rho, h, l_ops, gamma, DT)
        # ramener a la trace = n_occ (echelle du cas pur) pour comparer P_sig
        rho_scaled = rho * n_occ / np.trace(rho)
        psig[s] = psig_from_correlation(rho_scaled)
        purity[s] = float(np.real(np.trace(rho_scaled @ rho_scaled))) / n_occ

    result = {
        "gamma": gamma,
        "psig_initial": float(psig[0]),
        "psig_final": float(psig[-1]),
        "psig_max": float(psig.max()),
        "purity_final": float(purity[-1]),
        "transition_detectee": bool(psig.max() > 0.02),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"lindblad_gamma_{gamma:.4f}.npz",
            times=times, psig=psig, purity=purity, gamma=gamma,
        )
    return result


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    results = []
    for gamma in (0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2):
        r = run_lindblad(gamma, out_dir=out)
        results.append(r)
        print(f"gamma={gamma:8.4f}  P_sig_max={r['psig_max']:.4f}  "
              f"purete_finale={r['purity_final']:.4f}  detectee={r['transition_detectee']}")
    with open(out / "decoherence_results.json", "w") as f:
        json.dump(results, f, indent=2)
