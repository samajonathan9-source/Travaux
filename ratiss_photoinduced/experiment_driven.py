"""Expérience principale : rampe laser δ(t) à travers la transition topologique.

Protocole :
  - Chaîne SSH N=16, état initial = état fondamental à δ_start (phase triviale).
  - Rampe linéaire δ(t) : δ_start -> δ_end en temps T (traverse δ=0).
  - Propagation temps réel RK4 (l'état réel, qui peut retarder sur le Hamiltonien).
  - À chaque pas : P_sig(état évolué), P_sig(état fondamental instantané = référence
    adiabatique), gap instantané, corrélation bord-bord.
  - Mesure : instants de franchissement de seuil de chaque signal, comparés à
    l'instant t_c où δ(t_c) = 0 (transition du Hamiltonien).

Question de recherche : le signal topologique P_sig de l'état bascule-t-il
AVANT ou APRÈS les signaux classiques, et avec quel retard par rapport à t_c ?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.ssh_model import (
    correlation_matrix,
    edge_weight,
    evolve_orbitals,
    ground_state_orbitals,
    spectral_gap,
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_SITES = 16
DELTA_START = 0.4      # phase triviale
DELTA_END = -0.4       # phase topologique
T_RAMP = 200.0         # durée de la rampe (unités ħ/t)
DT = 0.1
SEUIL_PSIG = 0.02      # seuil de bascule P_sig (calibré sur le sweep statique)
SEUIL_EDGE = 0.05      # seuil de bascule corrélation bord-bord


def delta_of_t(t: float) -> float:
    frac = min(t / T_RAMP, 1.0)
    return DELTA_START + (DELTA_END - DELTA_START) * frac


def crossing_time(times: np.ndarray, values: np.ndarray, threshold: float, direction: str) -> float | None:
    """Premier instant où le signal franchit le seuil dans la direction voulue."""
    if direction == "rise":
        idx = np.nonzero(values >= threshold)[0]
    else:
        idx = np.nonzero(values <= threshold)[0]
    return float(times[idx[0]]) if idx.size else None


def run(out_dir: Path) -> dict:
    n_steps = int(T_RAMP / DT)
    times = np.linspace(0.0, T_RAMP, n_steps + 1)

    h0 = ssh_hamiltonian(N_SITES, DELTA_START)
    orb0 = ground_state_orbitals(h0)
    traj = evolve_orbitals(orb0, delta_of_t, DT, n_steps)

    psig_evol = np.empty(n_steps + 1)
    psig_adiab = np.empty(n_steps + 1)
    gap_inst = np.empty(n_steps + 1)
    edge_evol = np.empty(n_steps + 1)
    deltas = np.array([delta_of_t(t) for t in times])

    for s in range(n_steps + 1):
        h_t = ssh_hamiltonian(N_SITES, deltas[s])
        gap_inst[s] = spectral_gap(h_t)
        psig_adiab[s] = psig_from_correlation(correlation_matrix(ground_state_orbitals(h_t)))
        c_t = correlation_matrix(traj[s])
        psig_evol[s] = psig_from_correlation(c_t)
        edge_evol[s] = edge_weight(c_t)

    # Instants de bascule de chaque signal
    t_hamiltonian = T_RAMP / 2  # δ(t_c) = 0 par construction (rampe symétrique)
    t_psig_adiab = crossing_time(times, psig_adiab, SEUIL_PSIG, "rise")
    t_psig_evol = crossing_time(times, psig_evol, SEUIL_PSIG, "rise")
    t_edge_evol = crossing_time(times, edge_evol, SEUIL_EDGE, "rise")
    gap_min_t = float(times[np.argmin(gap_inst)])

    results = {
        "params": {
            "n_sites": N_SITES, "delta_start": DELTA_START, "delta_end": DELTA_END,
            "t_ramp": T_RAMP, "dt": DT, "seuil_psig": SEUIL_PSIG, "seuil_edge": SEUIL_EDGE,
        },
        "t_transition_hamiltonian": t_hamiltonian,
        "t_gap_minimum": gap_min_t,
        "t_psig_adiabatique": t_psig_adiab,
        "t_psig_etat_evolve": t_psig_evol,
        "t_edge_etat_evolve": t_edge_evol,
        "retard_psig_vs_hamiltonien": None if t_psig_evol is None else t_psig_evol - t_hamiltonian,
        "retard_edge_vs_hamiltonien": None if t_edge_evol is None else t_edge_evol - t_hamiltonian,
        "avance_psig_adiab_vs_hamiltonien": None if t_psig_adiab is None else t_psig_adiab - t_hamiltonian,
        "psig_final": float(psig_evol[-1]),
        "psig_adiab_final": float(psig_adiab[-1]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "driven_trajectory.npz",
        times=times, deltas=deltas, psig_evol=psig_evol, psig_adiab=psig_adiab,
        gap_inst=gap_inst, edge_evol=edge_evol,
    )
    with open(out_dir / "driven_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    res = run(out)
    print(json.dumps(res, indent=2, ensure_ascii=False))
