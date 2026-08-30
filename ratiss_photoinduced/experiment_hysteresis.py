"""Rampe aller-retour : hysteresis dynamique de la transition photo-induite.

Protocole :
  - Aller : delta 0.4 -> -0.4 en temps T (traverse la transition).
  - Retour : delta -0.4 -> 0.4 en temps T (revient).
  - On compare P_sig et |C(0,N-1)| de l etat reel a l aller et au retour.

Si l etat ne repasse pas par le meme chemin (P_sig retour != P_sig aller au
meme delta), c est l hysteresis dynamique : le systeme "se souvient" de la
transition. Apparente a Kibble-Zurek : le gel pres du point critique laisse
une trace.
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
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_SITES = 16
DELTA_A = 0.4
DELTA_B = -0.4
DT = 0.1


def delta_roundtrip(t: float, t_leg: float) -> float:
    """Rampe aller-retour : A->B en t_leg, puis B->A en t_leg."""
    if t <= t_leg:
        frac = t / t_leg
        return DELTA_A + (DELTA_B - DELTA_A) * frac
    frac = (t - t_leg) / t_leg
    return DELTA_B + (DELTA_A - DELTA_B) * min(frac, 1.0)


def run(t_leg: float = 200.0, out_dir: Path | None = None) -> dict:
    t_total = 2.0 * t_leg
    n_steps = int(t_total / DT)
    times = np.linspace(0.0, t_total, n_steps + 1)

    h0 = ssh_hamiltonian(N_SITES, DELTA_A)
    orb0 = ground_state_orbitals(h0)
    traj = evolve_orbitals(orb0, lambda t: delta_roundtrip(t, t_leg), DT, n_steps)

    deltas = np.array([delta_roundtrip(t, t_leg) for t in times])
    psig = np.array([psig_from_correlation(correlation_matrix(traj[s])) for s in range(n_steps + 1)])
    edge = np.array([edge_weight(correlation_matrix(traj[s])) for s in range(n_steps + 1)])

    # Reference adiabatique au meme delta (pour comparer aller vs retour)
    psig_ad = np.array([
        psig_from_correlation(correlation_matrix(ground_state_orbitals(ssh_hamiltonian(N_SITES, float(d)))))
        for d in deltas
    ])
    edge_ad = np.array([
        edge_weight(correlation_matrix(ground_state_orbitals(ssh_hamiltonian(N_SITES, float(d)))))
        for d in deltas
    ])

    # Separation aller / retour
    mid = n_steps // 2
    # Aire d hysteresis : integrale de |aller - retour| sur delta commun
    psig_aller, psig_retour = psig[: mid + 1], psig[mid:]
    edge_aller, edge_retour = edge[: mid + 1], edge[mid:]
    d_aller = deltas[: mid + 1]
    d_retour = deltas[mid:]
    # interpolation du retour sur la grille de delta de l aller (delta commun)
    order = np.argsort(d_retour)
    psig_retour_i = np.interp(d_aller, d_retour[order], psig_retour[order])
    edge_retour_i = np.interp(d_aller, d_retour[order], edge_retour[order])
    aire_psig = float(abs(np.trapezoid(np.abs(psig_aller - psig_retour_i), d_aller)))
    aire_edge = float(abs(np.trapezoid(np.abs(edge_aller - edge_retour_i), d_aller)))

    results = {
        "t_leg": t_leg,
        "aire_hysteresis_psig": aire_psig,
        "aire_hysteresis_edge": aire_edge,
        "psig_fin_retour": float(psig[-1]),
        "psig_ad_fin": float(psig_ad[-1]),
        "ecart_final_psig": float(abs(psig[-1] - psig_ad[-1])),
        "ecart_final_edge": float(abs(edge[-1] - edge_ad[-1])),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "roundtrip_trajectory.npz",
            times=times, deltas=deltas, psig=psig, edge=edge,
            psig_ad=psig_ad, edge_ad=edge_ad, t_leg=t_leg,
        )
        with open(out_dir / "roundtrip_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    for t_leg in (200.0, 400.0):
        r = run(t_leg=t_leg, out_dir=out if t_leg == 200.0 else None)
        print(f"T_leg={t_leg:6.1f}  aire_hyst_psig={r['aire_hysteresis_psig']:.5f}  "
              f"aire_hyst_edge={r['aire_hysteresis_edge']:.5f}  "
              f"ecart_final_psig={r['ecart_final_psig']:.5f}")
