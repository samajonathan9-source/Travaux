"""Rampes multi-vitesses : retard non-adiabatique du signal topologique.

Quand le "laser" balaie δ plus vite, l'état quantique ne suit plus
adiabatiquement : le signal P_sig de l'état réel retarde par rapport au
P_sig adiabatique. On mesure ce retard en fonction de la vitesse de rampe.
Physiquement apparenté à Kibble-Zurek : le système "gèle" près de la
transition et rate le basculement instantané.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.ssh_model import (
    correlation_matrix,
    ground_state_orbitals,
    evolve_orbitals,
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_SITES = 16
DELTA_START = 0.4
DELTA_END = -0.4
SEUIL_PSIG = 0.02
DT = 0.1


def run(out_dir: Path | None = None, t_ramps=(50.0, 100.0, 200.0, 400.0)) -> dict:
    results = {"t_ramps": [], "retard_psig": [], "psig_final_evolve": [], "psig_final_adiab": []}
    h0 = ssh_hamiltonian(N_SITES, DELTA_START)
    orb0 = ground_state_orbitals(h0)

    for t_ramp in t_ramps:
        n_steps = int(t_ramp / DT)
        times = np.linspace(0.0, t_ramp, n_steps + 1)

        def delta_of_t(t: float, tr=t_ramp) -> float:
            frac = min(t / tr, 1.0)
            return DELTA_START + (DELTA_END - DELTA_START) * frac

        traj = evolve_orbitals(orb0, delta_of_t, DT, n_steps)
        psig_evol = np.array([psig_from_correlation(correlation_matrix(traj[s])) for s in range(n_steps + 1)])
        psig_adiab = np.array([
            psig_from_correlation(correlation_matrix(ground_state_orbitals(ssh_hamiltonian(N_SITES, delta_of_t(t)))))
            for t in times
        ])

        def first_rise(values: np.ndarray) -> float | None:
            idx = np.nonzero(values >= SEUIL_PSIG)[0]
            return float(times[idx[0]]) if idx.size else None

        t_e = first_rise(psig_evol)
        t_a = first_rise(psig_adiab)
        retard = None if (t_e is None or t_a is None) else t_e - t_a

        results["t_ramps"].append(t_ramp)
        results["retard_psig"].append(retard)
        results["psig_final_evolve"].append(float(psig_evol[-1]))
        results["psig_final_adiab"].append(float(psig_adiab[-1]))
        print(f"T_ramp={t_ramp:6.1f}  retard P_sig={retard}  final evol/ad={psig_evol[-1]:.4f}/{psig_adiab[-1]:.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ramp_speeds.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    run(out_dir=out)
