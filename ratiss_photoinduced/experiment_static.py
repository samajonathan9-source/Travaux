"""Sweep statique : P_sig, gap, corrélation bord-bord en fonction de δ.

Valide que P_sig (Vietoris-Rips sur les corrélations) détecte la transition
topologique au bon endroit (δ=0 analytique), et mesure le précurseur de
taille finie : à N fini, la longueur de corrélation diverge avant δ=0, donc
les signaux de corrélation s'activent en avance — c'est le signal d'alerte.
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
    ground_state_orbitals,
    spectral_gap,
    ssh_hamiltonian,
    winding_number,
)
from ratiss_photoinduced.topology import psig_from_correlation


def run(n_sites: int = 16, n_delta: int = 41, out_dir: Path | None = None) -> dict:
    deltas = np.linspace(-0.6, 0.6, n_delta)
    psig = np.empty(n_delta)
    gap = np.empty(n_delta)
    edge = np.empty(n_delta)

    for i, d in enumerate(deltas):
        h = ssh_hamiltonian(n_sites, float(d))
        c = correlation_matrix(ground_state_orbitals(h))
        psig[i] = psig_from_correlation(c)
        gap[i] = spectral_gap(h)
        edge[i] = edge_weight(c)

    # Transition : en partant de la phase triviale (delta>0) vers la topologique,
    # premier delta (en balayage décroissant) où chaque signal s'active.
    def first_active(values: np.ndarray, seuil: float) -> int | None:
        for i in range(n_delta - 1, -1, -1):
            if values[i] > seuil:
                return i
        return None

    i_psig = first_active(psig, 0.02)
    i_edge = first_active(edge, 0.05)
    # Le gap d'une chaîne finie OBC s'effondre dès l'entrée en phase topologique
    # (états de bord) : le marqueur est la chute sous 50 % du gap de départ.
    i_gap = None
    for i in range(n_delta - 1, -1, -1):
        if gap[i] < 0.5 * gap[-1]:
            i_gap = i
            break

    results = {
        "n_sites": n_sites,
        "delta_transition_analytique": 0.0,
        "delta_psig_active": float(deltas[i_psig]) if i_psig is not None else None,
        "delta_gap_chute": float(deltas[i_gap]) if i_gap is not None else None,
        "delta_edge_active": float(deltas[i_edge]) if i_edge is not None else None,
        "precurseur_edge_en_delta": float(0.0 - deltas[i_edge]) if i_edge is not None else None,
        "deltas": deltas.tolist(),
        "psig": psig.tolist(),
        "gap": gap.tolist(),
        "edge": edge.tolist(),
        "winding": [int(round(winding_number(float(d)))) for d in deltas],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "static_sweep.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    res = run(out_dir=out)
    for k in ("delta_psig_active", "delta_gap_chute", "delta_edge_active", "precurseur_edge_en_delta"):
        print(f"{k}: {res[k]}")
