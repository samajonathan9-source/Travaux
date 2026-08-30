"""Métriques robustes pour P_sig sur hardware bruité.

Stratégies :
1. Seuillage : ignorer les corrélations < seuil (élimine le bruit faible)
2. Entropie de corrélation : alternative lisse à P_sig
3. Couplage : vote P_sig + corrélation bord-bord
4. Itération : moyenner sur deltas voisins (réduit le bruit)

Objectif : trouver une métrique où le contraste topologique survit au bruit QPU.
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
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_QUBITS = 4


def psig_thresholded(corr: np.ndarray, seuil: float = 0.05) -> float:
    """P_sig après seuillage : met à zéro les corrélations < seuil."""
    c = corr.copy()
    c[np.abs(c) < seuil] = 0.0
    return psig_from_correlation(c)


def correlation_entropy(corr: np.ndarray) -> float:
    """Entropie de la distribution des corrélations (alternative à P_sig).

    H = -sum(|C_ij| * log(|C_ij|)) sur les corrélations non nulles.
    Plus H est bas, plus la corrélation est structurée (topologique).
    """
    c = np.abs(corr.flatten())
    c = c[c > 1e-12]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def coupled_metric(corr: np.ndarray, seuil: float = 0.05) -> dict:
    """Vote combiné P_sig seuillé + corrélation bord-bord + entropie."""
    psig = psig_thresholded(corr, seuil)
    edge = edge_weight(corr)
    entropy = correlation_entropy(corr)
    return {
        "psig_seuille": psig,
        "edge": edge,
        "entropie": entropy,
        "score": psig + edge - 0.1 * entropy,
    }


def test_on_qpu_noise(delta: float, noise_level: float = 0.1, n_trials: int = 20, seuil: float = 0.05) -> dict:
    """Teste la métrique sur matrice bruitée (simule QPU)."""
    h = ssh_hamiltonian(N_QUBITS, delta)
    c_exact = correlation_matrix(ground_state_orbitals(h))

    scores = []
    np.random.seed(42)
    for _ in range(n_trials):
        c_noisy = c_exact + np.random.randn(*c_exact.shape) * noise_level
        m = coupled_metric(c_noisy, seuil)
        scores.append(m["score"])

    return {
        "score_moyen": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }


def evaluate_robustness(out_dir: Path | None = None, noise_level: float = 0.1):
    """Évalue la robustesse de la métrique couplée sur deltas variés."""
    deltas = np.linspace(-0.8, 0.8, 9)
    results = []

    for d in deltas:
        h = ssh_hamiltonian(N_QUBITS, float(d))
        c = correlation_matrix(ground_state_orbitals(h))
        exact = coupled_metric(c)
        noisy = test_on_qpu_noise(float(d), noise_level=noise_level)

        phase = "topologique" if d < 0 else ("transition" if abs(d) < 0.05 else "triviale")
        results.append({
            "delta": float(d),
            "phase": phase,
            "exact_score": exact["score"],
            "bruit_score": noisy["score_moyen"],
            "bruit_std": noisy["score_std"],
        })

    topo = [r for r in results if r["phase"] == "topologique"]
    triv = [r for r in results if r["phase"] == "triviale"]

    robustesse = {
        "topo_moyen": float(np.mean([r["bruit_score"] for r in topo])),
        "triv_moyen": float(np.mean([r["bruit_score"] for r in triv])),
        "contraste": float(np.mean([r["bruit_score"] for r in topo]) - np.mean([r["bruit_score"] for r in triv])),
        "contrast_positif": bool(np.mean([r["bruit_score"] for r in topo]) > np.mean([r["bruit_score"] for r in triv])),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "robustness_metrics.json", "w") as f:
            json.dump({"results": results, "robustesse": robustesse}, f, indent=2)

    return robustesse


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    for noise in (0.05, 0.1, 0.2):
        rob = evaluate_robustness(out_dir=out if noise == 0.1 else None, noise_level=noise)
        print(f"\nBruit={noise:.2f}: contraste={rob['contraste']:.4f}, positif={rob['contrast_positif']}")
        print(f"  topo={rob['topo_moyen']:.4f}, triv={rob['triv_moyen']:.4f}")
