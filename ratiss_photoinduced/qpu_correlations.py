"""Mesure des corrélations croisées C_ij sur QPU pour P_sig complet.

Pour mesurer C_ij = <c+_i c_j> sur hardware, on a besoin des termes :
  Re(C_ij) = (1/2)(<X_i X_j> + <Y_i Y_j>)

Pour SSH à 4 sites, les paires importantes sont :
  - voisins : (0,1), (1,2), (2,3)
  - bord-bord : (0,3) — signature des états de bord en phase topologique

On mesure <X_i X_j> et <Y_i Y_j> pour ces 4 paires.
Total : 8 circuits (4 XX + 4 YY) + 1 circuit Z (densités) = 9 circuits.
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
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_QUBITS = 4
PAIRS = [(0, 1), (1, 2), (2, 3), (0, 3)]  # voisins + bord-bord
DELTA_VALUES = (-0.5, 0.5)  # topologique vs trivial


def build_measurement_circuit(delta: float, pair: tuple[int, int], basis: str):
    """Circuit pour mesurer <basis_i basis_j> sur la paire (i,j).

    basis: 'XX' ou 'YY'
    """
    from qiskit import QuantumCircuit
    from ratiss_photoinduced.qpu_ssh import build_ssh_state_circuit

    qc, e0 = build_ssh_state_circuit(delta, N_QUBITS)
    i, j = pair

    # Rotation vers la base de mesure
    for q in (i, j):
        if basis == 'XX':
            qc.h(q)  # H -> mesure X
        elif basis == 'YY':
            qc.sdg(q)  # S+ -> mesure Y
            qc.h(q)    # H -> mesure X (après S+)

    qc.measure_all()
    return qc


def measure_pair_correlation(backend, delta: float, pair: tuple[int, int], basis: str, shots: int = 4096):
    """Mesure <basis_i basis_j> sur le QPU."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    qc = build_measurement_circuit(delta, pair, basis)
    tqc = transpile(qc, backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    # Calcule <basis_i basis_j> à partir des counts
    i, j = pair
    expectation = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        bit_i = int(bitstring[::-1][i])
        bit_j = int(bitstring[::-1][j])
        # eigenvalue : +1 si bits égaux, -1 si différents
        eigenvalue = 1.0 if bit_i == bit_j else -1.0
        expectation += eigenvalue * prob
    return expectation


def reconstruct_correlation_matrix(backend, delta: float, shots: int = 4096):
    """Reconstruit la matrice C_ij complète à partir des mesures QPU."""
    # Diagonale (densités) — circuit Z
    from ratiss_photoinduced.qpu_ssh import measure_correlations
    corr_diag, e0, _ = measure_correlations(backend, delta, shots)
    C = np.zeros((N_QUBITS, N_QUBITS), dtype=complex)
    C[np.diag_indices(N_QUBITS)] = np.diag(corr_diag)

    # Hors-diagonale (cohérences)
    for pair in PAIRS:
        i, j = pair
        xx = measure_pair_correlation(backend, delta, pair, 'XX', shots)
        yy = measure_pair_correlation(backend, delta, pair, 'YY', shots)
        # Re(C_ij) = (1/4)(<XX> + <YY>) pour fermions sans interaction
        # (facteur 1/4 de la transformation Jordan-Wigner)
        re_c = 0.25 * (xx + yy)
        C[i, j] = re_c
        C[j, i] = re_c  # symétrique

    return C, e0


def run_qpu_psig(backend, out_dir: Path | None = None, shots: int = 4096):
    """Mesure P_sig sur QPU pour delta = -0.5 (topo) et +0.5 (trivial)."""
    results = {}
    for delta in DELTA_VALUES:
        print(f"\nMesure QPU delta={delta:+.2f}...")
        C_qpu, e0 = reconstruct_correlation_matrix(backend, delta, shots)
        psig_qpu = psig_from_correlation(C_qpu)

        # Référence exacte
        h = ssh_hamiltonian(N_QUBITS, delta)
        C_exact = correlation_matrix(ground_state_orbitals(h))
        psig_exact = psig_from_correlation(C_exact)

        phase = "TOPOLOGIQUE" if delta < 0 else "TRIVIALE"
        results[f"delta_{delta:+.2f}"] = {
            "phase": phase,
            "e0": e0,
            "C_qpu": [[float(np.real(c)) for c in row] for row in C_qpu],
            "C_exact": [[float(np.real(c)) for c in row] for row in C_exact],
            "psig_qpu": float(psig_qpu),
            "psig_exact": float(psig_exact),
            "ecart_psig": float(abs(psig_qpu - psig_exact)),
        }
        print(f"  P_sig QPU   : {psig_qpu:.4f}")
        print(f"  P_sig exact : {psig_exact:.4f}")
        print(f"  ecart       : {abs(psig_qpu - psig_exact):.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "qpu_psig_results.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    # Test sur simulateur d'abord
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    print("=== TEST SUR SIMULATEUR ===")
    run_qpu_psig(backend, out_dir=out)
