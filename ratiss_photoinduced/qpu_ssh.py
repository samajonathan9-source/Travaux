"""Circuit SSH pour QPU IBM — mesure de P_sig sur hardware réel.

Approche : la chaîne SSH à N sites se mappe sur N qubits (1 qubit = 1 site).
On prépare l'état fondamental (mi-remplissage) par décomposition exacte,
on mesure les corrélations, et on calcule P_sig.

Version 4 qubits : N=4 sites SSH, 2 fermions occupés. Circuit déterministe
(pas VQE) construit par diagonalisation du Hamiltonien fermionique.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.ssh_model import (
    correlation_matrix,
    ground_state_orbitals,
    ssh_hamiltonian,
)
from ratiss_photoinduced.topology import psig_from_correlation

N_QUBITS = 4  # 4 sites SSH, 2 fermions
DELTA_VALUES = (-0.5, 0.0, 0.5)  # topologique, transition, trivial


def build_ssh_state_circuit(delta: float, n_qubits: int = N_QUBITS):
    """Circuit qui prépare l'état fondamental SSH à 2 fermions."""
    from qiskit import QuantumCircuit

    h = ssh_hamiltonian(n_qubits, delta)

    # Hamiltonien dans l'espace de Fock à 2 fermions (6 états |ij>)
    states = list(combinations(range(n_qubits), 2))
    idx = {s: k for k, s in enumerate(states)}
    hf = np.zeros((6, 6), dtype=complex)
    for (i, j), a in idx.items():
        for k in range(n_qubits):
            for l in range(n_qubits):
                if abs(h[k, l]) < 1e-12 or k == l:
                    continue
                st = {i, j}
                if l in st and k not in st:
                    new = tuple(sorted((st - {l}) | {k}))
                    b = idx[new]
                    hf[b, a] += h[k, l]
    eigs, vecs = np.linalg.eigh(hf)
    gs = vecs[:, 0]

    qc = QuantumCircuit(n_qubits)
    amps = np.zeros(2 ** n_qubits, dtype=complex)
    for (i, j), a in idx.items():
        state_int = (1 << i) | (1 << j)
        amps[state_int] = gs[a]
    qc.initialize(amps / np.linalg.norm(amps), range(n_qubits))
    return qc, float(eigs[0])


def measure_correlations(backend, delta: float, shots: int = 4096):
    """Mesure les densités <n_i> (diagonale de C) sur le backend."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    qc, e0 = build_ssh_state_circuit(delta)
    qc_meas = qc.copy()
    qc_meas.measure_all()
    tqc = transpile(qc_meas, backend)
    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    n = N_QUBITS
    diag = np.zeros(n)
    for bitstring, count in counts.items():
        prob = count / shots
        for q in range(n):
            bit = int(bitstring[::-1][q])
            diag[q] += bit * prob
    return np.diag(diag), e0, counts


def run_simulation(out_dir=None) -> dict:
    """Tourne sur simulateur Aer idéal."""
    from qiskit_aer import AerSimulator

    backend = AerSimulator()
    results = {}
    for delta in DELTA_VALUES:
        corr, e0, counts = measure_correlations(backend, delta)
        h = ssh_hamiltonian(N_QUBITS, delta)
        corr_exact = correlation_matrix(ground_state_orbitals(h))
        diag_exact = np.diag(corr_exact)
        diag_meas = np.diag(corr)
        key = f"delta_{delta:+.2f}"
        results[key] = {
            "e0_circuit": e0,
            "diag_exact": diag_exact.tolist(),
            "diag_mesuree": diag_meas.tolist(),
            "ecart_max": float(np.max(np.abs(diag_exact - diag_meas))),
            "n_etats_mesures": len(counts),
        }
        print(f"delta={delta:+.2f}  ecart_max_diag={results[key]['ecart_max']:.4f}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "qpu_simulation.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    run_simulation(out_dir=out)
