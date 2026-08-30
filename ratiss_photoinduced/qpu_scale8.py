"""Scale à 8 qubits pour QPU — contraste plus net avec plus de sites.

N=8 sites SSH, 4 fermions. L'espace de Fock a C(8,4)=70 états.
Le circuit est plus profond mais le contraste topologique devrait être plus net
que N=4 (edge plus grand, P_sig plus net).
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
from ratiss_photoinduced.robust_metrics import coupled_metric

N_QUBITS = 8
N_OCC = 4  # mi-remplissage


def build_ssh_8(delta: float):
    """Circuit SSH à 8 sites, 4 fermions."""
    from qiskit import QuantumCircuit
    from ratiss_photoinduced.ssh_model import ssh_hamiltonian

    h = ssh_hamiltonian(N_QUBITS, delta)
    states = list(combinations(range(N_QUBITS), N_OCC))
    idx = {s: k for k, s in enumerate(states)}
    hf = np.zeros((70, 70), dtype=complex)
    for st_tuple, a in idx.items():
        st = set(st_tuple)
        for k in range(N_QUBITS):
            for l in range(N_QUBITS):
                if abs(h[k, l]) < 1e-12 or k == l:
                    continue
                if l in st and k not in st:
                    new = tuple(sorted((st - {l}) | {k}))
                    b = idx[new]
                    hf[b, a] += h[k, l]
    eigs, vecs = np.linalg.eigh(hf)
    gs = vecs[:, 0]

    qc = QuantumCircuit(N_QUBITS)
    amps = np.zeros(2 ** N_QUBITS, dtype=complex)
    for st_tuple, a in idx.items():
        state_int = sum(1 << i for i in st_tuple)
        amps[state_int] = gs[a]
    qc.initialize(amps / np.linalg.norm(amps), range(N_QUBITS))
    return qc, float(eigs[0])


def measure_correlations_8(backend, delta: float, shots: int = 4096):
    """Mesure les densités sur 8 qubits."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    qc, e0 = build_ssh_8(delta)
    qc_meas = qc.copy()
    qc_meas.measure_all()
    tqc = transpile(qc_meas, backend)
    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    diag = np.zeros(N_QUBITS)
    for bitstring, count in counts.items():
        prob = count / shots
        for q in range(N_QUBITS):
            bit = int(bitstring[::-1][q])
            diag[q] += bit * prob
    return np.diag(diag), e0, counts


def run_scale_8(backend, out_dir: Path | None = None, shots: int = 4096):
    """Mesure le score robuste sur 8 qubits pour delta = -0.5 et +0.5."""
    results = {}
    for delta in (-0.5, 0.5):
        print(f"\nMesure 8 qubits delta={delta:+.2f}...")
        C_qpu, e0, _ = measure_correlations_8(backend, delta, shots)
        # Pour le score robuste, on utilise la diagonale (edge_weight sur diag)
        m = coupled_metric(C_qpu)
        phase = "TOPOLOGIQUE" if delta < 0 else "TRIVIALE"
        results[f"delta_{delta:+.2f}"] = {
            "phase": phase,
            "e0": e0,
            "score": m["score"],
            "edge": m["edge"],
            "psig": m["psig_seuille"],
        }
        print(f"  score robuste: {m['score']:.4f}")
        print(f"  edge: {m['edge']:.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "qpu_scale8.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    # Test sur simulateur
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    print("=== TEST SUR SIMULATEUR ===")
    run_scale_8(backend, out_dir=out)
