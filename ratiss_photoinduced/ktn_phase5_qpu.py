"""Phase 5 : validation QPU de la signature du tissage (KTN:Li) sur IBM.

On encode deux états à 4 qubits :
  - "tissé" (woven) : superposition délocalisée de configurations à 2
    fermions, mimant les brins entrelacés (corrélations croisées fortes).
  - "aligné" : état localisé (domaines parallèles), corrélations croisées
    faibles.

On mesure la diagonale + les corrélations XX/YY groupées, on reconstruit la
matrice C, et on applique la métrique couplée robuste. Si le contraste est
inversé par le bruit, on itère : variantes de circuits auxiliaires (seuillage
plus fort, paires réduites, post-sélection sur le bon secteur de nombre de
particules).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.robust_metrics import coupled_metric

N_QUBITS = 4
PAIRS = [(0, 1), (1, 2), (2, 3), (0, 3)]


def build_woven_state(n_qubits: int = N_QUBITS):
    """État "tissé" : superposition uniforme des 6 configurations à 2 fermions."""
    from qiskit import QuantumCircuit
    from itertools import combinations
    qc = QuantumCircuit(n_qubits)
    amps = np.zeros(2 ** n_qubits, dtype=complex)
    for pair in combinations(range(n_qubits), 2):
        amps[(1 << pair[0]) | (1 << pair[1])] = 1.0
    qc.initialize(amps / np.linalg.norm(amps), range(n_qubits))
    return qc


def build_aligned_state(n_qubits: int = N_QUBITS):
    """État "aligné" : |0011> (deux fermions adjacents, pas d'entrelacement)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    qc.x(1)
    return qc


def _counts_to_expectation(counts: dict, i: int, j: int, shots: int) -> float:
    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        bi = int(bitstring[::-1][i])
        bj = int(bitstring[::-1][j])
        exp_val += (1.0 if bi == bj else -1.0) * prob
    return exp_val


def reconstruct_C(backend, which: str, shots: int = 4096,
                  postselect: bool = False) -> np.ndarray:
    """Mesure et reconstruit la matrice de corrélation d'un état tissé/aligné."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    base = build_woven_state() if which == "woven" else build_aligned_state()
    sampler = Sampler(mode=backend)

    pubs = []
    for basis in ("Z", "XX", "YY"):
        qc = base.copy()
        if basis == "XX":
            for q in range(N_QUBITS):
                qc.h(q)
        elif basis == "YY":
            for q in range(N_QUBITS):
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        pubs.append(transpile(qc, backend, optimization_level=1))

    job = sampler.run(pubs, shots=shots)
    job_id = job.job_id()
    result = job.result()
    counts = [result[i].data.meas.get_counts() for i in range(3)]

    # Itération robuste : post-sélection sur le secteur à 2 particules,
    # uniquement sur la base Z (les rotations H des bases XX/YY changent le
    # nombre de particules mesuré — filtrer celles-ci biaiserait les moments)
    if postselect:
        counts[0] = {b: n for b, n in counts[0].items() if b.count("1") == 2}

    C = np.zeros((N_QUBITS, N_QUBITS), dtype=complex)
    # diagonale (densités)
    tot = sum(counts[0].values())
    diag = np.zeros(N_QUBITS)
    for bitstring, n in counts[0].items():
        p = n / tot
        for q in range(N_QUBITS):
            diag[q] += int(bitstring[::-1][q]) * p
    C[np.diag_indices(N_QUBITS)] = diag
    # hors-diagonale : C_ij = (1/4)(<XiXj> + <YiYj>)
    tot_xx = sum(counts[1].values())
    tot_yy = sum(counts[2].values())
    for (i, j) in PAIRS:
        xx = _counts_to_expectation(counts[1], i, j, tot_xx)
        yy = _counts_to_expectation(counts[2], i, j, tot_yy)
        re_c = 0.25 * (xx + yy)
        C[i, j] = re_c
        C[j, i] = re_c
    return C, job_id


def exact_reference(which: str) -> np.ndarray:
    """Matrice C exacte de l'état (validation simulation/hardware)."""
    if which == "woven":
        # Superposition uniforme des 6 configurations à 2 fermions :
        # diagonale = 3/6 = 1/2 (chaque site dans 3 paires),
        # hors-diagonale = 2/6 = 1/3 (2 paires contiennent j sans i).
        C = np.zeros((N_QUBITS, N_QUBITS), dtype=complex)
        C[np.diag_indices(N_QUBITS)] = 0.5
        for (i, j) in PAIRS:
            C[i, j] = 1.0 / 3.0
            C[j, i] = 1.0 / 3.0
        return C
    C = np.zeros((N_QUBITS, N_QUBITS))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    return C


def run(backend=None, out_dir: Path | None = None, shots: int = 4096,
        postselect: bool = True) -> dict:
    results = {}
    for which in ("woven", "aligned"):
        print(f"\n=== état {which} ===", flush=True)
        C_qpu, job_id = reconstruct_C(backend, which, shots, postselect=postselect)
        m_qpu = coupled_metric(C_qpu)
        C_ex = exact_reference(which)
        m_ex = coupled_metric(C_ex)
        results[which] = {
            "job_id": job_id,
            "score_qpu": m_qpu["score"],
            "edge_qpu": m_qpu["edge"],
            "psig_qpu": m_qpu["psig_seuille"],
            "score_exact": m_ex["score"],
            "edge_exact": m_ex["edge"],
            "psig_exact": m_ex["psig_seuille"],
        }
        print(f"  QPU   : score={m_qpu['score']:.4f} edge={m_qpu['edge']:.4f}", flush=True)
        print(f"  exact : score={m_ex['score']:.4f} edge={m_ex['edge']:.4f}", flush=True)

    results["contrast_qpu"] = results["woven"]["score_qpu"] - results["aligned"]["score_qpu"]
    results["contrast_exact"] = results["woven"]["score_exact"] - results["aligned"]["score_exact"]
    print(f"\nContraste QPU   : {results['contrast_qpu']:.4f}", flush=True)
    print(f"Contraste exact : {results['contrast_exact']:.4f}", flush=True)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ktn_phase5_qpu.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    from qiskit_aer import AerSimulator
    print("=== VALIDATION SIMULATEUR ===")
    run(backend=AerSimulator(), out_dir=out)
