"""Phase 4 : hysteresis thermique et régénération du tissage (KTN:Li).

Protocole :
  - Cycle thermique : le tissage "fond" (chauffage) puis se régénère en un
    nouveau motif (refroidissement). Modélisé comme dissolution progressive
    (interpolation woven -> bruit) puis régénération (seed neuf).
  - On mesure P_sig à chaque étape du cycle, pour plusieurs vitesses de
    refroidissement v_cool.
  - Aire d'hystérésis de P_sig(T) vs v_cool : loi de puissance à mesurer,
    comparée à l'exposant SSH (alpha = 1.05).
  - Invariance statistique : P_sig(motif régénéré) vs P_sig(motif initial)
    sur plusieurs cycles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.topology import rips_persistence
from ratiss_photoinduced.ktn_woven import (
    make_woven,
    distance_matrix,
    subsample,
    thermal_regenerate,
)


def _psig(pts: np.ndarray) -> float:
    return rips_persistence(distance_matrix(pts))["psig"]


def thermal_cycle(v_cool: float, n_steps: int = 20, n_sub: int = 80,
                  seed: int = 42) -> dict:
    """Un cycle chauffage->refroidissement, retourne les P_sig des deux jambes.

    Chauffage : interpolation linéaire du motif tissé vers du bruit uniforme
    (dissolution du tissage). Refroidissement : régénération d'un motif neuf,
    interpolé depuis le bruit à la vitesse v_cool (nombre de pas effectifs).
    """
    rng = np.random.default_rng(seed)
    base = subsample(make_woven(seed=seed), n_sub)
    noise_state = rng.normal(0.0, base.std(axis=0).mean(), base.shape)
    new_motif = subsample(thermal_regenerate(seed=seed + 1000), n_sub)

    # Jambes : dissolution (heat) puis régénération (cool)
    n_heat = max(int(n_steps * 0.5 / v_cool), 4)
    n_cool = max(int(n_steps * 0.5 * v_cool), 4)
    psig_heat = []
    for i in range(n_heat):
        f = i / (n_heat - 1)
        pts = (1 - f) * base + f * noise_state
        psig_heat.append(_psig(pts))
    psig_cool = []
    for i in range(n_cool):
        f = i / (n_cool - 1)
        pts = (1 - f) * noise_state + f * new_motif
        psig_cool.append(_psig(pts))

    # Aire d'hystérésis : écart aller/retour interpolé sur l'échelle commune
    grid = np.linspace(0.0, 1.0, 200)
    a_h = np.interp(grid, np.linspace(0, 1, n_heat), psig_heat)
    a_c = np.interp(grid, np.linspace(0, 1, n_cool), psig_cool)
    area = float(abs(np.trapezoid(np.abs(a_h - a_c), grid)))

    return {
        "v_cool": float(v_cool),
        "area_psig": area,
        "psig_initial": float(psig_heat[0]),
        "psig_regenerated": float(psig_cool[-1]),
        "invariance_rel": float(abs(psig_heat[0] - psig_cool[-1]) / max(psig_heat[0], 1e-12)),
    }


def run(out_dir: Path | None = None) -> dict:
    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for v in speeds:
        r = thermal_cycle(v)
        rows.append(r)
        print(f"v_cool={v:5.2f}  aire={r['area_psig']:.5f}  "
              f"P_sig init={r['psig_initial']:.4f}  regen={r['psig_regenerated']:.4f}  "
              f"invar={r['invariance_rel']:.3f}")

    v = np.array([r["v_cool"] for r in rows])
    a = np.array([r["area_psig"] for r in rows])
    coef = np.polyfit(np.log(v), np.log(a), 1)
    alpha = float(coef[0])
    invariance = float(np.mean([r["invariance_rel"] for r in rows]))

    results = {
        "speeds": v.tolist(),
        "areas": a.tolist(),
        "alpha_exponent": alpha,
        "ssh_reference_alpha": 1.05,
        "mean_regeneration_invariance": invariance,
        "rows": rows,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ktn_phase4_thermal.json", "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "artifacts"
    res = run(out_dir=out)
    print(f"\nExposant KTN mesuré: alpha={res['alpha_exponent']:.3f} (SSH ref: 1.05)")
    print(f"Invariance moyenne de régénération: {res['mean_regeneration_invariance']:.3f}")
