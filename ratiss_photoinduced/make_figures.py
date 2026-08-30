"""Génère les figures de l'expérience à partir des artefacts JSON/NPZ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "artifacts"


def fig_static():
    data = json.load(open(OUT / "static_sweep.json"))
    d = np.array(data["deltas"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(d, data["psig"], "o-", ms=3, color="tab:purple")
    axes[0].axvline(0, color="k", ls="--", alpha=0.4)
    axes[0].set_ylabel("P_sig (persistance H1)")
    axes[0].set_title("Sweep statique SSH N=%d — transition analytique a delta=0" % data["n_sites"])
    axes[1].plot(d, data["gap"], "s-", ms=3, color="tab:blue")
    axes[1].axvline(0, color="k", ls="--", alpha=0.4)
    axes[1].set_ylabel("Gap spectral")
    axes[2].plot(d, data["edge"], "^-", ms=3, color="tab:red")
    axes[2].axvline(0, color="k", ls="--", alpha=0.4)
    axes[2].set_ylabel("|C(0,N-1)| bord-bord")
    axes[2].set_xlabel("delta (dimerisation)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_static_sweep.png", dpi=150)
    plt.close(fig)


def fig_driven():
    z = np.load(OUT / "driven_trajectory.npz")
    res = json.load(open(OUT / "driven_results.json"))
    t = z["times"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    tc = res["t_transition_hamiltonian"]
    axes[0].plot(t, z["deltas"], color="tab:green")
    axes[0].axvline(tc, color="k", ls="--", alpha=0.5, label="delta=0 (transition)")
    axes[0].axhline(0, color="k", ls=":", alpha=0.3)
    axes[0].set_ylabel("delta(t)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Rampe laser a travers la transition topologique SSH")
    axes[1].plot(t, z["psig_adiab"], label="P_sig adiabatique", color="tab:purple")
    axes[1].plot(t, z["psig_evol"], label="P_sig etat reel", color="tab:orange", ls="--")
    axes[1].axvline(tc, color="k", ls="--", alpha=0.5)
    axes[1].set_ylabel("P_sig")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].plot(t, z["gap_inst"], color="tab:blue")
    axes[2].axvline(tc, color="k", ls="--", alpha=0.5)
    axes[2].set_ylabel("Gap instantane")
    axes[3].plot(t, z["edge_evol"], color="tab:red")
    axes[3].axvline(tc, color="k", ls="--", alpha=0.5)
    te = res.get("t_edge_etat_evolve")
    if te is not None:
        axes[3].axvline(te, color="tab:red", ls=":", alpha=0.7,
                        label="bascule bord-bord (t=%.1f)" % te)
        axes[3].legend(loc="upper left", fontsize=8)
    axes[3].set_ylabel("|C(0,N-1)|")
    axes[3].set_xlabel("temps (hbar/t)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_driven_transition.png", dpi=150)
    plt.close(fig)


def fig_ramps():
    data = json.load(open(OUT / "ramp_speeds.json"))
    fig, ax = plt.subplots(figsize=(7, 5))
    tr = data["t_ramps"]
    retards = [r if r is not None else float("nan") for r in data["retard_psig"]]
    ax.plot(tr, retards, "o-", color="tab:purple")
    ax.axhline(0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("Duree de rampe T (hbar/t)")
    ax.set_ylabel("Retard P_sig reel - adiabatique")
    ax.set_title("Suivi adiabatique du signal topologique vs vitesse de rampe")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ramp_speeds.png", dpi=150)
    plt.close(fig)



def fig_hysteresis():
    """Boucle d'hysteresis P_sig(delta) + aire vs vitesse de rampe."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Boucle (trajectoire rapide T_leg=30 sauvegardee)
    z = np.load(OUT / "roundtrip_trajectory.npz")
    d, psig, edge = z["deltas"], z["psig"], z["edge"]
    t_leg = float(z["t_leg"])
    n = len(d)
    mid = n // 2
    axes[0].plot(d[: mid + 1], psig[: mid + 1], "o-", ms=2, color="tab:blue",
                 label="aller (trivial -> topo)")
    axes[0].plot(d[mid:], psig[mid:], "s-", ms=2, color="tab:red",
                 label="retour (topo -> trivial)")
    axes[0].axvline(0, color="k", ls="--", alpha=0.4, label="delta=0")
    axes[0].set_xlabel("delta (dimerisation)")
    axes[0].set_ylabel("P_sig")
    axes[0].set_title("Boucle d'hysteresis topologique (T_leg=%g)" % t_leg)
    axes[0].legend(fontsize=8)

    # Aire vs vitesse
    sp = json.load(open(OUT / "hysteresis_speeds.json"))
    tr = np.array(sp["t_leg"])
    aire = np.array(sp["aire_psig"])
    axes[1].loglog(1.0 / tr, aire, "o-", color="tab:purple", label="aire P_sig")
    axes[1].set_xlabel("vitesse de rampe 1/T (t / hbar)")
    axes[1].set_ylabel("aire d'hysteresis")
    axes[1].set_title("Croissance de l'hysteresis avec la vitesse")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_hysteresis.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_static()
    fig_driven()
    fig_ramps()
    fig_hysteresis()
    print("figures ecrites dans", OUT)
