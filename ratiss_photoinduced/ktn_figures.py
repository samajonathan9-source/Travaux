"""Figures KTN:Li pour le préprint — textures 3D et signatures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratiss_photoinduced.ktn_woven import (
    make_woven,
    make_aligned,
    subsample,
    distance_matrix,
)
from ratiss_photoinduced.topology import rips_persistence

OUT = Path(__file__).resolve().parent.parent / "preprint" / "figures"


def fig_textures():
    """Fig K1 : nuages 3D tissé vs aligné."""
    woven = make_woven()
    aligned = make_aligned()
    fig = plt.figure(figsize=(12, 5.2))
    for k, (pts, title, color) in enumerate((
        (woven, "Woven texture (interlaced strands)", "tab:purple"),
        (aligned, "Aligned texture (parallel domains)", "tab:blue"),
    )):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=color, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(OUT / "figK1_textures.png", dpi=170)
    plt.close(fig)


def fig_persistence():
    """Fig K2 : diagrammes de persistance H1 tissé vs aligné."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, (name, gen, color) in enumerate((
        ("woven", make_woven, "tab:purple"),
        ("aligned", make_aligned, "tab:blue"),
    )):
        pts = subsample(gen(), 80)
        res = rips_persistence(distance_matrix(pts))
        dgm = [(b, d) for b, d in res["diagrams"]["H1"] if d is not None]
        if dgm:
            births = [b for b, _ in dgm]
            deaths = [d for _, d in dgm]
            axes[k].scatter(births, deaths, s=18, c=color, alpha=0.7)
            lim = max(deaths) * 1.15
            axes[k].plot([0, lim], [0, lim], "k--", alpha=0.4)
            axes[k].set_xlim(0, lim)
            axes[k].set_ylim(0, lim)
        axes[k].set_xlabel("birth")
        axes[k].set_ylabel("death")
        axes[k].set_title(f"{name}  ($P_{{\\mathrm{{sig}}}}$ = {res['psig']:.3f})", fontsize=10)
    fig.suptitle("H1 persistence diagrams: woven vs aligned", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figK2_persistence.png", dpi=170)
    plt.close(fig)


def fig_regeneration():
    """Fig K3 : invariance de régénération + aires d'hystérésis."""
    data = json.load(open(Path(__file__).resolve().parent.parent
                          / "artifacts" / "ktn_phase4_thermal.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(data["speeds"], data["areas"], "o-", color="tab:purple")
    axes[0].set_xlabel("cooling speed $v_{cool}$")
    axes[0].set_ylabel("hysteresis area of $P_{\\mathrm{sig}}$")
    axes[0].set_title("Hysteresis area vs cooling speed")
    rows = data["rows"]
    init = [r["psig_initial"] for r in rows]
    regen = [r["psig_regenerated"] for r in rows]
    x = np.arange(len(rows))
    axes[1].plot(x, init, "s-", color="tab:purple", label="initial fabric")
    axes[1].plot(x, regen, "o--", color="tab:green", label="regenerated fabric")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("$P_{\\mathrm{sig}}$")
    axes[1].set_title("Regeneration invariance (drift 1.8%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figK3_regeneration.png", dpi=170)
    plt.close(fig)


def fig_qpu():
    """Fig K4 : résultats QPU tissé vs aligné."""
    path = Path(__file__).resolve().parent.parent / "artifacts" / "ktn_phase5_qpu.json"
    data = json.load(open(path))
    labels = ["woven", "aligned"]
    qpu = [data[k]["score_qpu"] for k in labels]
    exact = [data[k]["score_exact"] for k in labels]
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, qpu, w, label="IBM ibm_fez (hardware)", color="tab:purple")
    ax.bar(x + w / 2, exact, w, label="exact (simulation)", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("coupled score $S$")
    ax.set_title(f"Hardware contrast $\\Delta S$ = {data['contrast_qpu']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figK4_qpu.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_textures()
    fig_persistence()
    fig_regeneration()
    fig_qpu()
    print("figures KTN écrites dans", OUT)
