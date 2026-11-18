"""Clean both datasets, parse processing text, audit, and write data/processed/."""
import _common  # noqa: F401
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steel_pinn import data, physics
from steel_pinn.config import FIGURES, PROCESSED, RESULTS


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    primary, log_p = data.clean(data.load_primary())
    bench, log_b = data.clean(data.load_benchmark())

    ratio, r_min = primary.YS / primary.UTS, physics.min_yield_ratio(primary.EL.values)
    audit = {
        "primary": log_p, "benchmark": log_b,
        "primary_targets": primary[["YS", "UTS", "EL"]].describe().round(1).to_dict(),
        "primary_processing_flags_mean": primary[data.FLAG_COLS].mean().round(3).to_dict(),
        "primary_missing_flags_mean": primary[data.MISSING_COLS].mean().round(3).to_dict(),
        "hardness_labels_available": False,
        "considere_hollomon_bound_satisfied_by_data": float((ratio >= r_min).mean()),
        "considere_bound_satisfied_with_margin": float((ratio >= r_min - 0.03).mean()),
        "benchmark_composition_overlap_groups": int(len(set(primary.group) & set(bench.group))),
    }
    primary.to_csv(PROCESSED / "steel_clean.csv", index=False)
    bench.to_csv(PROCESSED / "benchmark_clean.csv", index=False)
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ("primary", "benchmark")}, indent=2))
    print("Considere bound satisfied by data:", round(audit["considere_hollomon_bound_satisfied_by_data"], 3))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, t, lab in zip(ax, ["YS", "UTS", "EL"], ["Yield strength (MPa)", "Tensile strength (MPa)",
                                                   "Elongation (%)"]):
        a.hist(primary[t], bins=40, alpha=0.75, label="Birbilis 2025 (train)")
        a.hist(bench[t].dropna(), bins=25, alpha=0.6, label="Matminer (external)")
        a.set_xlabel(lab)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distributions.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    el = np.linspace(1, 50, 200)
    ax.scatter(primary.EL, ratio, s=6, alpha=0.4, label="records")
    ax.plot(el, physics.min_yield_ratio(el), "k-", label="Hollomon + Considère bound")
    ax.set_xlabel("Elongation (%)"); ax.set_ylabel("YS / UTS"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "considere_bound.png", dpi=150)


if __name__ == "__main__":
    main()
