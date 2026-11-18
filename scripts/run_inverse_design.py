"""Inverse design with NSGA-II on the trained PINN ensemble.

Example:
  python scripts/run_inverse_design.py --ys 800 --uts 1000 --hv 300 --el 12
"""
import _common  # noqa: F401
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from steel_pinn import training
from steel_pinn.config import FIGURES, MODELS, PROCESSED, RESULTS
from steel_pinn.inverse_design import Targets, run_nsga2, select_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ys", type=float, default=800)
    ap.add_argument("--uts", type=float, default=1000)
    ap.add_argument("--hv", type=float, default=300)
    ap.add_argument("--el", type=float, default=12)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--n-candidates", type=int, default=5)
    args = ap.parse_args()

    tidy = pd.read_csv(PROCESSED / "steel_clean.csv")
    model = training.Ensemble.load(MODELS / "pinn_ensemble.pt")
    targets = Targets(YS=args.ys, UTS=args.uts, HV=args.hv, EL=args.el)
    cand, bounds, domain, _ = run_nsga2(model, tidy, targets, pop_size=args.pop, n_gen=args.gens)
    short = select_candidates(cand, targets, bounds, n=args.n_candidates)

    cand.to_csv(RESULTS / "nsga2_population.csv", index=False)
    short.round(4).to_csv(RESULTS / "candidates.csv", index=False)
    (RESULTS / "inverse_design_config.json").write_text(json.dumps({
        "targets": vars(targets), "bounds": bounds.round(4).to_dict(orient="index"),
        "n_route_records": bounds.attrs.get("n_route_records"),
        "n_feasible": int(cand.feasible.sum()), "population": int(len(cand)),
        "ad_threshold": domain.threshold}, indent=2))

    show = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "aust_temp", "temper_temp",
            "YS", "YS_std", "UTS", "UTS_std", "HV", "EL", "EL_std", "ad_distance", "score"]
    print(f"feasible: {int(cand.feasible.sum())}/{len(cand)}")
    print(short[[c for c in show if c in short]].round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    f = cand[cand.feasible]
    sc = ax.scatter(f.EL, f.UTS, c=f.UTS_std, cmap="viridis", s=18)
    ax.scatter(cand[~cand.feasible].EL, cand[~cand.feasible].UTS, c="lightgrey", s=10,
               label="infeasible")
    ax.scatter(short.EL, short.UTS, facecolors="none", edgecolors="red", s=120, lw=1.5,
               label="shortlist")
    ax.axhline(args.uts, ls="--", c="k", lw=0.8); ax.axvline(args.el, ls="--", c="k", lw=0.8)
    plt.colorbar(sc, label="UTS ensemble std (MPa)")
    ax.set_xlabel("Predicted elongation (%)"); ax.set_ylabel("Predicted UTS (MPa)")
    ax.set_title("NSGA-II final population (oil quench + temper)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto_front.png", dpi=140)


if __name__ == "__main__":
    main()
