"""Composition-grouped 5-fold CV of all models: accuracy + physical consistency.

Models
  rf          Random forest on engineered + physics-descriptor features
  xgb         XGBoost on the same features
  mlp         Plain multi-output MLP (raw composition + processing, no physics)
  hybrid_nn   Physics-descriptor layer + hard constraints (UTS >= YS), no physics loss
  pinn        hybrid_nn + physics-informed loss (monotonicity, Considere, Tabor)
"""
import _common  # noqa: F401
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from steel_pinn import evaluation, training
from steel_pinn.config import FIGURES, PROCESSED, RESULTS, TARGET_LABELS, TARGETS

MODELS = {
    "rf": lambda k: training.TreeModel("rf", seed=k),
    "xgb": lambda k: training.TreeModel("xgb", seed=k),
    "mlp": lambda k: training.NNModel(training.NNConfig(kind="mlp", seed=k)),
    "hybrid_nn": lambda k: training.NNModel(training.NNConfig(kind="pinn", physics_loss=False, seed=k)),
    "pinn": lambda k: training.NNModel(training.NNConfig(kind="pinn", physics_loss=True, seed=k)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tidy = pd.read_csv(PROCESSED / "steel_clean.csv")
    summary, all_folds, preds = {}, {}, {}
    for name in args.models:
        t0 = time.time()
        folds, p = evaluation.grouped_cv(MODELS[name], tidy, n_splits=args.folds)
        summary[name] = evaluation.summarise_folds(folds)
        summary[name]["seconds"] = round(time.time() - t0, 1)
        all_folds[name], preds[name] = folds, p
        s = summary[name]
        print(f"{name:10s} R2 YS={s['YS_R2']:.3f} UTS={s['UTS_R2']:.3f} EL={s['EL_R2']:.3f} | "
              f"viol in-dist={s['viol_mean_violation']:.4f} OOD={s['oodviol_mean_violation']:.4f} "
              f"({s['seconds']}s)", flush=True)

    df = pd.DataFrame(summary).T
    df.to_csv(RESULTS / "cv_summary.csv")
    (RESULTS / "cv_folds.json").write_text(json.dumps(all_folds, indent=2))

    # parity plots for every model
    fig, axes = plt.subplots(len(preds), 3, figsize=(12, 3.6 * len(preds)), squeeze=False)
    for r, (name, p) in enumerate(preds.items()):
        for c, t in enumerate(TARGETS):
            a = axes[r, c]
            a.scatter(tidy[t], p[t], s=5, alpha=0.4)
            lo, hi = tidy[t].min(), tidy[t].max()
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set_title(f"{name}: {t}  R²={summary[name][t + '_R2']:.3f}", fontsize=10)
            a.set_xlabel("measured " + TARGET_LABELS[t]); a.set_ylabel("predicted")
    fig.tight_layout()
    fig.savefig(FIGURES / "parity_grouped_cv.png", dpi=120)

    # consistency comparison
    cols = [c for c in df.columns if c.startswith("oodviol_") and c != "oodviol_mean_violation"]
    ax = (df[cols] * 100).rename(columns=lambda c: c.replace("oodviol_", "")).T.plot.bar(figsize=(11, 4))
    ax.set_ylabel("% of off-distribution probes violating rule")
    ax.set_title("Physical-consistency violations (lower is better)")
    plt.tight_layout()
    plt.savefig(FIGURES / "physics_violations_ood.png", dpi=130)


if __name__ == "__main__":
    main()
