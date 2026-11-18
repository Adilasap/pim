"""Train the final PINN deep ensemble on all clean data and validate on the external dataset."""
import _common  # noqa: F401
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from steel_pinn import evaluation, training
from steel_pinn.config import FIGURES, MODELS, PROCESSED, RESULTS, TARGETS
from steel_pinn.inverse_design import ApplicabilityDomain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    tidy = pd.read_csv(PROCESSED / "steel_clean.csv")
    bench = pd.read_csv(PROCESSED / "benchmark_clean.csv")

    ens = training.Ensemble.fit(tidy, training.NNConfig(kind="pinn", physics_loss=True),
                                n_members=args.members)
    ens.save(MODELS / "pinn_ensemble.pt")
    xgb = training.TreeModel("xgb").fit(tidy)

    # External validation: a different source, high-strength (maraging/secondary-hardening)
    # steels with no processing information -> a deliberately hard extrapolation test.
    domain = ApplicabilityDomain(tidy)
    bench["ad_distance"] = domain.distance(bench)
    in_domain = bench.ad_distance <= domain.threshold
    report = {"n_external": int(len(bench)), "n_in_domain": int(in_domain.sum()),
              "ad_threshold": domain.threshold}
    for name, model in (("pinn_ensemble", ens), ("xgb", xgb)):
        p = model.predict(bench)
        report[name] = {"all": evaluation.regression_metrics(bench, p)}
        if in_domain.sum() >= 5:
            report[name]["in_domain"] = evaluation.regression_metrics(bench[in_domain], p[in_domain])
        if name == "pinn_ensemble":
            p_pinn = p
    (RESULTS / "external_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, t in zip(ax, TARGETS):
        a.errorbar(bench[t], p_pinn[t], yerr=p_pinn[t + "_std"], fmt="o", ms=3, alpha=0.5)
        lo, hi = min(bench[t].min(), p_pinn[t].min()), max(bench[t].max(), p_pinn[t].max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_title(f"External (Matminer) {t}")
        a.set_xlabel("measured"); a.set_ylabel("PINN ensemble")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_validation.png", dpi=130)


if __name__ == "__main__":
    main()
