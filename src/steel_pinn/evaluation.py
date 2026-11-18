"""Accuracy metrics, grouped cross-validation and model-agnostic physical-consistency probes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from . import physics
from .config import TARGETS
from .losses import CONSIDERE_MARGIN


def regression_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, targets=TARGETS) -> dict:
    out = {}
    for t in targets:
        m = y_true[t].notna().values
        if m.sum() < 2:
            continue
        yt, yp = y_true[t].values[m], y_pred[t].values[m]
        out[t] = {"R2": float(r2_score(yt, yp)), "MAE": float(mean_absolute_error(yt, yp)),
                  "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "n": int(m.sum())}
    return out


# Finite-difference probes: (output, column, step, required sign, mask column)
PROBES = [
    ("YS", "temper_temp", 50.0, -1, "is_tempered"),
    ("UTS", "temper_temp", 50.0, -1, "is_tempered"),
    ("EL", "temper_temp", 50.0, +1, "is_tempered"),
    ("YS", "C", 0.05, +1, None),
    ("UTS", "C", 0.05, +1, None),
    ("EL", "C", 0.05, -1, None),
    ("YS", "Mn", 0.2, +1, None),
    ("YS", "Si", 0.2, +1, None),
    ("UTS", "section_mm", 25.0, -1, "is_quenched"),
]
TOL = {"YS": 2.0, "UTS": 2.0, "EL": 0.2}       # changes below this are treated as "flat"


def physical_consistency(model, tidy: pd.DataFrame) -> dict:
    """Fraction of records whose predictions violate each physical rule (lower is better)."""
    base = model.predict(tidy)
    res = {"UTS<YS": float((base.UTS < base.YS - 1e-6).mean()),
           "EL<=0": float((base.EL <= 0).mean())}
    r_min = physics.min_yield_ratio(np.clip(base.EL.values, 0, None))
    res["considere"] = float((base.YS.values / base.UTS.values < r_min - CONSIDERE_MARGIN).mean())
    for out, col, step, sign, mask in PROBES:
        sub = tidy if mask is None else tidy[tidy[mask] == 1]
        if len(sub) == 0:
            continue
        bumped = sub.copy()
        bumped[col] = bumped[col] + step
        if col == "temper_temp":
            bumped[col] = bumped[col].clip(upper=760.0)
        delta = model.predict(bumped)[out].values - base.loc[sub.index, out].values
        res[f"d{out}/d{col}"] = float((sign * delta < -TOL[out]).mean())
    res["mean_violation"] = float(np.mean([v for v in res.values()]))
    return res


def perturbed_probe_set(tidy: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Off-distribution probe points (perturbed real records) - where an optimiser will search."""
    import torch

    from .models import BASE_COLS
    from .training import sample_collocation
    x = torch.tensor(tidy[BASE_COLS].values.astype(np.float32))
    z = sample_collocation(x, n, torch.Generator().manual_seed(seed)).numpy()
    return pd.DataFrame(z, columns=BASE_COLS)


def grouped_cv(make_model, tidy: pd.DataFrame, n_splits: int = 5, consistency: bool = True):
    """Composition-grouped K-fold CV: no composition appears in both train and test folds."""
    folds, preds = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(tidy, groups=tidy["group"])):
        model = make_model(k).fit(tidy.iloc[tr])
        p = model.predict(tidy.iloc[te])
        preds.append(p.assign(fold=k))
        row = {"fold": k, "metrics": regression_metrics(tidy.iloc[te], p)}
        if consistency:
            row["consistency"] = physical_consistency(model, tidy.iloc[te])
            row["consistency_ood"] = physical_consistency(
                model, perturbed_probe_set(tidy.iloc[te], seed=k))
        folds.append(row)
    return folds, pd.concat(preds).sort_index()


def summarise_folds(folds: list) -> dict:
    out = {}
    for t in TARGETS:
        for m in ("R2", "MAE", "RMSE"):
            vals = [f["metrics"][t][m] for f in folds if t in f["metrics"]]
            out[f"{t}_{m}"] = float(np.mean(vals))
            out[f"{t}_{m}_sd"] = float(np.std(vals))
    for block, prefix in (("consistency", "viol_"), ("consistency_ood", "oodviol_")):
        if block in folds[0]:
            for key in folds[0][block]:
                out[f"{prefix}{key}"] = float(np.mean([f[block][key] for f in folds]))
    return out
