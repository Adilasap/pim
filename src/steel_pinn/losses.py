"""Physics-informed loss terms.

L = L_data + lambda_phys(epoch) * (w_mono * L_mono + w_cons * L_considere + w_tabor * L_tabor)

* L_data       masked, standardised MSE (records may miss some properties).
* L_mono       sign constraints on input gradients, evaluated on collocation points:
                 tempering softening   dYS/dT_temper <= 0, dUTS/dT_temper <= 0, dEL/dT_temper >= 0
                 carbon strengthening  dYS/dC >= 0, dUTS/dC >= 0, dEL/dC <= 0
                 solid solution        dYS/dMn >= 0, dYS/dSi >= 0, dYS/dP >= 0
                 mass effect           dUTS/d(section) <= 0          (quenched steels only)
* L_considere  Hollomon + Considere: total elongation >= uniform elongation, i.e.
                 YS/UTS >= r_min(EL) - margin
* L_tabor      hardness head consistent with HV ~= UTS / 3.3 (soft, relative error)
"""
from __future__ import annotations

import torch

from . import physics
from .models import IDX

# (output, input feature, required sign of derivative, mask flag or None)
MONOTONIC_RULES = [
    ("YS", "temper_temp", -1, "is_tempered"),
    ("UTS", "temper_temp", -1, "is_tempered"),
    ("EL", "temper_temp", +1, "is_tempered"),
    ("YS", "C", +1, None),
    ("UTS", "C", +1, None),
    ("EL", "C", -1, None),
    ("YS", "Mn", +1, None),
    ("YS", "Si", +1, None),
    ("YS", "P", +1, None),
    ("UTS", "section_mm", -1, "is_quenched"),
]

CONSIDERE_MARGIN = 0.03   # ~10 % of records violate the idealised Hollomon bound; allow slack


def data_loss(pred: dict, y: torch.Tensor, y_std: torch.Tensor, targets=("YS", "UTS", "EL")):
    """Masked MSE in standardised units; NaN targets are ignored."""
    total, count = 0.0, 0
    for j, t in enumerate(targets):
        m = ~torch.isnan(y[:, j])
        if m.any():
            total = total + (((pred[t][m] - y[m, j]) / y_std[j]) ** 2).sum()
            count += int(m.sum())
    return total / max(count, 1)


def monotonic_loss(model, x: torch.Tensor, feat_std: dict, y_std: dict):
    """Hinge-squared penalty on wrong-signed input gradients (dimensionless)."""
    x = x.clone().requires_grad_(True)
    pred = model(x)
    loss = x.new_zeros(())
    grads = {}
    for out in {r[0] for r in MONOTONIC_RULES}:
        grads[out] = torch.autograd.grad(pred[out].sum(), x, create_graph=True)[0]
    for out, feat, sign, mask_col in MONOTONIC_RULES:
        g = grads[out][:, IDX[feat]] * feat_std[feat] / y_std[out]   # dimensionless sensitivity
        viol = torch.relu(-sign * g) ** 2
        if mask_col is not None:
            m = x[:, IDX[mask_col]].detach()
            viol = viol * m
            loss = loss + viol.sum() / m.sum().clamp(min=1.0)
        else:
            loss = loss + viol.mean()
    return loss / len(MONOTONIC_RULES)


def considere_loss(pred: dict, margin: float = CONSIDERE_MARGIN):
    ratio = pred["YS"] / pred["UTS"].clamp(min=1.0)
    r_min = physics.min_yield_ratio(pred["EL"])
    return (torch.relu(r_min - margin - ratio) ** 2).mean()


def tabor_loss(pred: dict):
    if "HV" not in pred:
        return pred["YS"].new_zeros(())
    hv_ref = physics.hardness_from_uts(pred["UTS"]).detach()
    return (((pred["HV"] - hv_ref) / hv_ref.clamp(min=1.0)) ** 2).mean()
