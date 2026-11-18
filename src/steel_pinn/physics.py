"""Metallurgical relationships used as physics descriptors and as soft constraints.

All functions take compositions in wt% and temperatures in degC. They accept numpy arrays
(or pandas Series); the ones used inside the training loss also accept torch tensors.

These are approximate, literature-standard engineering relations, not exact governing laws.
They are used (a) to build physically meaningful input descriptors and (b) as soft or
inequality constraints in the physics-informed loss. See docs/physics.md for references.
"""
from __future__ import annotations

import math

import numpy as np

KM_ALPHA = 0.011          # 1/K, Koistinen-Marburger coefficient for plain/low-alloy steels
HJ_CONSTANT = 20.0        # Hollomon-Jaffe constant for steels (t in hours)
UTS_PER_HV = 3.3          # UTS(MPa) ~= 3.3 * HV for carbon / low-alloy steels (ISO 18265-type tables)
EPS_YIELD = 0.002         # 0.2 % offset proof strain


def martensite_start(c, mn, ni, cr, mo):
    """Andrews (1965) linear Ms equation, degC."""
    return 539.0 - 423.0 * c - 30.4 * mn - 17.7 * ni - 12.1 * cr - 7.5 * mo


def km_martensite_fraction(ms, quench_temp=25.0, alpha=KM_ALPHA):
    """Koistinen-Marburger: fraction of austenite transformed to martensite on quenching to T."""
    under = np.clip(np.asarray(ms, dtype=float) - quench_temp, 0.0, None)
    return 1.0 - np.exp(-alpha * under)


def hollomon_jaffe(temper_temp_c, hours=1.0):
    """Tempering parameter P = T[K] (C + log10 t[h]) / 1000."""
    t_k = np.asarray(temper_temp_c, dtype=float) + 273.15
    return t_k * (HJ_CONSTANT + np.log10(hours)) / 1000.0


def carbon_equivalent(c, mn, cr, mo, v, ni, cu):
    """IIW carbon equivalent (hardenability / weldability proxy)."""
    return c + mn / 6.0 + (cr + mo + v) / 5.0 + (ni + cu) / 15.0


def solid_solution_strengthening(mn, si, p, cu, mo):
    """Pickering-type substitutional solid-solution increment for ferrite, MPa (approximate)."""
    return 32.0 * mn + 83.0 * si + 678.0 * p + 39.0 * cu + 11.0 * mo


def hardness_from_uts(uts):
    """Estimated Vickers hardness from tensile strength (approximate, +/- ~10 %)."""
    return uts / UTS_PER_HV


# ---------------------------------------------------------------------------------------
# Hollomon strain hardening + Considere necking criterion
# ---------------------------------------------------------------------------------------
# For sigma = K eps^n, the Considere criterion gives uniform true strain eps_u = n, and
#   YS  ~= K * 0.002^n            (true ~ engineering at 0.2 % strain)
#   UTS  = K * n^n * exp(-n)      (engineering stress at the onset of necking)
# so the yield ratio is   r(n) = YS/UTS = (0.002 e / n)^n ,  strictly decreasing for n > 0.002.
# Total elongation can never be smaller than uniform elongation exp(n)-1. Writing
# n_max = ln(1 + EL/100), the physical requirement n <= n_max is therefore equivalent to
#   YS/UTS >= r(n_max).
# This yields a differentiable inequality linking YS, UTS and EL.

def yield_ratio_from_n(n):
    """r(n) = (0.002 e / n)^n; works with numpy arrays and torch tensors."""
    try:
        import torch
        if isinstance(n, torch.Tensor):
            n = n.clamp(min=EPS_YIELD * math.e)
            return torch.exp(n * torch.log(EPS_YIELD * math.e / n))
    except ImportError:  # pragma: no cover
        pass
    n = np.clip(np.asarray(n, dtype=float), EPS_YIELD * math.e, None)
    return np.exp(n * np.log(EPS_YIELD * math.e / n))


def min_yield_ratio(el_percent):
    """Smallest yield ratio compatible with an elongation of EL % under Hollomon + Considere."""
    try:
        import torch
        if isinstance(el_percent, torch.Tensor):
            return yield_ratio_from_n(torch.log1p(el_percent.clamp(min=0.0) / 100.0))
    except ImportError:  # pragma: no cover
        pass
    return yield_ratio_from_n(np.log1p(np.clip(el_percent, 0.0, None) / 100.0))


def hardening_exponent_from_ratio(ratio, lo=EPS_YIELD * math.e, hi=0.6, iters=60):
    """Invert r(n) by bisection (numpy only). Returns n for each yield ratio."""
    r = np.clip(np.asarray(ratio, dtype=float), yield_ratio_from_n(hi), 0.9999)
    a = np.full_like(r, lo)
    b = np.full_like(r, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        go_right = yield_ratio_from_n(m) > r      # r(n) decreasing: need larger n
        a = np.where(go_right, m, a)
        b = np.where(go_right, b, m)
    return 0.5 * (a + b)


def uniform_elongation_from_ratio(ratio):
    """Engineering uniform elongation (%) implied by the yield ratio."""
    return 100.0 * (np.exp(hardening_exponent_from_ratio(ratio)) - 1.0)
