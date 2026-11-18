"""Neural networks: plain multi-output MLP and the physics-informed multi-task network.

Both networks take *base* inputs in physical units (wt%, degC, mm, 0/1 flags). A differentiable
`PhysicsDescriptorLayer` appends metallurgical descriptors (Ms, Koistinen-Marburger martensite
fraction, Hollomon-Jaffe parameter, carbon equivalent, solid-solution increment), so gradients of
the outputs with respect to e.g. carbon content are true total derivatives in physical units.
"""
from __future__ import annotations

import torch
from torch import nn

from . import physics
from .config import ELEMENTS, ROOM_T
from .data import FLAG_COLS, MISSING_COLS, NUM_PROCESS_COLS

BASE_COLS = ELEMENTS + FLAG_COLS + NUM_PROCESS_COLS + MISSING_COLS
IDX = {c: i for i, c in enumerate(BASE_COLS)}
N_PHYS = 5
Z_CLIP = 5.0


class PhysicsDescriptorLayer(nn.Module):
    """Torch twin of `data.add_physics_descriptors` (verified equal in tests)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = lambda name: x[:, IDX[name]]
        c, mn, si, cr, ni, mo = g("C"), g("Mn"), g("Si"), g("Cr"), g("Ni"), g("Mo")
        ms = physics.martensite_start(c, mn, ni, cr, mo)
        fm = (1.0 - torch.exp(-physics.KM_ALPHA * torch.relu(ms - ROOM_T))) * g("is_quenched")
        p_hj = (g("temper_temp") + 273.15) * physics.HJ_CONSTANT / 1000.0
        ce = physics.carbon_equivalent(c, mn, cr, mo, g("V"), ni, g("Cu"))
        ss = physics.solid_solution_strengthening(mn, si, g("P"), g("Cu"), mo)
        return torch.cat([x, torch.stack([ms, fm, p_hj, ce, ss], dim=1)], dim=1)


def _trunk(d_in: int, hidden: tuple[int, ...], dropout: float) -> tuple[nn.Sequential, int]:
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


class _Base(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, use_descriptors: bool):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.desc = PhysicsDescriptorLayer()
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))  # YS, UTS, EL
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def encode(self, x_raw):
        z = self.desc(x_raw) if self.use_descriptors else x_raw
        # clip to +/- Z_CLIP sigma: networks extrapolate linearly, so an input far outside the
        # training range (e.g. a rare element) would otherwise produce absurd predictions
        return ((z - self.x_mean) / self.x_std).clamp(-Z_CLIP, Z_CLIP)


class PlainMLP(_Base):
    """Baseline: unconstrained multi-output MLP on raw composition + processing."""

    def __init__(self, x_mean, x_std, y_mean, y_std, hidden=(128, 128, 64), dropout=0.05):
        super().__init__(x_mean, x_std, y_mean, y_std, use_descriptors=False)
        self.trunk, d = _trunk(len(BASE_COLS), hidden, dropout)
        self.head = nn.Linear(d, 3)

    def forward(self, x_raw):
        out = self.head(self.trunk(self.encode(x_raw))) * self.y_std + self.y_mean
        return {"YS": out[:, 0], "UTS": out[:, 1], "EL": out[:, 2]}


class PhysicsInformedNet(_Base):
    """Shared trunk + strength / hardness / ductility heads.

    Hard (architectural) constraints:  UTS = YS + softplus(.) >= YS ;  EL > 0 ;  HV > 0.
    Soft constraints live in `losses.py`.
    """

    def __init__(self, x_mean, x_std, y_mean, y_std, hidden=(128, 128, 64), dropout=0.05):
        super().__init__(x_mean, x_std, y_mean, y_std, use_descriptors=True)
        self.trunk, d = _trunk(len(BASE_COLS) + N_PHYS, hidden, dropout)
        self.strength_head = nn.Linear(d, 2)     # YS, (UTS - YS)
        self.hardness_head = nn.Linear(d, 1)     # HV (trained via Tabor consistency / labels if any)
        self.ductility_head = nn.Linear(d, 1)    # EL
        self.softplus = nn.Softplus()

    def forward(self, x_raw):
        h = self.trunk(self.encode(x_raw))
        s = self.strength_head(h)
        ys = self.y_mean[0] + self.y_std[0] * s[:, 0]
        uts = ys + self.y_std[1] * self.softplus(s[:, 1])
        el = self.y_std[2] * self.softplus(self.ductility_head(h)[:, 0] + 1.0)
        hv_scale = self.y_std[1] / physics.UTS_PER_HV
        hv = hv_scale * self.softplus(self.hardness_head(h)[:, 0] + 2.0)
        return {"YS": ys, "UTS": uts, "EL": el, "HV": hv}
