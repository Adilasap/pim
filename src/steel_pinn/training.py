"""Model fitting: tree baselines, plain MLP, physics-informed network and deep ensembles.

Every fitted model exposes  `predict(tidy_df) -> DataFrame[YS, UTS, EL, HV]`  so evaluation,
physical-consistency probes and inverse design treat all models identically.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from . import losses, physics
from .config import ELEMENTS, TARGETS
from .data import build_features
from .models import BASE_COLS, IDX, PhysicsDescriptorLayer, PhysicsInformedNet, PlainMLP

torch.set_num_threads(max(1, torch.get_num_threads()))


def _with_hv(df: pd.DataFrame) -> pd.DataFrame:
    if "HV" not in df:
        df["HV"] = physics.hardness_from_uts(df["UTS"].values)
    return df


# ---------------------------------------------------------------------------------------
# Tree baselines
# ---------------------------------------------------------------------------------------

class TreeModel:
    """One regressor per target on the engineered feature matrix (incl. physics descriptors)."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed, self.models = kind, seed, {}

    def _make(self):
        if self.kind == "rf":
            return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                         random_state=self.seed)
        return XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                            n_jobs=4, verbosity=0)

    def fit(self, tidy: pd.DataFrame):
        X = build_features(tidy).values
        for t in TARGETS:
            m = tidy[t].notna().values
            self.models[t] = self._make().fit(X[m], tidy[t].values[m])
        return self

    def predict(self, tidy: pd.DataFrame) -> pd.DataFrame:
        X = build_features(tidy).values
        return _with_hv(pd.DataFrame({t: self.models[t].predict(X) for t in TARGETS},
                                     index=tidy.index))


# ---------------------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------------------

@dataclass
class NNConfig:
    kind: str = "pinn"                 # "pinn" | "mlp"
    physics_loss: bool = True          # only meaningful for kind == "pinn"
    hidden: tuple = (128, 128, 64)
    dropout: float = 0.05
    epochs: int = 600
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    pretrain_frac: float = 0.25        # stage 1: data only; stage 2: physics weight ramps up
    lambda_phys: float = 1.0
    w_mono: float = 5.0
    w_considere: float = 20.0
    w_tabor: float = 1.0
    n_colloc: int = 256
    patience: int = 120
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _tensor(df: pd.DataFrame, cols) -> torch.Tensor:
    return torch.tensor(df[cols].values.astype(np.float32))


def sample_collocation(x: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """Unlabelled points spread around the training distribution (perturbed real records)."""
    idx = torch.randint(0, x.shape[0], (n,), generator=gen)
    z = x[idx].clone()
    el = [IDX[e] for e in ELEMENTS]
    z[:, el] = z[:, el] * torch.exp(0.3 * torch.randn(n, len(el), generator=gen))
    z[:, IDX["C"]] = (z[:, IDX["C"]] + 0.1 * (torch.rand(n, generator=gen) - 0.5)).clamp(min=0.0)
    tempered = z[:, IDX["is_tempered"]] > 0.5
    new_t = 150.0 + 555.0 * torch.rand(n, generator=gen)
    z[:, IDX["temper_temp"]] = torch.where(tempered, new_t, z[:, IDX["temper_temp"]])
    z[:, IDX["section_mm"]] = (z[:, IDX["section_mm"]]
                               * torch.exp(0.5 * torch.randn(n, generator=gen))).clamp(5.0, 200.0)
    z[:, IDX["aust_temp"]] += 30.0 * torch.randn(n, generator=gen)
    return z


class NNModel:
    def __init__(self, cfg: NNConfig):
        self.cfg = cfg
        self.net = None
        self.history = []

    # -- helpers --------------------------------------------------------------------------
    def _build(self, x: torch.Tensor, y: torch.Tensor):
        cfg = self.cfg
        z = PhysicsDescriptorLayer()(x) if cfg.kind == "pinn" else x
        x_mean, x_std = z.mean(0), z.std(0)
        x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)   # constant columns
        y_mean = torch.nanmean(y, 0)
        y_std = torch.tensor([np.nanstd(y[:, j].numpy()) for j in range(y.shape[1])])
        cls = PhysicsInformedNet if cfg.kind == "pinn" else PlainMLP
        return cls(x_mean, x_std, y_mean, y_std, hidden=cfg.hidden, dropout=cfg.dropout)

    def _phys_weight(self, epoch: int) -> float:
        cfg = self.cfg
        if cfg.kind != "pinn" or not cfg.physics_loss:
            return 0.0
        start = int(cfg.pretrain_frac * cfg.epochs)
        ramp = max(1, int(0.25 * cfg.epochs))
        return cfg.lambda_phys * float(np.clip((epoch - start) / ramp, 0.0, 1.0))

    # -- fit ------------------------------------------------------------------------------
    def fit(self, tidy: pd.DataFrame):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        gen = torch.Generator().manual_seed(cfg.seed)

        split = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=cfg.seed)
        tr_idx, va_idx = next(split.split(tidy, groups=tidy["group"]))
        tr, va = tidy.iloc[tr_idx], tidy.iloc[va_idx]
        x_tr, y_tr = _tensor(tr, BASE_COLS), _tensor(tr, TARGETS)
        x_va, y_va = _tensor(va, BASE_COLS), _tensor(va, TARGETS)

        net = self._build(x_tr, y_tr)
        feat_std = {c: float(x_tr[:, IDX[c]].std().clamp(min=1e-6)) for c in BASE_COLS}
        y_std_d = {t: float(net.y_std[j]) for j, t in enumerate(TARGETS)}
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        best, best_state, bad = np.inf, None, 0
        n = x_tr.shape[0]
        phys_start = int(cfg.pretrain_frac * cfg.epochs) if self._phys_weight(cfg.epochs) else 0
        for epoch in range(cfg.epochs):
            net.train()
            lam = self._phys_weight(epoch)
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, cfg.batch_size):
                b = perm[i:i + cfg.batch_size]
                pred = net(x_tr[b])
                loss = losses.data_loss(pred, y_tr[b], net.y_std)
                if lam > 0:
                    xc = sample_collocation(x_tr, cfg.n_colloc, gen)
                    pc = net(xc)
                    l_phys = (cfg.w_mono * losses.monotonic_loss(net, xc, feat_std, y_std_d)
                              + cfg.w_considere * losses.considere_loss(pc)
                              + cfg.w_tabor * losses.tabor_loss(pc))
                    loss = loss + lam * l_phys
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                v = float(losses.data_loss(net(x_va), y_va, net.y_std))
            self.history.append({"epoch": epoch, "val_loss": v, "lambda_phys": lam})
            # only select checkpoints once the physics loss is fully active
            if epoch >= phys_start and v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            elif epoch >= phys_start:
                bad += 1
                full_phys = lam >= cfg.lambda_phys or lam == 0.0
                if bad > cfg.patience and full_phys:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    # -- predict --------------------------------------------------------------------------
    def predict(self, tidy: pd.DataFrame) -> pd.DataFrame:
        with torch.no_grad():
            p = self.net(_tensor(tidy, BASE_COLS))
        return _with_hv(pd.DataFrame({k: v.numpy() for k, v in p.items()}, index=tidy.index))


class Ensemble:
    """Deep ensemble: mean prediction + epistemic spread (std across members)."""

    def __init__(self, members):
        self.members = members

    @classmethod
    def fit(cls, tidy, cfg: NNConfig, n_members: int = 5):
        members = []
        for k in range(n_members):
            c = copy.deepcopy(cfg)
            c.seed = cfg.seed + 1000 * k
            members.append(NNModel(c).fit(tidy))
        return cls(members)

    def predict_all(self, tidy):
        return [m.predict(tidy) for m in self.members]

    def predict(self, tidy: pd.DataFrame) -> pd.DataFrame:
        preds = self.predict_all(tidy)
        mean = sum(preds) / len(preds)
        std = np.sqrt(sum((p - mean) ** 2 for p in preds) / len(preds))
        out = mean.copy()
        for c in std.columns:
            out[f"{c}_std"] = std[c].values
        return out

    def save(self, path):
        torch.save({"cfg": [m.cfg for m in self.members],
                    "state": [m.net.state_dict() for m in self.members],
                    "buffers": [{k: getattr(m.net, k) for k in ("x_mean", "x_std", "y_mean", "y_std")}
                                for m in self.members]}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, weights_only=False)
        members = []
        for cfg, state, buf in zip(blob["cfg"], blob["state"], blob["buffers"]):
            netcls = PhysicsInformedNet if cfg.kind == "pinn" else PlainMLP
            net = netcls(buf["x_mean"], buf["x_std"], buf["y_mean"], buf["y_std"],
                         hidden=cfg.hidden, dropout=cfg.dropout)
            net.load_state_dict(state)
            net.eval()
            m = NNModel(cfg)
            m.net = net
            members.append(m)
        return cls(members)
