"""Inverse design: target properties -> composition + heat treatment, via NSGA-II.

Objectives (all minimised):   -UTS,  -EL,  normalised ensemble uncertainty
Constraints (g <= 0):         YS >= target, UTS >= target, HV >= target, EL >= target,
                              candidate inside the applicability domain of the training data.
Search bounds are taken from the training data (1st-99th percentile of the chosen route),
not chosen arbitrarily.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import ELEMENTS
from .data import DEFAULT_AUST_T, FLAG_COLS, build_features

DESIGN_ELEMENTS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V"]
DESIGN_PROCESS = ["aust_temp", "temper_temp"]
FIXED_TRACE = {"P": 0.015, "S": 0.015}          # typical residual levels in wt%


@dataclass
class Targets:
    YS: float = 800.0
    UTS: float = 1000.0
    HV: float = 300.0
    EL: float = 12.0


@dataclass
class Route:
    """Processing route held fixed during the search (oil quench + temper by default)."""
    name: str = "oil_quench_temper"
    flags: dict = field(default_factory=lambda: {"is_quenched": 1, "quench_oil": 1,
                                                 "is_tempered": 1})
    section_mm: float = 25.0


class ApplicabilityDomain:
    """k-NN distance in standardised feature space; threshold = 95th pct of training LOO distances."""

    def __init__(self, train_tidy: pd.DataFrame, k: int = 5, quantile: float = 0.95):
        X = build_features(train_tidy).values
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.train_dist = d[:, 1:].mean(1)               # exclude self
        self.threshold = float(np.quantile(self.train_dist, quantile))

    def distance(self, tidy: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(build_features(tidy).values)
        d, _ = self.nn.kneighbors(Z, n_neighbors=self.k)
        return d.mean(1)


def route_bounds(train: pd.DataFrame, route: Route, lo_q=0.01, hi_q=0.99) -> pd.DataFrame:
    sub = train
    for f, v in route.flags.items():
        sub = sub[sub[f] == v]
    sub = sub[(sub.temper_temp_missing == 0) & (sub.aust_temp_missing == 0)]
    b = sub[DESIGN_ELEMENTS + DESIGN_PROCESS].quantile([lo_q, hi_q]).T
    b.columns = ["lo", "hi"]
    b["hi"] = np.maximum(b["hi"], b["lo"] + 1e-3)
    b.attrs["n_route_records"] = int(len(sub))
    return b


def decode(X: np.ndarray, route: Route) -> pd.DataFrame:
    """Decision vectors -> tidy table with every column the feature builder needs."""
    df = pd.DataFrame(0.0, index=range(len(X)),
                      columns=ELEMENTS + FLAG_COLS + ["aust_temp", "temper_temp", "section_mm",
                                                      "aust_temp_missing", "temper_temp_missing",
                                                      "section_missing"])
    for j, c in enumerate(DESIGN_ELEMENTS + DESIGN_PROCESS):
        df[c] = X[:, j]
    for el, v in FIXED_TRACE.items():
        df[el] = v
    for f, v in route.flags.items():
        df[f] = v
    df["section_mm"] = route.section_mm
    if "aust_temp" not in DESIGN_PROCESS:
        df["aust_temp"] = DEFAULT_AUST_T
    df["Fe"] = 100.0 - df[ELEMENTS].sum(axis=1)
    return df


class AlloyDesignProblem(Problem):
    def __init__(self, model, domain: ApplicabilityDomain, bounds: pd.DataFrame,
                 targets: Targets, route: Route, unc_scale: dict):
        super().__init__(n_var=len(bounds), n_obj=3, n_ieq_constr=5,
                         xl=bounds["lo"].values, xu=bounds["hi"].values)
        self.model, self.domain, self.targets, self.route = model, domain, targets, route
        self.unc_scale = unc_scale

    def _evaluate(self, X, out, *args, **kwargs):
        tidy = decode(X, self.route)
        p = self.model.predict(tidy)
        unc = (p.YS_std / self.unc_scale["YS"] + p.UTS_std / self.unc_scale["UTS"]
               + p.EL_std / self.unc_scale["EL"]) / 3.0
        t = self.targets
        out["F"] = np.column_stack([-p.UTS, -p.EL, unc])
        out["G"] = np.column_stack([
            (t.YS - p.YS) / t.YS, (t.UTS - p.UTS) / t.UTS, (t.HV - p.HV) / t.HV,
            (t.EL - p.EL) / t.EL,
            self.domain.distance(tidy) / self.domain.threshold - 1.0,
        ])


def run_nsga2(model, train: pd.DataFrame, targets: Targets, route: Route | None = None,
              pop_size: int = 120, n_gen: int = 150, seed: int = 0):
    route = route or Route()
    bounds = route_bounds(train, route)
    domain = ApplicabilityDomain(train)
    unc_scale = {t: float(train[t].std()) for t in ("YS", "UTS", "EL")}
    problem = AlloyDesignProblem(model, domain, bounds, targets, route, unc_scale)
    res = minimize(problem, NSGA2(pop_size=pop_size, eliminate_duplicates=True),
                   get_termination("n_gen", n_gen), seed=seed, verbose=False)

    pop_X = res.pop.get("X")
    tidy = decode(pop_X, route)
    pred = model.predict(tidy)
    cand = pd.concat([tidy[DESIGN_ELEMENTS + list(FIXED_TRACE) + ["Fe"] + DESIGN_PROCESS]
                      .reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    cand["ad_distance"] = domain.distance(tidy)
    cand["ad_threshold"] = domain.threshold
    cand["feasible"] = (res.pop.get("CV").ravel() <= 0)
    rank = res.pop.get("rank")
    cand["pareto_rank"] = rank if rank is not None else -1
    return cand, bounds, domain, res


def select_candidates(cand: pd.DataFrame, targets: Targets, bounds: pd.DataFrame, n: int = 5,
                      min_comp_dist: float = 0.05) -> pd.DataFrame:
    """Shortlist feasible, low-uncertainty, diverse candidates (>= min_comp_dist apart).

    score = mean normalised margin above targets  -  uncertainty penalty  -  domain-distance penalty
    """
    f = cand[cand.feasible].copy()
    if f.empty:
        return f
    t = targets
    margin = ((f.YS / t.YS - 1) + (f.UTS / t.UTS - 1) + (f.HV / t.HV - 1) + (f.EL / t.EL - 1)) / 4
    rel_unc = (f.YS_std / f.YS + f.UTS_std / f.UTS + f.EL_std / f.EL) / 3
    f["score"] = margin.clip(upper=0.3) - 2.0 * rel_unc - 0.2 * (f.ad_distance / f.ad_threshold)
    f = f.sort_values("score", ascending=False)
    chosen = []
    comp = DESIGN_ELEMENTS + DESIGN_PROCESS
    scale = (bounds["hi"] - bounds["lo"]).loc[comp]      # RMS distance as a fraction of search range
    for i, row in f.iterrows():
        if all(np.sqrt((((row[comp] - f.loc[j, comp]) / scale) ** 2).mean()) > min_comp_dist
               for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return f.loc[chosen].reset_index(drop=True)
