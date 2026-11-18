"""Loading, cleaning, processing-text parsing and feature engineering.

Pipeline:  raw file  ->  tidy table (composition wt%, structured processing, targets)
                     ->  model features (composition + processing + physics descriptors)

`build_features` works on any tidy table, so the inverse-design module can build the exact same
features for candidate alloys that never existed in the dataset.
"""
from __future__ import annotations

import gzip
import json
import re

import numpy as np
import pandas as pd

from . import physics
from .config import BENCHMARK_JSON, ELEMENTS, PRIMARY_XLSX, ROOM_T

# Structured processing columns produced by `parse_processing`
FLAG_COLS = ["is_quenched", "quench_oil", "quench_water", "is_tempered", "is_normalized",
             "is_annealed", "is_carburized", "is_cold_drawn", "is_hot_rolled",
             "is_stress_relieved", "is_aged", "is_double_quenched", "no_process_info"]
NUM_PROCESS_COLS = ["aust_temp", "temper_temp", "section_mm"]
MISSING_COLS = ["aust_temp_missing", "temper_temp_missing", "section_missing"]
PHYSICS_COLS = ["Ms", "f_martensite", "P_hollomon_jaffe", "CE_iiw", "dsigma_ss"]

# Fill values for unreported numeric processing parameters (a missing-flag is always kept,
# so the model can tell a measured value from an imputed one).
DEFAULT_AUST_T = 845.0
DEFAULT_TEMPER_T = 425.0
DEFAULT_SECTION = 25.0

_TEMP = r"(\d{2,4})\s*C\b"


def _has(text: pd.Series, pattern: str) -> pd.Series:
    return text.str.contains(pattern, case=False, regex=True).astype(int)


def _temper_temp(s: str) -> float:
    """Tempering temperature; only sub-critical values (<= 760 C, below A1) are accepted."""
    pats = [_TEMP + r"\s*(?:\(\s*\d+\s*F\s*\))?\s*,?\s*Temper",      # "595C (1100F) Temper"
            r"Temper(?!atur)(?:ed)?\s*(?:At|To|:|,)?\s*\(?\s*" + _TEMP]         # "Tempered (150C" / "Tempered At 650C"
    for p in pats:
        for m in re.finditer(p, s, flags=re.I):
            t = float(m.group(1))
            if 100 <= t <= 760:
                return t
    return np.nan


def _aust_temp(s: str) -> float:
    """Last hardening/normalising temperature >= 700 C that is not the carburising temperature."""
    carb = re.search(r"Carburi[sz]ed\s*(?:At|To)?\s*" + _TEMP, s, flags=re.I)
    carb_t = float(carb.group(1)) if carb else None
    temps = [float(t) for t in re.findall(_TEMP, s)]
    temps = [t for t in temps if 700 <= t <= 1250]
    if carb_t is not None and carb_t in temps:
        temps.remove(carb_t)
    return temps[-1] if temps else np.nan


def _section(s: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*mm", s, flags=re.I)
    if not m:
        return np.nan
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return 0.5 * (lo + hi)


def parse_processing(text: pd.Series) -> pd.DataFrame:
    """Turn free-text processing descriptions into structured columns."""
    t = text.fillna("").astype(str)
    out = pd.DataFrame(index=t.index)
    out["is_carburized"] = _has(t, r"carburi[sz]")
    out["quench_oil"] = _has(t, r"\boil\b")
    out["quench_water"] = _has(t, r"water|brine")
    out["is_quenched"] = ((_has(t, r"quench") + out["is_carburized"] + out["quench_oil"]
                           + out["quench_water"]) > 0).astype(int)
    # carburised grades in this database are oil quenched unless stated otherwise
    out.loc[(out.is_carburized == 1) & (out.quench_water == 0), "quench_oil"] = 1
    out["is_tempered"] = _has(t, r"temper(?!atur)")
    out["is_normalized"] = _has(t, r"normali[sz]")
    out["is_annealed"] = _has(t, r"anneal|spheroidi")
    out["is_cold_drawn"] = _has(t, r"cold\s*drawn|cold\s*finish|cold\s*work")
    out["is_hot_rolled"] = _has(t, r"hot\s*rolled|as\s*rolled|hot\s*finish")
    out["is_stress_relieved"] = _has(t, r"stress\s*reliev")
    out["is_aged"] = _has(t, r"\baged\b|ageing|aging|precipitation")
    out["is_double_quenched"] = _has(t, r"double\s*quench")

    aust = t.map(_aust_temp)
    temper = t.map(_temper_temp)
    section = t.map(_section)

    out["aust_temp_missing"] = aust.isna().astype(int)
    out["aust_temp"] = aust.fillna(DEFAULT_AUST_T)
    # Not tempered -> treat the "tempering temperature" as room temperature (physically exact:
    # no tempering). Tempered but temperature unreported -> impute and flag.
    out["temper_temp_missing"] = ((out.is_tempered == 1) & temper.isna()).astype(int)
    temper = temper.where(out.is_tempered == 1, ROOM_T)
    out["temper_temp"] = temper.fillna(DEFAULT_TEMPER_T)
    out["section_missing"] = section.isna().astype(int)
    out["section_mm"] = section.fillna(DEFAULT_SECTION)

    route = out[[c for c in FLAG_COLS if c != "no_process_info"]].sum(axis=1)
    out["no_process_info"] = ((route == 0) & (out.aust_temp_missing == 1)).astype(int)
    return out[FLAG_COLS + NUM_PROCESS_COLS + MISSING_COLS]


def add_physics_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Append metallurgy-derived descriptors computed from composition and processing."""
    out = df.copy()
    out["Ms"] = physics.martensite_start(out.C, out.Mn, out.Ni, out.Cr, out.Mo)
    fm = physics.km_martensite_fraction(out.Ms.values, ROOM_T)
    out["f_martensite"] = fm * out.is_quenched.values       # only quenched steels form martensite
    out["P_hollomon_jaffe"] = physics.hollomon_jaffe(out.temper_temp.values)
    out["CE_iiw"] = physics.carbon_equivalent(out.C, out.Mn, out.Cr, out.Mo, out.V, out.Ni, out.Cu)
    out["dsigma_ss"] = physics.solid_solution_strengthening(out.Mn, out.Si, out.P, out.Cu, out.Mo)
    return out


FEATURES = ELEMENTS + FLAG_COLS + NUM_PROCESS_COLS + MISSING_COLS + PHYSICS_COLS


def build_features(tidy: pd.DataFrame) -> pd.DataFrame:
    """Model input matrix (always the same column order)."""
    return add_physics_descriptors(tidy)[FEATURES].astype(float)


def composition_group(df: pd.DataFrame) -> pd.Series:
    """Group id = rounded composition. Used for leakage-free (grouped) splits."""
    return df[ELEMENTS].round(3).astype(str).agg("|".join, axis=1)


# ---------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------

def load_primary(path=PRIMARY_XLSX) -> pd.DataFrame:
    """Birbilis, Ghorbani & Zhao (2025) steel database, 3,234 records, CC BY 4.0."""
    raw = pd.read_excel(path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    df = pd.DataFrame({"entry": raw["Entry"], "name": raw["Name"],
                       "processing": raw["Processing condition"].astype(str),
                       "cluster": raw["Cluster Number"], "cluster_label": raw["Cluster label"]})
    for el in ELEMENTS:
        df[el] = raw[el].astype(float) if el in raw.columns else 0.0
    df["Fe"] = raw["Fe"].astype(float)
    df["YS"] = raw["Yield strength (MPa)"].astype(float)
    df["UTS"] = raw["(Ultimate) Tensile strength (MPa)"].astype(float)
    df["EL"] = raw["Ductility (%)"].astype(float)
    df = pd.concat([df, parse_processing(df.processing)], axis=1)
    df["source"] = "birbilis2025"
    return df


def load_benchmark(path=BENCHMARK_JSON) -> pd.DataFrame:
    """Citrine/Matminer `steel_strength` (312 high-strength steels, composition only)."""
    with gzip.open(path, "rt") as fh:
        j = json.load(fh)
    raw = pd.DataFrame(j["data"], columns=j["columns"])
    df = pd.DataFrame({"name": raw["formula"]})
    for el in ELEMENTS:
        df[el] = raw[el.lower()].astype(float) if el.lower() in raw.columns else 0.0
    df["Fe"] = 100.0 - df[ELEMENTS].sum(axis=1)
    df["YS"] = raw["yield strength"].astype(float)
    df["UTS"] = raw["tensile strength"].astype(float)
    df["EL"] = raw["elongation"].astype(float)          # 9 missing -> NaN, kept as missing
    df["processing"] = ""                                # not reported in this dataset
    df = pd.concat([df, parse_processing(df.processing)], axis=1)
    df["source"] = "matminer_steel_strength"
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicates and physically impossible records; return an audit log."""
    log = {"n_raw": int(len(df))}
    key = ELEMENTS + ["processing", "YS", "UTS", "EL"]
    df = df.drop_duplicates(subset=key).copy()
    log["n_after_exact_dedup"] = int(len(df))
    bad = (df.YS > df.UTS * 1.001) | (df.YS <= 0) | (df.UTS <= 0) | (df.EL < 0)
    log["n_removed_ys_gt_uts_or_nonpositive"] = int(bad.sum())
    df = df[~bad].copy()
    log["n_clean"] = int(len(df))
    df["group"] = composition_group(df)
    log["n_composition_groups"] = int(df.group.nunique())
    return df.reset_index(drop=True), log
