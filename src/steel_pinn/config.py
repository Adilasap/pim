"""Shared constants: paths, element lists, targets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"

PRIMARY_XLSX = RAW / "steel_database_birbilis_2025.xlsx"
BENCHMARK_JSON = RAW / "steel_strength_matminer.json.gz"

# Union of alloying elements in the two steel datasets (Fe is the balance and is not a feature).
ELEMENTS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Nb", "Ti", "Al", "Cu",
            "Co", "W", "N", "P", "S", "B", "Zr"]

TARGETS = ["YS", "UTS", "EL"]          # yield strength (MPa), tensile strength (MPa), elongation (%)
TARGET_LABELS = {"YS": "Yield strength (MPa)", "UTS": "Tensile strength (MPa)",
                 "EL": "Elongation (%)", "HV": "Vickers hardness (HV, estimated)"}

ROOM_T = 25.0          # degC, final quench temperature assumed for Koistinen-Marburger
SEED = 42
