"""Predict properties for one steel with the trained PINN ensemble.

Example:
  python scripts/predict.py --comp C=0.4 Mn=0.8 Si=0.25 Cr=1.0 Mo=0.2 \
      --route oil_qt --aust 845 --temper 540 --section 25
"""
import _common  # noqa: F401
import argparse

import numpy as np

from steel_pinn import training
from steel_pinn.config import MODELS
from steel_pinn.inverse_design import DESIGN_ELEMENTS, FIXED_TRACE, Route, decode

ROUTES = {
    "oil_qt": {"is_quenched": 1, "quench_oil": 1, "is_tempered": 1},
    "water_qt": {"is_quenched": 1, "quench_water": 1, "is_tempered": 1},
    "normalized": {"is_normalized": 1},
    "annealed": {"is_annealed": 1},
    "hot_rolled": {"is_hot_rolled": 1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", nargs="+", required=True, help="El=wt%% pairs, e.g. C=0.4 Mn=0.8")
    ap.add_argument("--route", choices=ROUTES, default="oil_qt")
    ap.add_argument("--aust", type=float, default=845)
    ap.add_argument("--temper", type=float, default=540)
    ap.add_argument("--section", type=float, default=25)
    args = ap.parse_args()

    comp = {k: float(v) for k, v in (kv.split("=") for kv in args.comp)}
    x = np.array([[comp.get(e, 0.0) for e in DESIGN_ELEMENTS] + [args.aust, args.temper]])
    route = Route(flags=ROUTES[args.route], section_mm=args.section)
    tidy = decode(x, route)
    for el, v in comp.items():
        if el not in DESIGN_ELEMENTS:
            tidy[el] = v
    for el in FIXED_TRACE:
        tidy[el] = comp.get(el, FIXED_TRACE[el])
    if not ROUTES[args.route].get("is_tempered"):
        tidy["temper_temp"] = 25.0
    model = training.Ensemble.load(MODELS / "pinn_ensemble.pt")
    p = model.predict(tidy).iloc[0]
    for t, unit in (("YS", "MPa"), ("UTS", "MPa"), ("EL", "%"), ("HV", "HV (estimated)")):
        print(f"{t:4s} {p[t]:8.1f} ± {p[t + '_std']:.1f} {unit}")


if __name__ == "__main__":
    main()
