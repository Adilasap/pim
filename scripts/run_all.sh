#!/usr/bin/env bash
# Full reproducible pipeline: data -> benchmark -> final model -> inverse design.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/benchmark_models.py
python scripts/train_final.py
python scripts/run_inverse_design.py
