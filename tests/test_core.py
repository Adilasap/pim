import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steel_pinn import data, physics  # noqa: E402
from steel_pinn.config import PRIMARY_XLSX  # noqa: E402
from steel_pinn.models import BASE_COLS, PhysicsDescriptorLayer, PhysicsInformedNet  # noqa: E402


def test_parse_processing_examples():
    t = pd.Series([
        "Oil Quenched From 830C (1525F), 540C (1000F) Temper, 25 mm (1 In.) Round",
        "Carburized At 925C (1700F), Box Cooled, 775C (1430F) Reheat, Water Quenched, 175C (350F) Temper",
        "Cold Drawn, High Temperature, Stress Relieved, 22-32 mm (0.875-1.25 In) Round",
        "Grade B",
    ])
    p = data.parse_processing(t)
    assert p.loc[0, ["is_quenched", "quench_oil", "is_tempered"]].tolist() == [1, 1, 1]
    assert p.loc[0, "temper_temp"] == 540 and p.loc[0, "aust_temp"] == 830
    assert p.loc[0, "section_mm"] == 25
    assert p.loc[1, "aust_temp"] == 775 and p.loc[1, "temper_temp"] == 175  # not the 925 C carburise
    assert p.loc[1, "quench_water"] == 1 and p.loc[1, "quench_oil"] == 0
    assert p.loc[2, "is_tempered"] == 0 and p.loc[2, "section_mm"] == 27   # "Temperature" != temper
    assert p.loc[3, "no_process_info"] == 1


def test_considere_inverse_roundtrip():
    n = np.array([0.05, 0.1, 0.2, 0.3])
    r = physics.yield_ratio_from_n(n)
    assert np.all(np.diff(r) < 0)
    assert np.allclose(physics.hardening_exponent_from_ratio(r), n, atol=1e-6)


def test_martensite_physics():
    ms = physics.martensite_start(0.4, 0.8, 0.0, 1.0, 0.2)
    assert 300 < ms < 400
    assert physics.km_martensite_fraction(ms) > 0.95
    assert physics.km_martensite_fraction(10.0) == 0.0          # Ms below room temperature


def test_torch_descriptors_match_numpy():
    df = pd.DataFrame(np.abs(np.random.default_rng(0).normal(0.5, 0.3, (20, len(BASE_COLS)))),
                      columns=BASE_COLS)
    df["is_quenched"] = (df.is_quenched > 0.5).astype(float)
    df["temper_temp"] = 300.0
    ref = data.add_physics_descriptors(df)[data.PHYSICS_COLS].values
    got = PhysicsDescriptorLayer()(torch.tensor(df.values, dtype=torch.float32))[:, -5:].numpy()
    assert np.allclose(ref, got, rtol=1e-4, atol=1e-3)


def test_pinn_hard_constraints():
    d = len(BASE_COLS) + 5
    net = PhysicsInformedNet(torch.zeros(d), torch.ones(d), torch.tensor([600., 850., 18.]),
                             torch.tensor([250., 280., 7.]))
    out = net(torch.randn(64, len(BASE_COLS)) * 3)
    assert torch.all(out["UTS"] >= out["YS"]) and torch.all(out["EL"] > 0) and torch.all(out["HV"] > 0)


@pytest.mark.skipif(not PRIMARY_XLSX.exists(), reason="raw data not downloaded")
def test_primary_dataset_audit():
    df, log = data.clean(data.load_primary())
    assert log["n_raw"] == 3234
    assert np.allclose(df[data.ELEMENTS + ["Fe"]].sum(axis=1), 100, atol=0.05)
    assert (df.UTS >= df.YS * 0.999).all()
