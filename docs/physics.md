# Physics used in the model

The framework is a **hybrid physics-informed network**: established metallurgical relations enter
as (1) descriptors computed inside the network, (2) hard architectural constraints and (3) soft
loss terms evaluated on unlabelled collocation points. None of these relations is treated as an
exact governing PDE. Most are approximate engineering correlations.

## 1. Descriptors (differentiable layer, `models.PhysicsDescriptorLayer`)

| Descriptor | Relation | Notes |
|---|---|---|
| Martensite start `Ms` | Andrews (1965): `Ms = 539 − 423C − 30.4Mn − 17.7Ni − 12.1Cr − 7.5Mo` (°C) | Linear form, low-alloy steels |
| Martensite fraction `f_M` | Koistinen–Marburger: `f_M = 1 − exp[−0.011 (Ms − T_q)]`, `T_q` = 25 °C, only for quenched records | Gives the upper bound for full through-hardening. Section size and hardenability are left to the network |
| Tempering parameter `P` | Hollomon–Jaffe: `P = T[K](20 + log t[h]) / 1000`, with t = 1 h assumed | Untempered records use T = 25 °C |
| Carbon equivalent | IIW: `C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15` | Proxy for hardenability |
| Solid-solution increment | Pickering-type: `32Mn + 83Si + 678P + 39Cu + 11Mo` (MPa) | Approximate, for ferrite |

The network computes these from the raw inputs, so a gradient such as ∂UTS/∂C includes the
indirect effect through Ms, f_M and CE.

## 2. Hard constraints (built into the architecture)

* `UTS = YS + softplus(·)`, so UTS ≥ YS always holds.
* `EL = softplus(·) > 0` and `HV = softplus(·) > 0`.

## 3. Soft constraints (`losses.py`)

Total loss: `L = L_data + λ(epoch)·(5·L_mono + 20·L_Considère + 1·L_Tabor)`.

λ is 0 for the first 25 % of epochs (data pre-training). It then ramps linearly to 1 over the next
25 %. This is the two-stage schedule set out in the research proposal.

**Monotonicity (sign constraints on input gradients, evaluated on collocation points):**

| Rule | Physical basis | Applied to |
|---|---|---|
| ∂YS/∂T_temper ≤ 0, ∂UTS/∂T_temper ≤ 0, ∂EL/∂T_temper ≥ 0 | Tempering softening: carbide coarsening and recovery | Tempered records |
| ∂YS/∂C ≥ 0, ∂UTS/∂C ≥ 0, ∂EL/∂C ≤ 0 | Interstitial and carbide strengthening, and the strength–ductility trade-off (0–1.1 wt% C) | All |
| ∂YS/∂Mn, ∂YS/∂Si, ∂YS/∂P ≥ 0 | Substitutional solid-solution strengthening | All |
| ∂UTS/∂section ≤ 0 | Mass effect: larger sections cool more slowly and reach lower as-quenched hardness | Quenched records |

**Hollomon + Considère.** With `σ = K εⁿ`, necking starts at uniform true strain `ε_u = n`. The
yield ratio of such a material is `YS/UTS = (0.002·e/n)ⁿ`, which decreases strictly with n. Total
elongation cannot be smaller than uniform elongation `exp(n) − 1`. This gives the differentiable
inequality

```
YS/UTS ≥ r_min(EL) = (0.002·e / n_max)^n_max,   n_max = ln(1 + EL/100)
```

About 90 % of the measured records satisfy this inequality (see `results/data_audit.json` and
`results/figures/considere_bound.png`). The loss therefore allows a 0.03 margin. Some records fall
outside the idealised bound, for example because of yield-point elongation or measurement scatter.

**Tabor-type hardness consistency.** `HV ≈ UTS / 3.3` (ISO 18265-type conversion for carbon and
low-alloy steels, about ±10 %). The training data has **no hardness labels**, so the hardness head
learns only from this soft relation, and every hardness value reported is an *estimate*. If
hardness labels are added, the masked data loss uses them automatically.

## 4. What is *not* modelled (data not available)

* **Hall–Petch:** grain size is not reported in the datasets.
* **Explicit phase fractions:** there are no measured microstructures. The network has only the
  KM estimate and the route flags.
* **Tempering time:** it is rarely reported, so 1 h is assumed in P.
