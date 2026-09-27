# Compression Strategy, Not Compression Ratio: Explanation-Fidelity Trade-offs in Quantized and Pruned Cardiac Risk Models

> How much does compressing a cardiac-risk DNN for edge/IoMT deployment change its *explanations*, not just its accuracy? This repo trains, compresses, and audits a cardiac-risk classifier so that SHAP-based explanation fidelity — not just accuracy, latency, or size — determines what actually ships to the edge.

Wearable and bedside Internet-of-Medical-Things (IoMT) devices for cardiac risk screening must run on constrained hardware, which typically means compressing a neural network via quantization or pruning. Accuracy, latency, and size are the usual metrics for judging a compressed model — but none of them tell you whether the compressed model's *reasoning* still matches the original's. This project closes that gap.

We train a DNN on the IEEE DataPort heart disease dataset (n = 1,190), compress it into `float16`, `int8`, and `pruned + int8` TensorFlow Lite variants, and measure how far each variant's SHAP attributions drift from the full-precision `float32` reference — using top-k rank agreement, Spearman rank correlation, and a new **Importance-Weighted Explanation Fidelity (IWEF)** metric, all computed with an *exact* Shapley explainer over 10 random seeds and benchmarked against an explicit estimator noise floor.

**Headline finding:** discriminative performance is comparable across variants (AUC 0.93–0.95), but fidelity is not. Quantization alone drifts *below* the noise floor (Spearman ρ ≈ 0.89 vs. a floor of ρ ≈ 0.97), while pruning before quantization *exceeds* it (ρ ≈ 0.99) despite lower raw accuracy. **Compression strategy, not compression ratio, determines whether a model stays explainable** — and that has to be measured directly, not assumed from accuracy.

---

## Table of Contents

- [Why this matters](#why-this-matters)
- [Pipeline overview](#pipeline-overview)
- [Repository contents](#repository-contents)
- [Key results](#key-results)
- [Getting started](#getting-started)
- [Reproducing the paper's numbers](#reproducing-the-papers-numbers)
- [Exported artifacts](#exported-artifacts)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Why this matters

Edge-AI health literature typically treats a compressed model as interchangeable with its full-precision counterpart once accuracy is preserved — explanations included. This assumption is rarely tested, even though compression is known to change model behavior in ways aggregate accuracy hides. Two models can agree on most predictions and still disagree on *why*. For clinical decision support, where SHAP-style attributions are what clinicians and patients actually see, that gap matters.

To our knowledge, no prior cardiac-risk study has measured whether a compressed model's explanations survive quantization or pruning, or whether that survival depends on *which* compression technique is used rather than only on how aggressive it is.

## Pipeline overview

A seven-stage pipeline, fully reproduced in the accompanying notebook:

1. **Data & preprocessing** — IEEE DataPort D1 (five merged cohorts, N = 1,190, 11 clinical features), disguised-missing-value imputation, standardization fit on the training partition only.
2. **Stratified splits & multi-seed protocol** — 80/20 train/test, 85/15 train/validation, repeated over `S = 10` seeds.
3. **Model training** — SVM / Random Forest / KNN baselines (5-fold grid search) plus a DNN tuned over a full 4×3×2 = 24-configuration grid (architecture × dropout × learning rate), trained with class-weighted cross-entropy.
4. **TFLite compression** — four headline variants (`float32` reference, `float16`, `int8`, `pruned+int8`) plus two ablations (`finetune_only`, `pruned_f32`) that isolate fine-tuning and pruning effects from quantization.
5. **Explanation-fidelity engine** (core contribution) — exact SHAP over all 2¹¹ = 2,048 feature coalitions for every test sample and variant; top-k Jaccard agreement, Spearman ρ, IWEF, an explicit float32-vs-float32 noise floor, McNemar's exact test for prediction-level change, and a Shapley-linearity bound on attribution rank swaps.
6. **Uncertainty-aware tier selection** — Algorithm 1 selects the smallest compressed variant whose fidelity *and* accuracy clear operator-set thresholds at their lower 95% confidence bound, deferring to the cloud if none qualify.
7. **Three-tier IoMT deployment** — wearable/sensor → edge gateway (compressed DNN) → cloud (float32 confirmatory inference on flagged high-risk cases).

## Repository contents

| File | Description |
|---|---|
| `IoMT_Cardiac_Risk_Edge_Compression_Pipeline_v4.ipynb` | End-to-end, single-notebook pipeline — data loading through final figures. Runs in Google Colab or local Jupyter (CPU only). |
| `paper/` *(add your PDF here)* | Manuscript describing the methodology and results in full (AISTATS-format submission). |
| `outputs/` *(generated on run)* | All CSVs, figures, LaTeX macros, and the reproducibility manifest produced by the notebook. |

> This is a single, working research notebook, not a packaged library — every section is independently re-runnable and writes its outputs to `OUT_DIR` (`./outputs` locally, `/content/outputs` on Colab).

## Key results

| Model | Accuracy | AUC | Size (KB) | Top-k Jaccard | Spearman ρ | IWEF |
|---|---|---|---|---|---|---|
| DNN float32 (reference) | 0.839 ± 0.019 | 0.948 | 40.76 | 1.000 | 1.000 | 1.000 |
| **Noise floor** (float32 vs. itself) | – | – | – | 0.833 | 0.974 | 0.873 |
| DNN float16 (TFLite) | 0.859 ± 0.014 | 0.948 | 22.29 | 0.743 ± 0.047 | 0.897 ± 0.020 | 0.813 ± 0.024 |
| DNN int8 (TFLite) | 0.857 ± 0.015 | 0.947 | 17.25 | 0.741 ± 0.045 | 0.893 ± 0.021 | 0.812 ± 0.023 |
| **DNN pruned+int8 (TFLite)** | 0.836 ± 0.018 | 0.928 | 17.25 | **0.970 ± 0.017** | **0.991 ± 0.006** | **0.979 ± 0.007** |

Quantization alone (`float16`, `int8`) falls **below** the noise floor — its explanations are, statistically, no more faithful than the reference model re-explained with a different background sample. Pruning before quantization **exceeds** the noise floor at the same on-disk size, despite lower raw accuracy. McNemar's exact test finds no significant prediction-level change from float32 for any variant — compression here changes *how faithfully the model explains itself*, not *what it predicts*.

## Getting started

**Requirements:** Python 3, run on CPU (no GPU required/assumed).

Core dependencies (installed at the top of the notebook):

```bash
pip install shap tensorflow-model-optimization
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy
```

Open `IoMT_Cardiac_Risk_Edge_Compression_Pipeline_v4.ipynb` in Google Colab or Jupyter and run top to bottom. Set the hardware accelerator to **CPU**.

The notebook includes `RUN_MODE` toggles for a fast smoke test — use this first, since the full `N_SEEDS = 10`, full-238-sample exact-SHAP run is considerably slower than a single-seed pass.

## Reproducing the paper's numbers

1. Run Sections 0–8 for a single-seed sanity check of every metric (baselines, compression, fidelity engine, noise floor).
2. Run Section 9 (multi-seed loop, `N_SEEDS = 10`) for the headline mean ± 95% CI numbers reported in Table 1 — this is the slowest section; use `RUN_MODE="smoke"` first.
3. Run Section 10 for the de-duplication sensitivity check (D1 contains 272 exact duplicate rows).
4. Run Section 12 for Algorithm 1's uncertainty-aware tier selection and the paired Wilcoxon significance tests between variants.
5. Run Section 13 to regenerate `paper_macros.tex` — every number quoted in the manuscript is written here directly from the results tables, so the paper and the code cannot silently diverge.
6. For real edge-hardware numbers (not the CPU-proxy latency reported here), copy the script in Section 17 to a Raspberry Pi or ESP32-class device — this is **not** meant to run inside Colab.

## Exported artifacts

Everything is written to `OUT_DIR`, including: baseline and full model comparison tables, the 24-config DNN grid-search log, McNemar significance results, CPU-proxy latency/size benchmarks, the sparsity sweep, all confusion matrices/ROC curves, single-run and multi-seed fidelity tables (with the noise floor row), the Shapley-linearity bound check, the permutation-importance cross-check, dedup-sensitivity results, paired Wilcoxon tests, `tier_selection.json` (Algorithm 1's output), the final `table1_final.csv`, `paper_macros.tex`, the trade-off and seven-panel synthesis figures, the pipeline diagram, a full `reproducibility_manifest.json`, and `X_test.npy`/`y_test.npy` for handoff to physical edge-hardware benchmarking.

## Limitations

- D1 is a single retrospective tabular dataset (272 retained exact duplicates); accuracy figures may be mildly optimistic, though the fidelity *ordering* across variants is unchanged after de-duplication.
- `float16` and `int8` fidelity remain statistically indistinguishable from each other even at 10 seeds.
- Latency/size are a cloud-CPU proxy, not physical edge hardware (a Raspberry Pi / ESP32 script is included for follow-up).
- The DNN trails Random Forest in raw accuracy by design — it's the deployment target because it has a direct quantization/pruning path, not because it's the strongest classifier.
- SHAP and magnitude pruning are the only explainer and compression strategy tested in depth (a permutation-importance cross-check is included as a second, model-agnostic explainer).

## Citation

If you use this code or build on this work, please cite the accompanying manuscript (currently under review):

```
Anonymous Author. "Compression Strategy, Not Compression Ratio: Explanation-Fidelity
Trade-offs in Quantized and Pruned Cardiac Risk Models." Under review, AISTATS 2027.
```

## License

Add a license of your choice (e.g. MIT, Apache-2.0) before making this repository public.
