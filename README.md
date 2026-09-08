# Trustworthy TinyML for IoMT: Explanation-Fidelity Trade-offs in Compressed Cardiac Risk Screening Models

Code, experiments, and reproducibility artifacts for the paper:

> Md Irfanul Kabir Hira, Anichur Rahman, Md Shohel Rana. **"Trustworthy TinyML for IoMT: Explanation-Fidelity Trade-offs in Compressed Cardiac Risk Screening Models."**

This repository quantifies whether SHAP-based explanations survive model compression (float16, int8, and magnitude-pruning + int8 quantization) when a deep neural network (DNN) for cardiac risk screening is deployed to constrained IoMT edge hardware.

---

## 1. Overview

Wearable and bedside IoMT devices for cardiac risk screening must run on constrained edge hardware, which typically requires compressing deep learning models via quantization or pruning. Prior TinyML-for-health work benchmarks compression on **accuracy, latency, and memory footprint alone** — leaving open whether a compressed model's *explanations* still match those of its full-precision counterpart.

This project addresses that gap by:

1. Training a deployment-target DNN alongside classical baselines (**SVM, Random Forest, KNN**) on the IEEE DataPort heart disease dataset (D1, n = 1,190).
2. Compressing the DNN into **four TensorFlow Lite variants**: float32 (reference), float16, int8, and pruned + int8.
3. Benchmarking each variant on **accuracy, model size, and inference latency**.
4. Quantifying **explanation-fidelity drift** between the float32 reference and each compressed variant using:
   - Top-k SHAP feature-rank agreement (Jaccard index)
   - Spearman rank correlation of SHAP attribution vectors
   - McNemar's paired significance test on predictions
   - Multi-seed variance estimation (95% CI)
5. Cross-checking the DNN's global SHAP feature ranking against Random Forest's independent feature importances for clinical plausibility.
6. Deriving a **three-tier IoMT edge–cloud architecture** that pairs a fidelity-optimized edge model with cloud-based full-precision confirmation for high-risk cases.

### Key finding

Explanation fidelity depends more on **how** a model is compressed than on model size alone:

| Compression | Spearman ρ (vs. float32) | Top-k Jaccard |
|---|---|---|
| float16 | ≈ 0.87 | 0.73 |
| int8 | ≈ 0.86 | 0.72 |
| **pruned + int8** | **≈ 0.98** | **0.93** |

Combining magnitude pruning with int8 quantization preserves SHAP explanation fidelity substantially better than quantization alone, at a comparable model footprint — even though it does **not** have the best raw accuracy.

---

## 2. Repository Structure

```
.
├── data/
│   ├── raw/                     # Original IEEE DataPort heart disease dataset (D1)
│   └── processed/                # Imputed / standardized train-test splits
├── src/
│   ├── preprocessing.py          # Imputation, standardization, stratified splitting
│   ├── train_baselines.py        # SVM / Random Forest / KNN training + grid search
│   ├── train_dnn.py               # DNN hyperparameter search, class-weighted training
│   ├── compress_tflite.py         # float32 / float16 / int8 / pruned+int8 conversion
│   ├── benchmark_edge.py          # Latency & model-size benchmarking (cloud CPU proxy)
│   ├── edge_benchmark_device.py   # Standalone script for Raspberry Pi / ESP32 execution
│   ├── shap_fidelity.py           # SHAP computation, Jaccard, Spearman, McNemar's test
│   ├── multiseed_variance.py      # Multi-seed pipeline repetition + CI estimation
│   └── plotting.py                # Trade-off, ROC, confusion matrix, SHAP figures
├── notebooks/
│   └── analysis.ipynb             # End-to-end exploratory walkthrough
├── results/
│   ├── tables/                    # Table 1 & Table 2 outputs (CSV)
│   ├── figures/                   # Fig. 1–3 reproductions
│   └── reproducibility_manifest.json  # Seeds, hyperparameters, pruning sparsity, library versions
├── requirements.txt
├── LICENSE
└── README.md
```

*(Adjust paths above to match your actual repository layout before publishing.)*

---

## 3. Dataset

**IEEE DataPort Heart Disease Dataset (D1)**
Combines the Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog heart disease cohorts.

- **Samples:** 1,190 patient records
- **Features:** 11 clinical features (age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, oldpeak, ST slope)
- **Target:** Binary presence/absence of heart disease
- **Source:** Siddhartha, M. *Heart Disease Dataset (Comprehensive)*. IEEE DataPort. https://doi.org/10.21227/dz4t-cm36

> The dataset is not redistributed in this repository. Download it directly from IEEE DataPort and place it under `data/raw/`.

---

## 4. Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Core dependencies

- Python 3.10+
- TensorFlow / Keras
- TensorFlow Lite
- `tensorflow-model-optimization` (magnitude pruning)
- scikit-learn
- SHAP
- `tflite-runtime` (for physical edge-hardware benchmarking)
- NumPy, pandas, matplotlib, scipy

---

## 5. Usage

```bash
# 1. Preprocess data (imputation + standardization + stratified split)
python src/preprocessing.py --input data/raw/D1.csv --output data/processed/

# 2. Train classical baselines
python src/train_baselines.py --data data/processed/

# 3. Train and tune the deployment-target DNN
python src/train_dnn.py --data data/processed/ --config configs/dnn_search.yaml

# 4. Compress the trained DNN into four TFLite variants
python src/compress_tflite.py --model artifacts/dnn_float32.keras --output artifacts/tflite/

# 5. Benchmark latency & size (cloud-CPU proxy)
python src/benchmark_edge.py --variants artifacts/tflite/

# 5b. Benchmark on physical edge hardware (Raspberry Pi / ESP32)
python src/edge_benchmark_device.py --variants artifacts/tflite/

# 6. Compute SHAP explanation-fidelity metrics
python src/shap_fidelity.py --reference artifacts/tflite/float32.tflite --variants artifacts/tflite/

# 7. Multi-seed variance estimation
python src/multiseed_variance.py --seeds 5 --config configs/pipeline.yaml

# 8. Generate figures / tables
python src/plotting.py --results results/
```

---

## 6. Methodology Summary

| Stage | Description |
|---|---|
| 1. Data acquisition & preprocessing | Zero-value imputation (cholesterol, resting BP), z-score standardization |
| 2. Stratified train/test split | 80/20 split, seed-controlled, class ratio preserved |
| 3. Model training | SVM / RF / KNN (GridSearchCV) + DNN (hyperparameter search, class-weighted BCE, batch norm, dropout) |
| 4. TFLite compression | float32 (ref), float16, int8, pruned + int8 |
| 5. Edge benchmarking | Latency & size — cloud-CPU proxy, then Raspberry Pi / ESP32 |
| 6. Explainability & fidelity | SHAP top-k Jaccard, Spearman ρ, McNemar's test, multi-seed CI |
| 7. Trade-off & plausibility analysis | Accuracy vs. size vs. fidelity (Pareto plot); DNN-SHAP vs. RF importance cross-check |
| 8. Manuscript artifacts | Result tables, trade-off figure, reproducibility manifest |

---

## 7. Results

Full results, figures, and tables are provided in the paper (Sections 4 and Tables 1–2) and reproduced under `results/`. Headline numbers:

- **DNN (float32) AUC:** 0.930 | **Accuracy:** 0.882
- **Random Forest AUC:** 0.980 | **Accuracy:** 0.933 (best raw accuracy, not the deployment target)
- **float16 / int8:** accuracy and AUC closely track the float32 reference; moderate SHAP fidelity drift (ρ ≈ 0.86–0.87)
- **pruned + int8:** lower raw accuracy (0.819) but substantially higher explanation fidelity (ρ ≈ 0.98) at a comparable footprint

Top SHAP-ranked features — **oldpeak, chest pain type, ST slope** — align with Random Forest's independent feature ranking and established clinical cardiac-risk indicators.

---

## 8. Sources / References

The full reference list corresponding to the paper's related-work and comparison sections:

1. Ashfaq, M.T., Javaid, N., Alrajeh, N., Ali, S.S. *An explainable AI based new deep learning solution for efficient heart disease prediction at early stages.* Evolving Systems 16(1), 33 (2025).
2. Cenitta, D., Arul, N., Arjunan, R.V., Chadaga, K., Andrew, J. *An explainable artificial intelligence framework for ischemic heart disease prediction using enhanced squirrel search feature selection.* Scientific Reports (2026).
3. Eshwarappa, N.M., Baghban, H., Hsu, C.H., Hsu, P.Y., Hwang, R.H., Chen, M.Y. *Communication-efficient and privacy-preserving federated learning for medical image classification in multi-institutional edge computing.* Journal of Cloud Computing 14(1), 44 (2025).
4. Essahraui, S., Lamaakal, I. *A comprehensive survey of TinyML-based biometric recognition for IoT edge devices.* IEEE Internet of Things Journal (2026).
5. Gogi, G., Gurung, S., Gegov, A., Arabikhan, F., Ichtev, A. *Trustworthy and reliable AI for heart disease diagnosis.* IJCNN 2025, pp. 1–7. IEEE.
6. Ivanov, D.A., Larionov, D.A., Maslennikov, O.V., Voevodin, V.V. *Neural network compression for reinforcement learning tasks.* Scientific Reports 15(1), 9718 (2025).
7. Keivanimehr, A.R., Akbari, M. *TinyML and edge intelligence applications in cardiovascular disease: A survey.* Computers in Biology and Medicine 186, 109653 (2025).
8. Khalid, M.I., Hussain, A., Hussain, N., Alkhalifah, T. *Lightweight and interpretable edge intelligence AI with intrusion detection for trustworthy cardiac arrhythmia in medical IoT.* Scientific Reports (2026).
9. Khan, M.A., Saudagar, A.K.J., Yaqoob, M.M., Nazir, M., Yousafzai, A., Khaliq uz Zaman, S., Alkhrijah, Y.M., Mazhar, T. *Federated learning for heart disease detection and classification in edge enabled IoMT-based healthcare.* Computing 107(11), 219 (2025).
10. Khan, S., Perumal, K., Alsolai, H., Aljohani, A. *FedTinyMed: Federated learning enabled tiny multi-task machine learning model for smart healthcare monitoring for IoMT.* Computers and Electrical Engineering 128, 110761 (2025).
11. Salih, A.M., Galazzo, I.B., Gkontra, P., Rauseo, E., Lee, A.M., Lekadir, K., Radeva, P., Petersen, S.E., Menegaz, G. *A review of evaluation approaches for explainable AI with applications in cardiology.* Artificial Intelligence Review 57(9), 240 (2024).
12. Sen, J., Bhattacharya, S. *XAI in heart disease: A review of concepts, applications and limitations.* Archives of Computational Methods in Engineering, pp. 1–35 (2026).
13. Siddhartha, M. *Heart Disease Dataset (Comprehensive).* IEEE DataPort (2020). https://doi.org/10.21227/dz4t-cm36
14. Talukder, M.A., Talaat, A.S., Kazi, M., Khraisat, A. *XAI-HD: an explainable artificial intelligence framework for heart disease detection.* Artificial Intelligence Review 58(12), 385 (2025).
15. Umar, M.A., Abuali, N., Shuaib, K., Awad, A.I. *An explainable artificial intelligence and Internet of Things framework for monitoring and predicting cardiovascular disease.* Engineering Applications of Artificial Intelligence 144, 110138 (2025).
16. Vani, M.S., Sudhakar, R.V., Mahendar, A., Ledalla, S., Radha, M., Sunitha, M. *Personalized health monitoring using explainable AI: bridging trust in predictive healthcare.* Scientific Reports 15(1), 31892 (2025).
17. Wang, Z., Hu, Y., Hu, Q., Bai, D. *Explainable AI-enabled wearable sensors for real-time thrombotic event early warning in connected health environments.* IEEE Transactions on Consumer Electronics (2026).
18. Xi, L., Li, C., Anari, M.S., Rezaee, K. *Integrating wearable health devices with AI and edge computing for personalized rehabilitation.* Journal of Cloud Computing 14(1), 64 (2025).

---

## 9. Limitations

- D1 is retrospective, tabular clinical data — **not** a live physiological sensor stream; this work isolates the compression-and-explainability question as a precursor to sensor-based validation.
- Edge latency/size benchmarks were obtained primarily via a cloud-CPU proxy; full-scale physical Raspberry Pi / ESP32 validation is packaged as a standalone script but not exhaustively reported.
- The deployment-target DNN's raw accuracy trails Random Forest, reflecting a deliberate choice to prioritize a model family with a direct quantization/pruning pathway.
- Fidelity was evaluated using SHAP only and magnitude pruning as the sole pruning strategy; generalization to other explainers (LIME, Integrated Gradients) or structured compression (distillation, channel pruning) is left to future work.

---

## 10. Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{hira2026trustworthy,
  title     = {Trustworthy TinyML for IoMT: Explanation-Fidelity Trade-offs in Compressed Cardiac Risk Screening Models},
  author    = {Hira, Md Irfanul Kabir and Rahman, Anichur and Rana, Md Shohel},
  year      = {2026}
}
```

---

## 11. License

Specify your license here (e.g., MIT, Apache 2.0). Add a `LICENSE` file to the repository root.

## 12. Contact

- Md Irfanul Kabir Hira — irfanhira11@niter.edu.bd
- Anichur Rahman — ar36248@georgiasouthern.edu
- Md Shohel Rana (Corresponding author) — mrana@georgiasouthern.edu / gudla@cse.msstate.edu
