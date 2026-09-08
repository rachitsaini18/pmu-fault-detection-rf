# An Explainable Random Forest Framework for Real-Time Transmission Line Fault Detection and Relay Decision/SCADA Support Using PMU Data

A course project for **Machine Learning Techniques (ULC601)**, Electrical & Instrumentation Engineering Department, Thapar Institute of Engineering & Technology, Patiala.

**Authors:** Rachit Saini (102304007), Aashray Sharma (102304037)
**Supervisor:** Dr. Alok Kumar Shukla, Assistant Professor, EIED

## Overview

This project implements an end-to-end machine learning pipeline for automated detection and classification of transmission line faults from Phasor Measurement Unit (PMU) data. A Random Forest classifier is trained on three-phase current and voltage measurements (Ia, Ib, Ic, Va) to distinguish six operational states — Normal A, Normal B, LG Fault, LL Fault, LLG Fault, and Healthy — and is paired with SHAP-based explainability and an interactive SCADA-style monitoring dashboard, all running natively inside a single Google Colab notebook.

**Key results:**
- 87.50% overall validation accuracy
- Perfect F1-scores (1.000) for LG Fault and Healthy classes; F1 ≥ 0.909 for all fault classes
- AUC = 1.000 for LG Fault, LL Fault, LLG Fault, and Healthy on the held-out test set
- 1,999 fault events detected (22.9% fault rate) across 8,736 unseen test samples, at an average model confidence of 89.7%

## Repository contents

```
├── fault_detection.ipynb              Jupyter/Colab notebook (13 cells)
├── ml_framework_fault_detection.py    Same pipeline as a flat script (Colab's own .py export)
├── data/
│   ├── train_dataset.csv              8,640 rows × 11 cols — Jan–Mar 2025
│   ├── test_dataset.csv               8,736 rows × 11 cols — Apr–Jun 2025
│   ├── synthetic_fault_data_2017_2025.csv   510 rows × 17 cols — 2017–2025 feeder history
│   └── dataset.csv                    85,573 rows × 7 cols — raw PMU waveform (11-class labels)
├── results/
│   ├── all_predictions.csv            Model predictions on the full test set
│   └── classification_report.csv      Per-class precision/recall/F1 on the validation split
├── requirements.txt
└── .gitignore
```

`fault_detection.ipynb` and `ml_framework_fault_detection.py` implement the identical 13-cell pipeline — the notebook for interactive/Colab use, the script as Colab's own flat-file export (useful for diffing or running headless).

## Dataset description

| Dataset | File | Rows × Cols | Span |
|---|---|---|---|
| Training | `data/train_dataset.csv` | 8,640 × 11 | Jan–Mar 2025 |
| Test | `data/test_dataset.csv` | 8,736 × 11 | Apr–Jun 2025 |
| Synthetic historical | `data/synthetic_fault_data_2017_2025.csv` | 510 × 17 | 2017–2025 |
| Raw PMU waveform | `data/dataset.csv` | 85,573 × 7 | — |

Training/test class distribution:

| Class | Label | Train Count | Test Count |
|---|---|---|---|
| 0 | Normal A | 980 | 1,028 |
| 1 | Normal B | 1,012 | 981 |
| 2 | LG Fault | 1,394 | 1,414 |
| 3 | LL Fault | 375 | 383 |
| 4 | LLG Fault | 199 | 202 |
| 5 | Healthy | 4,680 | 4,728 |
| — | **TOTAL** | **8,640** | **8,736** |

## Methodology

### Feature engineering & selection
- Six candidate electrical features: `Ia, Ib, Ic, Va, Vb, Vc`
- A preliminary Random Forest (100 trees, depth 15) ranked feature importances: **Ic 30.2%, Ib 29.8%, Va 20.6%, Ia 19.3%**
- Variance Inflation Factor (VIF) analysis found `Vb` and `Vc` highly collinear with `Va` (Pearson r > 0.90, VIF > 10) and excluded them
- Final feature set: **`[Ia, Ib, Ic, Va]`**
- Recursive Feature Elimination (RFE) on the synthetic historical dataset separately selected `Year, Month_Enc, Fault_Duration_s, Fault_Location_km` as most informative for aggregate feeder-level fault trends

### Model training

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Balance of accuracy vs. training time (~45s on Colab CPU) |
| `max_depth` | 20 | Deep enough for 6-class boundaries, avoids underfitting |
| `min_samples_split` | 2 | Default; data is clean/synthetic |
| `min_samples_leaf` | 1 | Fine-grained leaf purity for minority classes |
| `random_state` | 42 | Reproducibility |
| Train/Val split | 80% / 20% | Stratified |
| Scaler | MinMaxScaler | Fit on training partition only, maps to [0, 1] |

## Results

**Validation KPIs**

| Accuracy | Macro F1 | Weighted Precision | Weighted Recall |
|---|---|---|---|
| 87.50% | 0.8019 | 0.8749 | 0.8750 |

**Per-class report** (also in `results/classification_report.csv`)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Normal A | 0.4649 | 0.4388 | 0.4514 | 196 |
| 1 — Normal B | 0.4836 | 0.5099 | 0.4964 | 202 |
| 2 — LG Fault | 1.0000 | 1.0000 | 1.0000 | 279 |
| 3 — LL Fault | 0.9359 | 0.9733 | 0.9542 | 75 |
| 4 — LLG Fault | 0.9459 | 0.8750 | 0.9091 | 40 |
| 5 — Healthy | 1.0000 | 1.0000 | 1.0000 | 936 |

The two Normal A/B steady-state classes are the main source of confusion (they differ only in subtle parameter offsets and carry no safety consequence, since neither is a fault condition). All fault classes and the Healthy state are classified essentially perfectly — this is the primary safety-critical objective, since it means the system does not miss actual faults or raise false trip alarms on healthy operation.

**Test set predictions** (Apr–Jun 2025, 8,736 records — full detail in `results/all_predictions.csv`)

| Class | Count | % |
|---|---|---|
| Healthy | 4,735 | 54.2% |
| LG Fault | 1,414 | 16.2% |
| Normal B | 1,098 | 12.6% |
| Normal A | 904 | 10.3% |
| LL Fault | 392 | 4.5% |
| LLG Fault | 193 | 2.2% |

Average confidence: 89.7% (median 100%).

### Explainability (SHAP)
`shap.TreeExplainer` provides exact per-prediction Shapley values against a 100-sample background set, exposing which of the four input features pushed each prediction toward its predicted class.

### Relay action logic

| Action | Predicted Classes | Interpretation |
|---|---|---|
| **TRIP** ⚠️ | 2, 3, 4 | LG / LL / LLG fault detected — activate protective relay |
| **BLOCK** ✅ | 0, 1, 5 | Normal or Healthy operation — no protective action required |

### SCADA dashboard
An `ipywidgets`-based dashboard (date-range slider, fault-type filter, plot-mode toggle, CSV export) renders fault distribution, frequency-over-time, and raw signal panels entirely within the notebook output.

## Notebook structure

The pipeline runs as 13 sequential cells:

| Cell | Name | Function |
|---|---|---|
| 1 | Install packages | pip-installs `shap` and `statsmodels` |
| 2 | Core imports | Imports, global `STATE` dict, dark plot theme, `hdr()`/`kpi_row()` display helpers |
| 3 | Upload datasets | `google.colab.files.upload()`; auto-routes by filename |
| 4 | Data preview | head(5), class distribution, null checks per dataset |
| 5 | Feature config | Interactive feature/hyperparameter widgets |
| 6 | Feature analysis | Correlation heatmap + VIF; RF importances; RFE |
| 7 | Train RF | `MinMaxScaler` → RF fit → KPI cards, confusion matrix + report |
| 8 | ROC + Learning curve | One-vs-rest ROC; CV=3 learning curve |
| 9 | Select source | Choose prediction target dataset |
| 10 | Predictions | Scale → predict → KPIs, plots, time-trend, table |
| 11 | SHAP XAI | `TreeExplainer`; waveform + SHAP chart; TRIP/BLOCK |
| 12 | SCADA dashboard | Interactive `ipywidgets` monitoring dashboard |
| 13 | Export results | Writes `all_predictions.csv`, `classification_report.csv` |

All intermediate objects (dataframes, model, scaler, features, predictions) are held in a single global `STATE` dict shared across cells within the Colab session.

## Running the notebook

**In Google Colab (recommended, as originally built):**
1. Upload `fault_detection.ipynb` to Colab.
2. Run cells in order (Runtime → Run all, or Shift+Enter through each cell).
3. When prompted in Cell 3, upload the four files from `data/`.

**Locally (Jupyter):**
The notebook uses `google.colab.files` for upload/download, which is Colab-specific. To run locally:
1. `pip install -r requirements.txt`
2. In Cell 3, replace `files.upload()` with local `pd.read_csv('data/train_dataset.csv')`-style calls for each dataset.
3. In Cells 3 and 13, remove or guard the `files.download(...)` calls (they will fail outside Colab) — `to_csv(...)` already writes the files locally, so this is optional.
4. Run the rest of the notebook as-is; all `ipywidgets` cells (5, 9, 12) work in both JupyterLab and classic Jupyter Notebook with the `ipywidgets` extension enabled.

## Discussion — strengths & limitations

**Strengths:** near-perfect fault detection (F1 = 1.000 across fault + healthy classes), high prediction confidence (median 100%), per-prediction SHAP explainability, and a self-contained zero-dependency (beyond pip) execution environment.

**Limitations:** Normal A/B confusion (F1 ≈ 0.45–0.50, safety-irrelevant since neither is a fault state); training data is synthetically generated and has not been validated against field-recorded PMU noise, GPS jitter, or packet loss; the four-feature set is fixed at configuration time rather than dynamically selected; the model classifies fault type but does not estimate fault location.

## Future work
- Hyperparameter optimization (GridSearchCV / Optuna) targeting Normal A/B disambiguation
- Add sequence-component and phase-angle features
- RF + XGBoost/LSTM ensemble to exploit temporal structure
- Fault localization via regression alongside classification
- Robustness testing under injected GPS jitter / measurement noise
- Incremental/online learning for seasonal grid adaptation

## References

1. Sazal, M. M. H. et al. "Ensemble learning based transmission line fault classification using phasor measurement unit (PMU) data with explainable AI (XAI)." *PLOS ONE*, Feb. 2024.
2. "Robust fault detection and classification in power transmission lines via ensemble machine learning models." *Scientific Reports*, Jan. 2025.
3. Jiang, T. et al. "Line Faults Classification Using Machine Learning on Three Phase Voltages Extracted from Large Dataset of PMU Measurements." OSTI Tech. Report, 2022.
4. Faza, A. et al. "Optimal PMU Placement for Fault Classification and Localization Using Enhanced Feature Selection in Machine Learning Algorithms." *Int. Journal of Energy Research*, Wiley, 2024.
5. Jain, P. et al. "Fault Diagnosis in Power Transmission Line using Decision Tree and Random Forest Classifier." IEEE Conf. Publication, 2023.

Full reference list, including dataset/methodology GitHub repositories, is in the project report (Section 13).

## License

Add a license of your choice (e.g. MIT) if you intend this repository to be reused by others.
