# ⚕️ Algorithmic Fairness & Bias Auditing in Healthcare AI

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-FF6B6B)](https://shap.readthedocs.io)
[![AIF360](https://img.shields.io/badge/AIF360-0.6-0062FF)](https://aif360.mybluemix.net)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io)

**A production-grade, research-quality framework for detecting, explaining, and mitigating algorithmic bias in clinical decision support systems.**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Metrics](#-fairness-metrics) • [Results](#-sample-results) • [Dashboard](#-interactive-dashboard) • [Research](#-research-paper)

</div>

---

## 🎯 Overview

Algorithmic bias in healthcare AI is not a hypothetical concern — it is a documented crisis. A landmark 2019 study (Obermeyer et al., *Science*) showed that a widely deployed clinical algorithm systematically underestimated the health needs of Black patients due to a flawed proxy variable. Similar disparities have been documented in sepsis prediction, dermatology AI, and cardiac risk scoring.

This project provides a **complete, end-to-end auditing framework** that enables:

- 🔍 **Detection** — Quantify disparities using five canonical fairness metrics  
- 💡 **Explanation** — Attribute bias to specific features using SHAP  
- 🛡️ **Mitigation** — Apply pre- and post-processing debiasing techniques  
- 📊 **Reporting** — Generate structured audit reports for governance and compliance  

---

## ⚡ Features

| Feature | Description |
|---------|-------------|
| 🧬 Synthetic data generator | Realistic clinical dataset with controllable bias injection |
| 📏 5 fairness metrics | DI, SPD, EOD, AOD, PPD with threshold interpretation |
| 🌲 3 model architectures | Logistic Regression, Random Forest, Gradient Boosting |
| 🔍 SHAP explainability | Global & local explanations + bias attribution |
| 🛡️ 3 mitigation strategies | Reweighing, threshold calibration, adversarial proxy |
| 📊 10+ visualizations | Disparity plots, ROC by group, calibration curves |
| 🖥️ Streamlit dashboard | Interactive bias audit web application |
| 📄 Auto report generation | Markdown audit report for governance |
| ✅ Unit tests | 25+ tests covering metrics, mitigation, and data |
| 🎓 Jupyter notebook | Step-by-step educational walkthrough |

---

## 🏗️ Architecture

```
algorithmic-fairness-audit/
│
├── 📁 src/
│   ├── data/
│   │   ├── generator.py          # Synthetic clinical dataset + bias injection
│   │   └── preprocessor.py       # Encoding, scaling, AIF360 integration
│   │
│   ├── models/
│   │   └── trainer.py            # Model training, evaluation, persistence
│   │
│   ├── fairness/
│   │   ├── metrics.py            # DI, SPD, EOD, AOD, PPD computation
│   │   └── mitigation.py         # Reweighing, threshold calibration, adversarial proxy
│   │
│   ├── explainability/
│   │   └── shap_explainer.py     # SHAP global/local explanations, bias attribution
│   │
│   ├── visualization/
│   │   └── plots.py              # 10+ fairness-specific plots
│   │
│   └── utils/
│       └── reporting.py          # Report generation, JSON persistence
│
├── 📁 scripts/
│   └── run_full_audit.py         # End-to-end pipeline CLI
│
├── 📁 notebooks/
│   └── fairness_audit_demo.ipynb # Interactive educational walkthrough
│
├── 📁 dashboards/
│   └── app.py                    # Streamlit interactive dashboard
│
├── 📁 tests/
│   └── test_fairness_audit.py    # 25+ unit tests
│
└── 📁 reports/                   # Generated audit outputs
    ├── fairness_audit_report.md
    ├── audit_results.json
    └── plots/
```

### Pipeline Flow

```
Raw Data
    │
    ▼
[ClinicalDataGenerator]  ──── bias_strength ────►  Controlled Disparity
    │
    ▼
[ClinicalPreprocessor]   ──── encode/scale ──────►  X_train, X_test, df splits
    │
    ▼
[ClinicalModelTrainer]   ──── fit/evaluate ──────►  Predictions + Probabilities
    │                                                    │
    ▼                                                    ▼
[FairnessAuditor]  ─────────────────────────────►  DI / SPD / EOD / AOD / PPD
    │
    ▼
[SHAPExplainer]    ─────────────────────────────►  Feature Importance + Bias Attribution
    │
    ▼
[Mitigation]       ──┬──  Reweighing (pre)
                     ├──  Threshold Calibration (post)
                     └──  Adversarial Proxy (in)
    │
    ▼
[Comparison]       ─────────────────────────────►  Before vs After Metrics
    │
    ▼
[ReportGenerator]  ─────────────────────────────►  Markdown Report + JSON + Plots
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algorithmic-fairness-audit.git
cd algorithmic-fairness-audit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Run Full Audit (CLI)

```bash
# Default run (Logistic Regression, 5000 samples, bias=0.7)
python scripts/run_full_audit.py

# Custom configuration
python scripts/run_full_audit.py \
    --model random_forest \
    --samples 8000 \
    --bias 0.8 \
    --sensitive race_binary \
    --output-dir reports/

# Skip SHAP for faster execution
python scripts/run_full_audit.py --no-shap
```

### Launch Streamlit Dashboard

```bash
streamlit run dashboards/app.py
# Opens at http://localhost:8501
```

### Run Tests

```bash
pytest tests/ -v --tb=short
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/fairness_audit_demo.ipynb
```

---

## 📐 Fairness Metrics

### Definitions

All metrics compare outcomes between a **privileged group** (e.g., White patients) and an **unprivileged group** (e.g., non-White patients). A **favorable outcome** is "not readmitted" (label = 0).

| Metric | Formula | Fair Zone | Interpretation |
|--------|---------|-----------|----------------|
| **Disparate Impact (DI)** | P(Ŷ=fav \| unpriv) / P(Ŷ=fav \| priv) | [0.80, 1.25] | < 0.8 violates 80% rule |
| **Statistical Parity Diff. (SPD)** | P(Ŷ=pos \| unpriv) − P(Ŷ=pos \| priv) | [−0.10, 0.10] | Negative → model over-predicts risk for unprivileged |
| **Equal Opportunity Diff. (EOD)** | TPR(unpriv) − TPR(priv) | [−0.10, 0.10] | Negative → unprivileged group misses more true positives |
| **Average Odds Diff. (AOD)** | ½[(FPR diff.) + (TPR diff.)] | [−0.10, 0.10] | Combines both error types |
| **Predictive Parity Diff. (PPD)** | PPV(unpriv) − PPV(priv) | [−0.10, 0.10] | Precision gap across groups |

### The 80% Rule

The **Disparate Impact ratio** has a legally recognized threshold from the U.S. Equal Employment Opportunity Commission's **"4/5ths rule"** (1978): a selection rate for any group that is less than 80% of the highest-selected group constitutes evidence of adverse impact. This threshold has been widely adopted as the de facto standard in algorithmic fairness.

```
DI < 0.80  →  ❌ BIASED  (model systematically disadvantages unprivileged group)
0.80 ≤ DI ≤ 1.25  →  ✅ FAIR
DI > 1.25  →  ⚠️  REVERSE BIAS  (model may over-correct)
```

---

## 🛡️ Bias Mitigation

### 1. Reweighing (Pre-processing)
**Stage:** Training data  
**Method:** Assigns importance weights to training samples to balance the joint distribution of (group, label) without modifying the data itself.  
**Best for:** When you have full control over the training pipeline.  
**Reference:** Kamiran & Calders (2012)

```python
from src.fairness.mitigation import CustomReweighing
rw = CustomReweighing(sensitive_col='race_binary')
rw.fit(df_train)
weights = rw.get_weights()
model.fit(X_train, y_train, sample_weight=weights)
```

### 2. Group Threshold Calibration (Post-processing)
**Stage:** Model predictions  
**Method:** Finds individualized decision thresholds per demographic group that maximize F1 while minimizing the Equal Opportunity gap.  
**Best for:** Black-box models where retraining is not possible.  
**Reference:** Hardt, Price & Srebro (2016)

```python
from src.fairness.mitigation import GroupThresholdCalibrator
cal = GroupThresholdCalibrator(sensitive_col='race_binary')
cal.fit(df_test, y_test, y_proba)
y_pred_fair = cal.predict(df_test, y_proba)
```

### 3. Adversarial Debiasing Proxy (In-processing)
**Stage:** Model training  
**Method:** Iteratively reweighs samples based on current fairness gaps, approximating adversarial debiasing without requiring a TensorFlow adversarial network.  
**Best for:** When lightweight in-processing mitigation is desired.

---

## 📊 Sample Results

### Fairness Metrics — Baseline vs Mitigated

| Metric | Before | Status | After | Status | Improvement |
|--------|--------|--------|-------|--------|-------------|
| Disparate Impact | 0.641 | ❌ | 0.872 | ✅ | +36% |
| Statistical Parity Diff. | −0.187 | ❌ | −0.047 | ✅ | +75% |
| Equal Opportunity Diff. | −0.221 | ❌ | −0.082 | ✅ | +63% |
| Average Odds Diff. | −0.194 | ❌ | −0.061 | ✅ | +69% |
| Model Accuracy | 0.784 | — | 0.761 | — | −3% (tradeoff) |

> The small accuracy reduction is the **fairness-accuracy tradeoff** — a fundamental tension in fair ML. The framework makes this tradeoff transparent and configurable.

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides a fully interactive bias audit experience:

- **Dataset Tab**: Upload your own CSV or generate synthetic data with adjustable bias
- **Model Tab**: Train and evaluate classification models, view feature importances
- **Fairness Tab**: Run full fairness audit, view metrics, ROC curves by group
- **Explainability Tab**: Compute SHAP values, view bias attribution analysis
- **Mitigation Tab**: Apply mitigation, compare before/after, download audit report

```bash
streamlit run dashboards/app.py
```

---

## 🔬 Healthcare Relevance

This framework targets real-world clinical AI deployment scenarios including:

| Use Case | Sensitive Attributes | Fairness Concern |
|----------|---------------------|-----------------|
| Hospital readmission prediction | Race, insurance, ZIP | Algorithm may underserve low-income or minority patients |
| Sepsis early warning | Race, age | Delayed alerts for certain demographics |
| Cardiac risk scoring | Sex, race | Documented gender and racial bias in standard calculators |
| Mental health triage | Race, insurance | Disparate access to mental healthcare resources |
| Organ transplant priority | Race, geography | Geographic and socioeconomic disparities |

---

## 📖 Research Paper

See [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) for a full mini research paper including:
- Systematic literature review of healthcare AI bias
- Methodology and experimental setup
- Quantitative results and fairness analysis
- Discussion of limitations and ethical considerations

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/ -v -k "TestFairnessMetrics"

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

Test coverage includes:
- Data generation correctness and reproducibility
- Fairness metric mathematical accuracy
- Bias mitigation weight computation
- Model training, evaluation, save/load
- Preprocessing pipeline integrity

---

## 🗺️ Future Work

- [ ] **Intersectional fairness** — Audit compound sensitive attributes (race × gender)
- [ ] **Causal fairness** — Counterfactual fairness via causal graphs
- [ ] **Temporal drift monitoring** — Track fairness metrics over time in production
- [ ] **Federated auditing** — Privacy-preserving fairness analysis across institutions
- [ ] **FHIR integration** — Direct connection to healthcare data standards
- [ ] **Regulatory compliance** — EU AI Act, ONC HTI-1 compliance reporting

---

## 📚 References

1. Obermeyer, Z. et al. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
2. Kamiran, F. & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1-33.
3. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
4. Bellamy, R. et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. *IBM Journal of R&D*, 63(4/5).
5. Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS*.
6. Mehrabi, N. et al. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6).

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/causal-fairness`)
3. Write tests for new functionality
4. Submit a pull request with a clear description

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the research community with care.**  
*If this project aids your work, please cite it and star ⭐ the repository.*

</div>
