# 🫁 Medical-AI-Diagnosis

> **An Uncertainty-Aware and Explainable AI System for Reliable Medical Image Diagnosis**

Research and development of trustworthy AI systems for chest X-ray diagnosis using deep learning, uncertainty estimation, and explainable AI.

---

## 📌 Overview

Deep learning has achieved remarkable success in medical image diagnosis. However, most models operate as **black boxes** — providing only a predicted class label without explaining:

- *Why* the prediction was made
- *How confident* the model actually is

In healthcare, overconfident wrong predictions can lead to dangerous clinical decisions and erode trust in AI systems.

This project proposes an **Uncertainty-Aware Explainable AI (XAI) system** that:

- ✅ Predicts disease from chest X-ray images
- ✅ Quantifies prediction reliability using uncertainty estimation
- ✅ Visualizes decision reasoning using Grad-CAM heatmaps
- ✅ Automatically flags uncertain cases for expert review

The goal is to transform AI from an autonomous decision-maker into a **clinical decision support system**.

---

## 🎯 Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | Disease Classification | Classify chest X-rays into NORMAL vs PNEUMONIA |
| 2 | Uncertainty Estimation | Measure model confidence using MC Dropout |
| 3 | Explainable AI | Generate Grad-CAM heatmaps for visual reasoning |
| 4 | Trust Assessment | Combine uncertainty + explainability for reliability scoring |
| 5 | Risk-Aware Decision Support | Flag low-confidence predictions for human review |

---

## 🏗️ System Architecture

```
Medical X-ray Image
        │
        ▼
  Image Preprocessing
  (Resize, Normalize)
        │
        ▼
  Deep Learning Model
  (CNN Architecture)
        │
        ▼
  Monte Carlo Dropout
  (T stochastic forward passes)
        │
        ▼
  Prediction Distribution
        │
        ├──── Uncertainty Estimation
        │         • Predictive Entropy
        │         • Predictive Variance
        │
        └──── Explainability
                  • Grad-CAM Heatmap
        │
        ▼
   Decision Engine
        │
        ├── High Confidence + Valid Heatmap → ✅ Accept Prediction
        └── High Uncertainty + Scattered Heatmap → ⚠️ Refer to Expert
```

---

## 🤖 Models

| Model | Description | Status |
|-------|-------------|--------|
| ResNet-18 | Baseline — uncertainty & explainability experiments | ✅ Phase 1 |
| ResNet-50 | Deeper network for improved feature learning | 🔄 Phase 2 |
| DenseNet-121 | Benchmark model for medical X-ray studies (CheXNet) | 🔄 Phase 2 |
| EfficientNet-B0 | Modern efficient CNN architecture | 🔄 Phase 2 |

---

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1 Score, AUC-ROC, Confusion Matrix

### Calibration
- **ECE** (Expected Calibration Error) — lower is better
- Reliability Diagram
- Confidence Distribution

### Uncertainty Analysis
- Predictive Entropy
- Predictive Variance
- Error-uncertainty correlation

### Explainability
- Grad-CAM visual inspection
- Heatmap localization quality

---

## 📈 Model Comparison

| Model | Accuracy | F1 Score | AUC-ROC | ECE | Sensitivity | Specificity |
|-------|----------|----------|---------|-----|-------------|-------------|
| ResNet-18 | — | — | — | — | — | — |
| ResNet-50 | — | — | — | — | — | — |
| DenseNet-121 | — | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — | — |

*Table will be updated as Phase 2 results come in.*

---

## 🔬 Uncertainty Estimation — MC Dropout

Instead of a single forward pass, the model runs **T stochastic forward passes** with dropout enabled at inference time.

```
1. Enable dropout layers during testing
2. Run T = 30 forward passes on the same image
3. Collect T prediction probability vectors
4. Compute mean prediction → final output
5. Compute entropy/variance → uncertainty score
```

**Interpretation:**
- Low variance across T passes → model is **confident**
- High variance across T passes → model is **uncertain** → flag for review

---

## 🗺️ Grad-CAM Explainability

Grad-CAM generates heatmaps highlighting which regions of the X-ray influenced the prediction.

| Heatmap Pattern | Interpretation |
|-----------------|----------------|
| Focused on lung region | ✅ Model reasoning is valid |
| Scattered across image | ⚠️ Model may be confused — flag for review |

---

## 🚦 Decision Engine

| Condition | Action |
|-----------|--------|
| Low uncertainty + focused heatmap | ✅ Accept prediction |
| High uncertainty OR scattered heatmap | ⚠️ Flag — refer to radiologist |

---

## 💡 Example Output

```
========== AI Diagnosis Report ==========
Prediction   : PNEUMONIA
Confidence   : 0.82
Uncertainty  : 0.09 (Low)
Heatmap      : Localized in lung opacity region
Decision     : ✅ Reliable — supports clinician decision

─────────────────────────────────────────

Prediction   : PNEUMONIA
Confidence   : 0.53
Uncertainty  : 0.67 (High)
Heatmap      : Scattered — no clear focus
Decision     : ⚠️ FLAGGED — refer to radiologist
```

---

## 📁 Project Structure

```
Medical-AI-Diagnosis/
│
├── phase1/
│   └── resnet18_phase1.py          # Baseline model + MC Dropout + Grad-CAM
│
├── phase2/
│   ├── densenet121_phase2.py       # DenseNet training + calibration (Tanishka)
│   ├── resnet50_phase2.py          # ResNet-50 training (Tanush)
│   ├── efficientnet_phase2.py      # EfficientNet training (Shubhankar)
│   └── model_comparison.py        # Cross-model comparison table
│
├── results/
│   └── metrics_comparison.csv     # Final numbers from all models
│
└── README.md
```

---

## ⚙️ Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Uncertainty-Xai-AI/Medical-AI-Diagnosis.git
cd Medical-AI-Diagnosis
```

**2. Install dependencies**
```bash
pip install torch torchvision scikit-learn matplotlib Pillow tqdm
```

**3. Download dataset**

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```python
# In Google Colab
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip -d chest_xray_data
```

**4. Run a model**
```bash
python phase2/densenet121_phase2.py
```

---

## 👥 Team

| Member | Model | Contribution |
|--------|-------|--------------|
| Trika | ResNet-18 | Uncertainty estimation, Grad-CAM, reliability analysis |
| Tanishka | DenseNet-121 | Model training, calibration analysis, medical metrics |
| Tanush | ResNet-50 | Model training, optimization, class imbalance fix |
| Shubhankar | EfficientNet-B0 | Model training, comparative analysis, final pipeline |

**Guided by:** Dr. Priyanka Deshmukh & Dr. Hema Karande
**Institution:** Symbiosis Institute of Technology, Pune

---

## 🌟 Key Insight

> *The model does not only predict disease.*
> *It measures its own doubt, explains its reasoning, and knows when to ask for human help.*

---

## 📚 References

1. Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection," Stanford, 2017
2. Gal & Ghahramani, "Dropout as a Bayesian Approximation," ICML 2016
3. Selvaraju et al., "Grad-CAM," ICCV 2017
4. Leibig et al., "Leveraging Uncertainty from Deep Neural Networks," Scientific Reports, 2017
5. Begoli et al., "The need for uncertainty quantification in medical AI," Nature MI, 2019
