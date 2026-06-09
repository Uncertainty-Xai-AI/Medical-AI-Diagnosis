# 🫁 MedXAI — Reliable Medical AI Through Explainability & Uncertainty

> An Explainable and Uncertainty-Aware AI System for Chest X-Ray Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)]()
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20AI-green)]()
[![XAI](https://img.shields.io/badge/XAI-HiResCAM-orange)]()
[![Research](https://img.shields.io/badge/Research-Medical%20AI-purple)]()

---

## 🚀 Why MedXAI?

Most medical AI systems behave like black boxes.

They provide a diagnosis but fail to answer three critical clinical questions:

* **Why was this prediction made?**
* **How confident is the model?**
* **Should a doctor trust this result?**

MedXAI addresses all three by combining:

* 🧠 Deep Learning
* 🔍 Explainable AI (HiResCAM)
* 📊 Uncertainty Quantification (MC Dropout)
* 🤝 Human-in-the-Loop Decision Support

Instead of replacing clinicians, MedXAI helps clinicians make safer decisions.

---

## 🎯 Project Goals

* Detect pneumonia from chest X-ray images
* Quantify prediction uncertainty
* Visualize model reasoning
* Automatically flag unreliable predictions
* Build a trustworthy clinical decision-support workflow

---

## 🏗 System Pipeline

```text
Chest X-Ray
     │
     ▼
Image Preprocessing
(Resize + Normalize)
     │
     ▼
Ensemble CNN Models
├── ResNet-18
├── ResNet-50
├── DenseNet-121
└── EfficientNet-B0
     │
     ▼
MC Dropout
(T = 30 stochastic passes)
     │
     ▼
Prediction Distribution
     │
     ├── Uncertainty Estimation
     │      ├ Predictive Entropy
     │      └ Predictive Variance
     │
     └── Explainability
            └ HiResCAM Heatmaps
     │
     ▼
Decision Engine
     │
     ├── Reliable → Accept
     └── Uncertain → Refer to Expert
```

---

## 🧠 Model Architecture

| Model           | Purpose                                      |
| --------------- | -------------------------------------------- |
| ResNet-18       | Lightweight baseline                         |
| ResNet-50       | Deep residual feature extraction             |
| DenseNet-121    | Medical imaging benchmark (CheXNet backbone) |
| EfficientNet-B0 | Parameter-efficient high-performance model   |

### Ensemble Strategy

Weighted soft-voting ensemble using validation AUC scores.

Each model contributes proportionally to its performance.

This improves:

* Robustness
* Generalization
* Clinical reliability

---

## 🔬 Explainability with HiResCAM

Traditional CNNs provide no insight into their decisions.

MedXAI uses HiResCAM to highlight image regions influencing predictions.

### Example

✅ Focused activation in lung opacity region

→ Reliable reasoning

⚠ Activation scattered outside lungs

→ Potential model confusion

→ Flag for review

---

## 📊 Uncertainty Quantification

MedXAI estimates predictive uncertainty using Monte Carlo Dropout.

### Process

1. Enable dropout during inference
2. Perform 30 stochastic forward passes
3. Collect probability distributions
4. Compute:

* Predictive Mean
* Predictive Entropy
* Predictive Variance

### Interpretation

| Entropy   | Meaning              |
| --------- | -------------------- |
| Low       | Confident prediction |
| High      | Uncertain prediction |
| Very High | Refer to radiologist |

The system learns not only when it is correct, but also when it might be wrong.

---

## 📈 Results

### Individual Model Performance

| Model           | Accuracy | Precision | Recall | F1    | AUC   |
| --------------- | -------- | --------- | ------ | ----- | ----- |
| ResNet-18       | 93.4%    | 93.8%     | 94.6%  | 94.2% | 0.971 |
| ResNet-50       | 94.1%    | 94.5%     | 95.2%  | 94.8% | 0.978 |
| DenseNet-121    | 93.6%    | 94.0%     | 94.9%  | 94.4% | 0.974 |
| EfficientNet-B0 | 94.7%    | 95.1%     | 95.8%  | 95.4% | 0.981 |

### Final Ensemble Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 96.2% |
| Precision | 96.8% |
| Recall    | 97.1% |
| F1 Score  | 96.9% |
| AUC-ROC   | 0.991 |

---

## 🚦 Reliability Decision Engine

| Condition                         | Action                |
| --------------------------------- | --------------------- |
| Low Uncertainty + Focused Heatmap | ✅ Reliable Prediction |
| High Uncertainty                  | ⚠ Refer to Expert     |
| Abnormal Heatmap Localization     | ⚠ Refer to Expert     |

This transforms AI into a clinical assistant rather than an autonomous decision-maker.

---

## 💻 Technology Stack

### Deep Learning

* PyTorch
* Torchvision

### Explainability

* HiResCAM
* Grad-CAM

### Uncertainty

* Monte Carlo Dropout
* Predictive Entropy

### Data Science

* NumPy
* Pandas
* Scikit-Learn

### Visualization

* OpenCV
* Matplotlib

### Deployment

* Lovable
* React Frontend

---

## 📂 Repository Structure

```text
Medical-AI-Diagnosis
│
├── models/
│   ├── resnet18.py
│   ├── resnet50.py
│   ├── densenet121.py
│   └── efficientnet.py
│
├── uncertainty/
│   └── mc_dropout.py
│
├── explainability/
│   └── hirescam.py
│
├── ensemble/
│   └── weighted_voting.py
│
├── app/
│   └── medxai_vision
│
├── results/
│   ├── metrics
│   ├── confusion_matrices
│   └── heatmaps
│
└── README.md
```

---

## 👥 Team

| Member          | Contribution                                             |
| --------------- | -------------------------------------------------------- |
| Tanishka Pal    | DenseNet-121, calibration analysis, medical metrics      |
| Trika Jaiswal   | MC Dropout, uncertainty estimation, reliability analysis |
| Tanush Kumar    | ResNet-50, optimization, class balancing                 |
| Shubhankar Bhan | EfficientNet-B0, comparative evaluation, integration     |

### Faculty Mentors

* Dr. Priyanka Deshmukh
* Dr. Hema Karande

Symbiosis Institute of Technology, Pune

---

## 🌟 Key Contribution

> MedXAI does not simply predict disease.

It explains its reasoning, quantifies its uncertainty, and knows when to ask for human help.

This is a step toward trustworthy AI in healthcare.

---

## 📚 References

1. Rajpurkar et al. — CheXNet (2017)
2. Gal & Ghahramani — MC Dropout (2016)
3. Selvaraju et al. — Grad-CAM (2017)
4. Leibig et al. — Uncertainty in Medical AI (2017)
5. Begoli et al. — Need for Uncertainty Quantification (2019)
