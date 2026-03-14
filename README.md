# Medical-AI-Diagnosis
Research and development of trustworthy AI systems for medical image diagnosis using deep learning, uncertainty estimation, and explainable AI.
Overview

Deep learning has achieved remarkable success in medical image diagnosis, particularly in tasks such as chest X-ray disease detection. However, most deep learning models operate as black boxes, providing only a predicted class label without explaining:

Why the prediction was made

How confident the model is

In healthcare, overconfident incorrect predictions can lead to dangerous decisions and reduce trust in AI systems.

Medical images often contain ambiguity due to noise, patient variability, and rare pathological patterns. Therefore, intelligent diagnostic systems must:

✔ Express predictive uncertainty
✔ Provide interpretable explanations
✔ Detect low-confidence predictions

This project proposes an Uncertainty-Aware Explainable AI system that:

predicts disease from medical images

quantifies prediction reliability

visualizes decision reasoning

flags uncertain cases for expert review

The goal is to transform AI from an autonomous decision maker into a clinical decision support system.

Objectives

The system is designed to achieve the following objectives:

1️⃣ Disease Classification

Develop deep learning models to classify diseases from chest X-ray images.

2️⃣ Uncertainty Estimation

Measure model confidence using stochastic inference techniques.

3️⃣ Explainable AI

Generate visual explanations highlighting image regions responsible for predictions.

4️⃣ Trust Assessment

Combine uncertainty signals and visual explanations to evaluate prediction reliability.

5️⃣ Risk-Aware Decision Support

Automatically flag low-confidence predictions for human expert validation.

System Architecture
Medical Image
     │
     ▼
Image Preprocessing
     │
     ▼
Deep Learning Model
(CNN Architecture)
     │
     ▼
Monte Carlo Dropout
(Stochastic Forward Passes)
     │
     ▼
Prediction Distribution
     │
     ├── Predictive Uncertainty
     │      • Entropy
     │      • Variance
     │
     └── Explainability
            • Grad-CAM Heatmap
     │
     ▼
Decision Engine
     │
     ├── High Confidence → Accept Prediction
     └── High Uncertainty → Refer to Expert
Engineering System
System Components
Component	Description
Input	Chest X-ray medical image
Preprocessing	Image normalization and resizing
CNN Model	Feature extraction and classification
Monte Carlo Dropout	Enables stochastic inference
Uncertainty Estimation	Measures prediction variability
Grad-CAM	Generates explanation heatmaps
Decision Engine	Determines whether to trust or flag prediction
Key Variables
Category	Variables
Input	X-ray image, preprocessing pipeline
Model	CNN weights, dropout layers
Prediction	Class probabilities
Uncertainty	Entropy, predictive variance
Explainability	Heatmap intensity
Decision	Confidence threshold
Important Parameters

T → Number of Monte Carlo inference runs

Dropout probability

Confidence threshold

Heatmap localization threshold

Deep Learning Models

The project evaluates multiple CNN architectures commonly used in medical imaging.

Model	Description
ResNet-18	Baseline architecture used for uncertainty and explainability experiments
ResNet-50	Deeper network for improved feature learning
DenseNet-121	Benchmark model used in medical X-ray studies
EfficientNet-B0	Efficient modern CNN architecture

A comparative study is conducted to determine the best performing model across evaluation metrics.

Uncertainty Estimation

To measure model confidence, we use Monte Carlo Dropout.

Method

Instead of performing a single forward pass, the model performs multiple stochastic forward passes during inference.

Steps:

Enable dropout during testing

Run T forward passes

Collect prediction probabilities

Compute statistical uncertainty measures

Uncertainty Metrics

Predictive Entropy

Predictive Variance

Error-uncertainty correlation

If predictions vary significantly across runs → uncertainty is high.

Explainable AI

To understand model decisions, we use Grad-CAM (Gradient-weighted Class Activation Mapping).

Grad-CAM produces heatmaps highlighting regions in the image that influence the prediction.

Example interpretations:

Heatmap Pattern	Meaning
Focused region in lungs	Model reasoning is valid
Scattered heatmap	Model may be confused

This allows clinicians to verify whether the model is focusing on relevant anatomical regions.

Trust-Aware Decision Engine

The system combines uncertainty estimation and explainability to evaluate prediction reliability.

Condition	System Action
High confidence + focused heatmap	Accept prediction
High uncertainty + scattered heatmap	Flag case for doctor

This ensures the AI system acts as a clinical assistant rather than an autonomous authority.

Model Evaluation

Evaluation follows research-standard metrics used in medical AI studies.

1️⃣ Classification Metrics

These metrics evaluate diagnostic performance.

Accuracy

Precision

Recall / Sensitivity

Specificity

F1 Score

Confusion Matrix

AUC-ROC

2️⃣ Calibration Metrics

Calibration measures whether model confidence reflects true accuracy.

Metrics used:

Expected Calibration Error (ECE)

Brier Score

Reliability Diagram

Predictive Entropy

Predictive Variance

Error-uncertainty correlation

3️⃣ Explainability Evaluation

Explainability quality is assessed using:

Grad-CAM visual inspection

Explanation relevance

Sanity checks

Expert qualitative analysis

4️⃣ Statistical Validation

To ensure robustness:

Cross-validation

Standard deviation across runs

5️⃣ System Performance Metrics

Additional system analysis includes:

Inference time

Model size

Throughput

📊 Comparative Model Analysis

All trained models are compared across evaluation metrics.

Example comparison table:

Model	Accuracy	F1 Score	AUC-ROC	ECE	Parameters
ResNet-18	—	—	—	—	—
ResNet-50	—	—	—	—	—
DenseNet-121	—	—	—	—	—
EfficientNet-B0	—	—	—	—	—

The best performing model is used in the final diagnosis pipeline.

Example Prediction Pipeline
Input X-ray
     ↓
Model Prediction
     ↓
Monte Carlo Dropout
     ↓
Uncertainty Estimation
     ↓
Grad-CAM Heatmap
     ↓
Trust Decision Engine
Example Output
Prediction: Pneumonia
Confidence: 0.82
Uncertainty: Low
Explanation: Heatmap localized in lung opacity region
Decision: Accept Prediction

Or

Prediction: Pneumonia
Confidence: 0.53
Uncertainty: High
Explanation: Scattered heatmap
Decision: Refer to Radiologist

Installation

Clone the repository:

git clone https://github.com/your-org/medical-ai-diagnosis.git
cd medical-ai-diagnosis

Install dependencies:

pip install -r requirements.txt
Running Inference

Example command:

python diagnose_image.py --image sample_xray.png

The system outputs:

predicted class

prediction confidence

uncertainty score

Grad-CAM visualization

Project Structure
medical-ai-diagnosis
│
├── models
│   ├── resnet18
│   ├── resnet50
│   ├── densenet121
│   └── efficientnet
│
├── uncertainty
│   └── mc_dropout.py
│
├── explainability
│   └── grad_cam.py
│
├── evaluation
│   └── metrics.py
│
├── data
│   └── chest_xray_dataset
│
├── train.py
├── inference.py
└── README.md
Team Members
Member	Contribution
Trika	Uncertainty estimation + Grad-CAM
Tanishka	DenseNet training and evaluation
Tanush	ResNet-50 training and optimization
Shubhankar	EfficientNet training and model comparison

Impact

This system improves AI safety in healthcare by:

✔ Quantifying prediction uncertainty
✔ Providing interpretable explanations
✔ Detecting unreliable predictions
✔ Supporting clinicians in medical decision making

 Key Insight

The model does not only predict disease.
It measures its own doubt, explains its reasoning, and knows when to ask for human help.
