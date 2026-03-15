# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, classification_report
)
from sklearn.calibration import calibration_curve

from tqdm import tqdm

print("All imports successful!")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DEVICE + TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Same transforms as Phase 1 for consistency across all models
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # DenseNet also uses 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])   # ImageNet normalization
])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

data_dir = "chest_xray_data/chest_xray"

full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=transform)

# 80/20 train-val split (same as Phase 1)
train_size = int(0.8 * len(full_train))
val_size   = len(full_train) - train_size
train_data, val_data = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

class_names = full_train.classes  # ['NORMAL', 'PNEUMONIA']
print("Classes:", class_names)
print(f"Train size: {train_size} | Val size: {val_size} | Test size: {len(test_data)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CLASS IMBALANCE — WEIGHTED LOSS
# ─────────────────────────────────────────────────────────────────────────────

# The dataset has ~3x more PNEUMONIA than NORMAL images.
# Without correction, the model tends to over-predict PNEUMONIA.
# We give NORMAL class a higher weight so the model pays more attention to it.

# Count samples per class in training set
train_labels = [full_train.targets[i] for i in train_data.indices]
normal_count    = train_labels.count(0)  # class 0 = NORMAL
pneumonia_count = train_labels.count(1)  # class 1 = PNEUMONIA
total_count     = len(train_labels)

# Weight = total / (num_classes * count_of_class)
weight_normal    = total_count / (2 * normal_count)
weight_pneumonia = total_count / (2 * pneumonia_count)

class_weights = torch.tensor([weight_normal, weight_pneumonia]).to(device)
print(f"Class counts — NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")
print(f"Weights      — NORMAL: {weight_normal:.4f}, PNEUMONIA: {weight_pneumonia:.4f}")

# Weighted loss criterion — this is the key fix vs Phase 1
criterion = nn.CrossEntropyLoss(weight=class_weights)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DENSENET-121 MODEL
# ─────────────────────────────────────────────────────────────────────────────

# DenseNet-121 connects each layer to every other layer in a feed-forward
# fashion. This helps with vanishing gradients and feature reuse.
# It is the architecture used in CheXNet (chest X-ray benchmark paper).

model = models.densenet121(weights="IMAGENET1K_V1")

# DenseNet's final classifier is model.classifier (not model.fc like ResNet)
num_ftrs = model.classifier.in_features  # 1024 for DenseNet-121

model.classifier = nn.Sequential(
    nn.Dropout(0.5),          # Dropout for regularization (also needed for MC Dropout later)
    nn.Linear(num_ftrs, 2)    # Binary: NORMAL vs PNEUMONIA
)

model = model.to(device)
print("DenseNet-121 loaded and modified for binary classification.")

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Same lr as Phase 1

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, device):
    """Train the model for one epoch. Returns avg loss and accuracy."""
    model.train()
    running_loss, running_corrects, total = 0, 0, 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss     += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        total            += labels.size(0)

    return running_loss / len(loader), (running_corrects.double() / total).item()


def evaluate(model, loader, device):
    """Evaluate the model. Returns avg loss and accuracy."""
    model.eval()
    running_loss, running_corrects, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss     += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            total            += labels.size(0)

    return running_loss / len(loader), (running_corrects.double() / total).item()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

epochs = 5
best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc   = 0.0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(epochs):
    print(f"\n========== Epoch {epoch+1}/{epochs} ==========")

    train_loss, train_acc = train_one_epoch(model, train_loader, device)
    val_loss,   val_acc   = evaluate(model, val_loader, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print("✅ Best model saved!")

# Load best weights back into model
model.load_state_dict(best_model_wts)
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(history["val_loss"],   label="Val Loss",   marker="o")
axes[0].set_title("DenseNet-121 — Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(history["val_acc"],   label="Val Acc",   marker="o")
axes[1].set_title("DenseNet-121 — Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("densenet_training_curves.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: COLLECT TEST PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

# We collect labels, predicted classes, and probabilities for all test images.
# These are reused across all evaluation sections below.

model.eval()

all_labels = []  # True labels
all_preds  = []  # Predicted class indices
all_probs  = []  # Softmax probabilities for PNEUMONIA class (class 1)

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Collecting test predictions"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        preds   = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(PNEUMONIA)

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

print(f"Test predictions collected. Total samples: {len(all_labels)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: MEDICAL EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

precision   = precision_score(all_labels, all_preds)
recall      = recall_score(all_labels, all_preds)      # Sensitivity
f1          = f1_score(all_labels, all_preds)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc         = roc_auc_score(all_labels, all_probs)

print("\n========== DenseNet-121 Medical Evaluation Metrics ==========")
print(f"Confusion Matrix:\n{cm}")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"\nPrecision        : {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity       : {specificity:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"AUC-ROC           : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


# ── Confusion Matrix Heatmap ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title("DenseNet-121 — Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("densenet_confusion_matrix.png", dpi=150)
plt.show()


# ── ROC Curve ─────────────────────────────────────────────────────────────────

fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"DenseNet-121 (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("DenseNet-121 — ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("densenet_roc_curve.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: CALIBRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Calibration answers: "When the model says 80% confident, is it right 80% of the time?"
# A well-calibrated model follows the diagonal line on the reliability diagram.
# ECE (Expected Calibration Error) quantifies how far off the calibration is.

# ── 11a. Reliability Diagram ─────────────────────────────────────────────────

# sklearn's calibration_curve bins predictions and computes mean confidence
# vs actual fraction of positives per bin
fraction_of_positives, mean_predicted_value = calibration_curve(
    all_labels, all_probs, n_bins=10, strategy="uniform"
)

plt.figure(figsize=(7, 6))
plt.plot(mean_predicted_value, fraction_of_positives,
         "s-", color="steelblue", label="DenseNet-121", linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1],
         "k--", label="Perfect calibration", linewidth=1.5)

plt.fill_between(mean_predicted_value, fraction_of_positives, mean_predicted_value,
                 alpha=0.15, color="steelblue", label="Calibration gap")

plt.xlabel("Mean Predicted Probability (Confidence)", fontsize=12)
plt.ylabel("Fraction of Positives (Actual Accuracy)", fontsize=12)
plt.title("DenseNet-121 — Reliability Diagram (Calibration Curve)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("densenet_reliability_diagram.png", dpi=150)
plt.show()

print("How to read: Points above the diagonal = underconfident (model less sure than it should be).")
print("             Points below the diagonal = overconfident (model more sure than it should be).")


# ── 11b. ECE Score ───────────────────────────────────────────────────────────

def compute_ece(labels, probs, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures the weighted average gap between predicted confidence
    and actual accuracy across bins.
    Lower ECE = better calibrated model.
    Range: 0.0 (perfect) to 1.0 (worst).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        # Find samples whose predicted probability falls in this bin
        in_bin = (probs >= lower) & (probs < upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            # Accuracy in this bin = fraction of correct predictions
            bin_accuracy = (labels[in_bin] == (probs[in_bin] >= 0.5).astype(int)).mean()
            # Confidence in this bin = mean predicted probability
            bin_confidence = probs[in_bin].mean()
            # Weighted by how many samples are in this bin
            ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)

    return ece


ece = compute_ece(all_labels, all_probs, n_bins=10)
print(f"\n========== Calibration Score ==========")
print(f"ECE (Expected Calibration Error): {ece:.4f}")
if ece < 0.05:
    print("Interpretation: Excellent calibration (ECE < 0.05)")
elif ece < 0.10:
    print("Interpretation: Good calibration (ECE < 0.10)")
elif ece < 0.20:
    print("Interpretation: Moderate miscalibration — model confidence doesn't perfectly match accuracy")
else:
    print("Interpretation: Poor calibration — model is significantly over/underconfident")



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: UNCERTAINTY VS CORRECTNESS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Key question: Is the model MORE uncertain when it is WRONG?
# If yes, our uncertainty-based flagging system is justified.
# We use prediction entropy as the uncertainty measure.
# Entropy = -sum(p * log(p)) — higher entropy means more uncertainty.

# ── 12a. Compute entropy for all test images ──────────────────────────────────

# all_probs is P(PNEUMONIA). Build full probability array [P(NORMAL), P(PNEUMONIA)]
all_probs_full = np.column_stack([1 - all_probs, all_probs])

# Entropy for each sample
entropy_all = -np.sum(all_probs_full * np.log(all_probs_full + 1e-8), axis=1)

# Correctness: 1 if prediction matches true label, 0 otherwise
correct_mask = (all_preds == all_labels).astype(int)

entropy_correct = entropy_all[correct_mask == 1]  # Entropy for correct predictions
entropy_wrong   = entropy_all[correct_mask == 0]  # Entropy for wrong predictions

print("\n========== Uncertainty vs Correctness ==========")
print(f"Avg entropy — Correct predictions : {entropy_correct.mean():.4f}")
print(f"Avg entropy — Wrong predictions   : {entropy_wrong.mean():.4f}")
print(f"(Higher entropy for wrong = model correctly uncertain when making mistakes)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: FINAL SUMMARY PRINT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("     DENSENET-121 PHASE 2 — COMPLETE RESULTS SUMMARY")
print("="*55)
print(f"  Model              : DenseNet-121 (pretrained ImageNet)")
print(f"  Dataset            : Chest X-ray Pneumonia (Kaggle)")
print(f"  Training epochs    : {epochs}")
print(f"  Class imbalance fix: Weighted CrossEntropyLoss")
print(f"  Best Val Accuracy  : {best_val_acc:.4f}")
print()
print(f"  ── Medical Metrics ──────────────────────────────")
print(f"  Precision          : {precision:.4f}")
print(f"  Recall/Sensitivity : {recall:.4f}")
print(f"  Specificity        : {specificity:.4f}")
print(f"  F1-score           : {f1:.4f}")
print(f"  AUC-ROC            : {auc:.4f}")
print()
print(f"  ── Calibration ──────────────────────────────────")
print(f"  ECE Score          : {ece:.4f}  (lower is better)")
print()
print(f"  ── Uncertainty Analysis ─────────────────────────")
print(f"  Avg entropy (correct) : {entropy_correct.mean():.4f}")
print(f"  Avg entropy (wrong)   : {entropy_wrong.mean():.4f}")
print("="*55)
print("\nAll plots saved as PNG files in current directory.")
print("Files: densenet_training_curves.png, densenet_confusion_matrix.png,")
print("       densenet_roc_curve.png, densenet_reliability_diagram.png,")
print("       densenet_confidence_histogram.png, densenet_accuracy_per_bin.png")
