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
