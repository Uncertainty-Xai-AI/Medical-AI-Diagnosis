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

