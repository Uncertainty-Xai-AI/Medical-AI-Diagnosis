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
