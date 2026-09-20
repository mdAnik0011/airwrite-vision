"""Compact 26-class CNN and versioned checkpoint format."""

from pathlib import Path
from string import ascii_uppercase

import torch
from torch import nn

CLASSES = list(ascii_uppercase)
PREPROCESSING = "white-ink-crop-pad-center28-v1"


class AirWriteCNN(nn.Module):
    """CPU-friendly convolutional letter classifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_checkpoint(path: Path, device="cpu") -> AirWriteCNN:
    """Load tensor-only checkpoints and reject incompatible preprocessing/class maps."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a dictionary of weights and metadata")
    if checkpoint.get("classes") != CLASSES or checkpoint.get("preprocessing") != PREPROCESSING:
        raise ValueError("Incompatible checkpoint. Train with this AirWrite version.")
    model = AirWriteCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model
