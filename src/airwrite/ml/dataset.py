"""EMNIST orientation/labels and optional custom air-writing samples."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST

from airwrite.ml.model import CLASSES
from airwrite.preprocessing import preprocess_canvas


class LetterTransform:
    """Transpose raw EMNIST into upright orientation, then share runtime preprocessing."""

    def __call__(self, image):
        upright = np.asarray(image, dtype=np.uint8).T.copy()
        result = preprocess_canvas(upright)
        return torch.from_numpy(
            result if result is not None else np.zeros((1, 28, 28), dtype=np.float32)
        )


def zero_based_label(label: int) -> int:
    """EMNIST Letters uses 1..26; the network uses 0..25."""
    return label - 1


def emnist(root: Path, train: bool = True, download: bool = True):
    """Load the official letters split; hold out its test partition for evaluation."""
    return EMNIST(
        str(root),
        split="letters",
        train=train,
        download=download,
        transform=LetterTransform(),
        target_transform=zero_based_label,
    )


class CustomLetters(Dataset):
    """Load clean white-on-black PNG files from label directories A through Z."""

    def __init__(self, root: Path):
        self.samples = [
            (p, label)
            for label, name in enumerate(CLASSES)
            for p in sorted((root / name).glob("*.png"))
        ]
        if not self.samples:
            raise ValueError(f"No custom PNG samples found in {root}/A through Z")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            result = preprocess_canvas(np.asarray(image.convert("L")))
        if result is None:
            raise ValueError(f"Empty custom sample: {path}")
        return torch.from_numpy(result), label


def save_orientation_preview(dataset, path: Path):
    """Save a labeled upright sample per class for human orientation verification."""
    from PIL import ImageDraw

    sheet = Image.new("RGB", (13 * 60, 2 * 80), "#20252b")
    draw = ImageDraw.Draw(sheet)
    found = set()
    for image, label in dataset:
        if label in found:
            continue
        found.add(label)
        tile = Image.fromarray((image[0].numpy() * 255).astype(np.uint8)).resize((56, 56))
        x, y = (label % 13) * 60, (label // 13) * 80
        sheet.paste(ImageOps.colorize(tile, "black", "white"), (x, y))
        draw.text((x + 22, y + 60), CLASSES[label], fill="white")
        if len(found) == 26:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
