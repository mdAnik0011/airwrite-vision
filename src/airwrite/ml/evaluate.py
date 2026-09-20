"""Headless held-out evaluation with per-class metrics and confusion matrix."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from airwrite.ml.dataset import CustomLetters, emnist
from airwrite.ml.model import CLASSES, load_checkpoint


def main(argv=None):
    """Evaluate official EMNIST test data or a separate custom holdout directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/airwrite_cnn.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--custom-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/evaluation.json"))
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.model, device).eval()
    dataset = (
        CustomLetters(args.custom_dir) if args.custom_dir else emnist(args.data_dir, train=False)
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    matrix = torch.zeros((26, 26), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=args.batch_size):
            predictions = model(images.to(device)).argmax(1).cpu()
            matrix += torch.bincount(labels * 26 + predictions, minlength=26 * 26).reshape(26, 26)
    counts = matrix.sum(1)
    results = {
        "dataset": str(args.custom_dir) if args.custom_dir else "EMNIST Letters test",
        "samples": matrix.sum().item(),
        "overall_accuracy": matrix.diag().sum().item() / max(1, matrix.sum().item()),
        "per_class_accuracy": {
            label: matrix[i, i].item() / counts[i].item() if counts[i] else None
            for i, label in enumerate(CLASSES)
        },
        "classes": CLASSES,
        "confusion_matrix_rows": "true",
        "confusion_matrix_columns": "predicted",
        "confusion_matrix": matrix.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Accuracy: {results['overall_accuracy']:.2%}; results: {args.output}")


if __name__ == "__main__":
    main()
