"""Reproducible EMNIST baseline and optional custom-data fine-tuning."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from airwrite.ml.dataset import CustomLetters, emnist, save_orientation_preview
from airwrite.ml.model import CLASSES, PREPROCESSING, AirWriteCNN, load_checkpoint


def run_epoch(model, loader, device, optimizer=None):
    """Compute sample-weighted loss and accuracy, with gradients only when training."""
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum / max(1, count), correct / max(1, count)


def main(argv=None):
    """Train with a seeded validation split and save the best validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--custom-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/airwrite_cnn.pt"))
    parser.add_argument("--limit", type=int, help="Subset size for pipeline smoke tests")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(args.seed)
    dataset = emnist(args.data_dir)
    save_orientation_preview(dataset, Path("data/processed/emnist_orientation.png"))
    if args.limit is not None:
        indices = torch.randperm(len(dataset), generator=generator)[: args.limit]
        dataset = Subset(dataset, indices.tolist())
    validation_count = max(1, int(len(dataset) * 0.1))
    train, validation = random_split(
        dataset, [len(dataset) - validation_count, validation_count], generator=generator
    )
    if args.custom_dir:
        # Custom samples augment training only; EMNIST validation/test remain separate.
        train = ConcatDataset([train, CustomLetters(args.custom_dir)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        load_checkpoint(args.checkpoint, device) if args.checkpoint else AirWriteCNN().to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    best = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; train={len(train)} validation={len(validation)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, device)
        metrics = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=val_loss,
            validation_accuracy=val_accuracy,
        )
        history.append(metrics)
        print(json.dumps(metrics))
        if val_accuracy > best:
            best = val_accuracy
            temporary = args.output.with_suffix(".tmp")
            torch.save(
                dict(
                    model_state=model.state_dict(),
                    classes=CLASSES,
                    preprocessing=PREPROCESSING,
                    seed=args.seed,
                    metrics=metrics,
                ),
                temporary,
            )
            temporary.replace(args.output)
    args.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
