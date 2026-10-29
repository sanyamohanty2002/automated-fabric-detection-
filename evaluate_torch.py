"""Evaluate a saved ResNet18 checkpoint without running training again.

Expected layout:
    data/train/<class_name>/...
    data/valid/<class_name>/...
    data/test/<class_name>/...

Expected checkpoint:
    outputs/resnet18_best.pth

Run:
    python evaluate_torch.py --data-dir data --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ResNet18 checkpoint without retraining."
    )
    parser.add_argument("--data-dir", default="data", help="Folder containing train/valid/test")
    parser.add_argument("--output-dir", default="outputs", help="Folder for checkpoint and results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for Windows reliability")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be a positive integer")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "resnet18_best.pth"
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"Missing test folder: {test_dir}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=evaluation_transform)
    if not test_dataset.classes:
        raise SystemExit(f"No class folders/images found under {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # The training script saved a plain state_dict in resnet18_best.pth.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint and test dataset do not match. Confirm the same class folders "
            "and model architecture were used during training.\n" + str(exc)
        ) from exc
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    image_names: list[str] = []
    inference_seconds = 0.0
    total_seen = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            total_seen += targets.size(0)

    # ImageFolder order matches the DataLoader order because shuffle=False.
    image_names = [Path(path).name for path, _ in test_dataset.samples]
    classes = test_dataset.classes
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    latency_ms = inference_seconds / max(total_seen, 1) * 1000

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_name", "true_label", "predicted_label", "confidence_score"],
        )
        writer.writeheader()
        for name, true_idx, pred_idx, confidence in zip(
            image_names, y_true, y_pred, confidences
        ):
            writer.writerow({
                "image_name": name,
                "true_label": classes[true_idx],
                "predicted_label": classes[pred_idx],
                "confidence_score": f"{confidence:.6f}",
            })

    metrics = {
        "model": "ResNet18 pretrained (fine-tuned)",
        "device": str(device),
        "class_to_idx": test_dataset.class_to_idx,
        "test_images": len(test_dataset),
        "test_accuracy": accuracy,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "test_macro_f1": macro_f1,
        "test_forward_latency_ms_per_image_batch_averaged": latency_ms,
        "test_confusion_matrix_rows_true_columns_predicted": matrix,
        "classification_report": report,
        "checkpoint": str(checkpoint_path),
        "test_predictions_csv": str(csv_path),
        "latency_note": (
            "Forward-pass timing averaged over test batches; excludes image decoding and network time."
        ),
        "training_note": (
            "Training was interrupted after four complete epochs. This file evaluates the saved best "
            "validation checkpoint; it does not resume training or estimate the interrupted epoch time."
        ),
    }
    metrics_path = output_dir / "torch_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved checkpoint evaluation complete")
    print(f"Device:          {device}")
    print(f"Classes:         {classes}")
    print(f"Test images:     {len(test_dataset)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Latency:         {latency_ms:.2f} ms/image (forward pass, batch-averaged)")
    print(f"Metrics saved:   {metrics_path}")
    print(f"Predictions CSV: {csv_path}")
    print("Note: class 'Waste' has only one test image, so its class-specific metrics are not reliable.")


if __name__ == "__main__":
    main()
