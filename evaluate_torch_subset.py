"""Evaluate the saved ResNet18 checkpoint on a selected labelled subset.

Expected layout:
    data/test50/Defect free/*.jpg
    data/test50/Defected/*.jpg
    data/test50/Waste/*.jpg
    outputs/resnet18_best.pth

Run from the project root:
    python evaluate_torch_subset.py --data-dir data/test50 --output-dir outputs

This script does not retrain the model and does not call Roboflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet18 checkpoint on a labelled image subset.")
    parser.add_argument("--data-dir", default="data/test50", help="Folder containing class subfolders, e.g. data/test50")
    parser.add_argument("--output-dir", default="outputs", help="Folder containing checkpoint and receiving results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for Windows reliability")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "resnet18_best.pth"
    if not data_dir.is_dir():
        raise SystemExit(f"Image folder not found: {data_dir.resolve()}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path.resolve()}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) == 0:
        raise SystemExit(f"No supported images found under {data_dir.resolve()}")

    # Keep class ordering identical to ImageFolder's alphabetical class order.
    classes = dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))

    state = torch.load(checkpoint_path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint doesn't match the class count/architecture in this dataset. "
            "Check that data/test50 has the same three class folders used for training.\n" + str(exc)
        ) from exc
    model.to(device).eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confidence_values: list[float] = []
    inference_seconds = 0.0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)
            y_true.extend(targets.tolist())
            y_pred.extend(preds.cpu().tolist())
            confidence_values.extend(confidences.cpu().tolist())
            total += len(targets)

    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    latency_ms = 1000 * inference_seconds / max(total, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "torch_test50_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "true_label", "predicted_label", "confidence_score", "correct"])
        writer.writeheader()
        for (image_path, true_idx), pred_idx, conf in zip(dataset.samples, y_pred, confidence_values):
            writer.writerow({
                "image_name": Path(image_path).name,
                "true_label": classes[true_idx],
                "predicted_label": classes[pred_idx],
                "confidence_score": f"{conf:.6f}",
                "correct": str(true_idx == pred_idx),
            })

    metrics = {
        "model": "ResNet18 saved best-validation checkpoint",
        "device": str(device),
        "classes": classes,
        "class_to_idx": dataset.class_to_idx,
        "test_images": len(dataset),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "forward_latency_ms_per_image_batch_averaged": latency_ms,
        "confusion_matrix_rows_true_columns_predicted": matrix,
        "classification_report": report,
        "predictions_csv": str(csv_path),
        "latency_note": "PyTorch forward-pass timing on local device; excludes image loading and network time.",
        "comparison_note": "Use the same filenames as outputs/predictions.csv from Roboflow before making a direct paired comparison.",
    }
    metrics_path = output_dir / "torch_test50_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved checkpoint evaluated on test50 subset (no retraining).")
    print(f"Device:          {device}")
    print(f"Classes:         {classes}")
    print(f"Images:          {len(dataset)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Latency:         {latency_ms:.2f} ms/image (forward pass, batch-averaged)")
    print(f"Predictions CSV: {csv_path.resolve()}")
    print(f"Metrics JSON:    {metrics_path.resolve()}")
    print("Reminder: only one Waste image is present in this subset, so class-specific Waste metrics are not reliable.")


if __name__ == "__main__":
    main()
