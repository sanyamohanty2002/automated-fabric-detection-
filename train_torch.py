"""Fine-tune a pretrained ResNet18 on the same train/valid/test split as Roboflow.

Expected dataset layout:
    data/split/train/defect_free/...
    data/split/train/defective/...
    data/split/valid/defect_free/...
    data/split/valid/defective/...
    data/split/test/defect_free/...
    data/split/test/defective/...

Training augmentation is applied only to the train set: horizontal flip and mild
brightness/contrast variation. Validation/test data are not augmented.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(num_classes: int) -> torch.nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_seen = 0
    correct = 0
    all_true, all_pred, all_confidence, all_names = [], [], [], []
    inference_seconds = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                sync_device(device)
                start = time.perf_counter()
                outputs = model(inputs)
                sync_device(device)
                inference_seconds += time.perf_counter() - start
                loss = criterion(outputs, targets)

        probabilities = torch.softmax(outputs.detach(), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        correct += int((predicted == targets).sum().item())
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(predicted.cpu().tolist())
        all_confidence.extend(confidence.cpu().tolist())

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": correct / max(total_seen, 1),
        "true": all_true,
        "pred": all_pred,
        "confidence": all_confidence,
        "inference_seconds": inference_seconds,
        "count": total_seen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 for fabric image classification.")
    parser.add_argument("--data-dir", default="data/split", help="Folder containing train/valid/test class folders")
    parser.add_argument("--output-dir", default="outputs", help="Folder for model and metrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive integers")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        if not (data_dir / split).is_dir():
            raise SystemExit(f"Missing split folder: {data_dir / split}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(data_dir / "valid", transform=evaluation_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=evaluation_transform)
    if train_dataset.class_to_idx != valid_dataset.class_to_idx or train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise SystemExit("Class folders differ across train/valid/test. Ensure both classes exist in every split.")

    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    classes = train_dataset.classes
    model = make_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_val_accuracy = -1.0
    checkpoint_path = output_dir / "resnet18_best.pth"
    history_path = output_dir / "training_history.csv"
    history = []
    total_training_start = time.perf_counter()

    print(f"Device: {device}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Images: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")
    print("Note: pretrained ResNet18 weights may need to download on first run.")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_result = run_epoch(model, valid_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "valid_loss": valid_result["loss"],
            "valid_accuracy": valid_result["accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f}, train acc={row['train_accuracy']:.4f} | "
            f"valid loss={row['valid_loss']:.4f}, valid acc={row['valid_accuracy']:.4f} | "
            f"{epoch_seconds:.1f}s"
        )
        if valid_result["accuracy"] > best_val_accuracy:
            best_val_accuracy = valid_result["accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    training_seconds = time.perf_counter() - total_training_start
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "class_to_idx.json").write_text(json.dumps(train_dataset.class_to_idx, indent=2), encoding="utf-8")

    # Test the best validation checkpoint only once after training.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_result = run_epoch(model, test_loader, criterion, device)
    y_true, y_pred = test_result["true"], test_result["pred"]
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    inference_ms_per_image = (test_result["inference_seconds"] / max(test_result["count"], 1)) * 1000

    test_predictions_path = output_dir / "test_predictions.csv"
    with test_predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "true_label", "predicted_label", "confidence_score"])
        writer.writeheader()
        for path, true_idx, pred_idx, confidence in zip(
            test_dataset.samples,
            y_true,
            y_pred,
            test_result["confidence"],
        ):
            writer.writerow({
                "image_name": Path(path[0]).name,
                "true_label": classes[true_idx],
                "predicted_label": classes[pred_idx],
                "confidence_score": f"{confidence:.6f}",
            })

    metrics = {
        "model": "ResNet18 pretrained (fine-tuned)",
        "device": str(device),
        "class_to_idx": train_dataset.class_to_idx,
        "train_images": len(train_dataset),
        "valid_images": len(valid_dataset),
        "test_images": len(test_dataset),
        "epochs_requested": args.epochs,
        "best_validation_accuracy": best_val_accuracy,
        "training_seconds_including_validation": training_seconds,
        "test_accuracy": accuracy,
        "test_macro_precision": macro_precision,
        "test_macro_recall": macro_recall,
        "test_macro_f1": macro_f1,
        "test_forward_latency_ms_per_image_batch_averaged": inference_ms_per_image,
        "test_confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "classification_report": report,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "test_predictions_csv": str(test_predictions_path),
        "latency_note": "Forward-pass timing is averaged over test batches and excludes image decoding/network time; GPU timing is synchronized.",
    }
    metrics_path = output_dir / "torch_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTest results")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro precision:{macro_precision:.4f}")
    print(f"Macro recall:   {macro_recall:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Training time:  {training_seconds:.1f} seconds")
    print(f"Inference:      {inference_ms_per_image:.2f} ms/image (averaged over batches)")
    print(f"Metrics saved:  {metrics_path}")


if __name__ == "__main__":
    main()
