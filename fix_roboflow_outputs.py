"""Rebuild classification CSV/evaluation from saved raw Workflow responses.

This script makes no Roboflow API calls, so it does not use additional inference credits.
Run from the project root:
    python fix_roboflow_outputs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("outputs")
RAW_PATH = ROOT / "roboflow_raw_responses.json"
PREDICTIONS_PATH = ROOT / "predictions.csv"
COMPARISON_PATH = ROOT / "roboflow_comparison.csv"
SUMMARY_PATH = ROOT / "roboflow_summary.json"


def confidence_value(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score /= 100.0
    return max(0.0, min(score, 1.0))


def extract_prediction(obj: Any) -> tuple[str, float] | None:
    if isinstance(obj, list):
        candidates = [extract_prediction(item) for item in obj]
        candidates = [item for item in candidates if item is not None]
        return max(candidates, key=lambda item: item[1]) if candidates else None
    if not isinstance(obj, dict):
        return None

    labels = obj.get("predicted_classes")
    class_scores = obj.get("predictions")
    if isinstance(labels, list) and labels:
        label = str(labels[0])
        if isinstance(class_scores, dict):
            details = class_scores.get(label)
            if isinstance(details, dict):
                score = details.get("confidence", details.get("score"))
                if score is not None:
                    return label, confidence_value(score)

    # Simple classification response shape.
    label = obj.get("top") or obj.get("predicted_class") or obj.get("class") or obj.get("label")
    score = obj.get("confidence", obj.get("score"))
    if label is not None and score is not None and not isinstance(score, (dict, list)):
        return str(label), confidence_value(score)

    predictions = obj.get("predictions")
    if isinstance(predictions, dict):
        # Class names map to details with confidence. Metadata like numeric "time"
        # is deliberately ignored.
        scores = []
        for name, details in predictions.items():
            if isinstance(details, dict):
                value = details.get("confidence", details.get("score"))
                if value is not None:
                    scores.append((str(name), confidence_value(value)))
        if scores:
            return max(scores, key=lambda item: item[1])

    candidates = [extract_prediction(value) for value in obj.values()]
    candidates = [item for item in candidates if item is not None]
    return max(candidates, key=lambda item: item[1]) if candidates else None


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw response file not found: {RAW_PATH}")

    raw_rows = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    pred_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, str]] = []
    errors: list[str] = []

    for row in raw_rows:
        image_name = row.get("image_name", "unknown")
        if "error" in row:
            errors.append(f"{image_name}: {row['error']}")
            continue
        prediction = extract_prediction(row.get("response"))
        if prediction is None:
            errors.append(f"{image_name}: unable to parse workflow response")
            continue

        label, confidence = prediction
        true_label = row.get("true_label_from_folder")
        pred_rows.append({
            "image_name": image_name,
            "predicted_label": label,
            "confidence_score": f"{confidence:.6f}",
        })
        comparison_rows.append({
            "image_name": image_name,
            "true_label": str(true_label or ""),
            "predicted_label": label,
            "confidence_score": f"{confidence:.6f}",
            "correct": str(bool(true_label) and normalize_label(str(true_label)) == normalize_label(label)),
        })

    ROOT.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "predicted_label", "confidence_score"])
        writer.writeheader()
        writer.writerows(pred_rows)

    with COMPARISON_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_name", "true_label", "predicted_label", "confidence_score", "correct"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    labelled = [r for r in comparison_rows if r["true_label"]]
    correct_count = sum(r["correct"] == "True" for r in labelled)
    summary["successful_predictions"] = len(pred_rows)
    summary["failed_predictions"] = len(errors)
    summary["ground_truth_images"] = len(labelled)
    summary["correct_predictions"] = correct_count
    summary["accuracy_on_processed_foldered_images"] = (correct_count / len(labelled)) if labelled else None
    summary["accuracy_note"] = "Computed from parent-folder names in data/test50; verify the folders contain correct ground-truth labels."
    summary["parser_note"] = "Labels extracted from predicted_classes and corresponding per-class confidence; numeric metadata such as 'time' is not a class label."
    summary["parse_errors"] = errors
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Rebuilt outputs from saved raw JSON; no new Roboflow API calls made.")
    print(f"Predictions CSV: {PREDICTIONS_PATH.resolve()} ({len(pred_rows)} rows)")
    print(f"Comparison CSV:  {COMPARISON_PATH.resolve()} ({len(comparison_rows)} rows)")
    print(f"Summary JSON:    {SUMMARY_PATH.resolve()}")
    if labelled:
        print(f"Accuracy on labelled images: {correct_count}/{len(labelled)} = {correct_count / len(labelled):.2%}")
    if errors:
        print(f"Parsing/errors: {len(errors)}; inspect outputs/roboflow_summary.json")


if __name__ == "__main__":
    main()
