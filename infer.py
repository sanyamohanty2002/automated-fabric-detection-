"""Run a Roboflow Workflow over a folder of classification images.

Expected input layout:
    data/test50/
      Defect free/...
      Defected/...
      Waste/...

Environment variables (PowerShell):
    $env:ROBOFLOW_API_KEY = "<new API key>"
    $env:ROBOFLOW_WORKSPACE = "sanya-mohanty"   # optional
    $env:ROBOFLOW_WORKFLOW_ID = "fabric-defect-detection-idm-2-vfabric-defect-detection-idm-2-1-resnet18-t1-logic"  # optional

Example:
    python infer.py --input-dir data/test50 --output outputs/predictions.csv --limit 50

CSV output columns: image_name, predicted_label, confidence_score
A JSON summary and raw workflow responses are also saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from inference_sdk import InferenceHTTPClient, InferenceConfiguration

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_WORKSPACE = "sanya-mohanty"
DEFAULT_WORKFLOW_ID = (
    "fabric-defect-detection-idm-2-vfabric-defect-detection-idm-2-1-resnet18-t1-logic"
)


def confidence_value(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score /= 100.0
    return max(0.0, min(score, 1.0))


def _find_prediction_in_object(obj: Any) -> tuple[str, float] | None:
    """Extract a class label and confidence from Roboflow Workflow responses.

    Prefer explicit `predicted_classes` and the matching per-class confidence.
    Search nested Workflow outputs before interpreting arbitrary numeric fields
    (such as the inference runtime `time`) as class scores.
    """
    if isinstance(obj, list):
        candidates = [
            candidate
            for item in obj
            if (candidate := _find_prediction_in_object(item)) is not None
        ]
        return max(candidates, key=lambda item: item[1]) if candidates else None

    if not isinstance(obj, dict):
        return None

    # Roboflow classification output shape:
    # {"predicted_classes": ["Defected"],
    #  "predictions": {"Defected": {"confidence": 0.99, ...}}}
    predicted_classes = obj.get("predicted_classes")
    class_scores = obj.get("predictions")
    if isinstance(predicted_classes, list) and predicted_classes and isinstance(class_scores, dict):
        chosen_label = str(predicted_classes[0])
        details = class_scores.get(chosen_label)
        if isinstance(details, dict):
            raw_score = details.get("confidence", details.get("score"))
            if raw_score is not None:
                return chosen_label, confidence_value(raw_score)

    # Search likely Workflow output containers first. This avoids mistaking
    # metadata such as {"time": 0.007} for a class named "time".
    for key in ("predictions", "output", "outputs", "result", "results"):
        value = obj.get(key)
        if isinstance(value, (dict, list)):
            candidate = _find_prediction_in_object(value)
            if candidate is not None:
                return candidate

    # Common direct classification response: {"top": "Defected", "confidence": 0.96}
    label = obj.get("top") or obj.get("predicted_class") or obj.get("class") or obj.get("label")
    score = obj.get("confidence", obj.get("score"))
    if label is not None and score is not None and not isinstance(score, (dict, list)):
        return str(label), confidence_value(score)

    # Some APIs return a plain map of class names to confidence records.
    # Only accept this fallback after nested output fields have been checked.
    metadata_keys = {
        "inference_id", "time", "resolved_model", "image", "predictions",
        "predicted_classes", "prediction_type", "parent_id", "root_parent_id",
        "confidence", "score", "class_id", "model_id", "model_package_id",
        "backend", "quantization", "width", "height",
    }
    class_candidates: list[tuple[str, float]] = []
    for class_name, details in obj.items():
        if class_name in metadata_keys:
            continue
        if isinstance(details, dict):
            raw_score = details.get("confidence", details.get("score"))
            if raw_score is not None:
                class_candidates.append((str(class_name), confidence_value(raw_score)))
        elif isinstance(details, (float, int)) and not isinstance(details, bool):
            class_candidates.append((str(class_name), confidence_value(details)))
    if class_candidates:
        return max(class_candidates, key=lambda item: item[1])

    # Last resort for other workflow wrapper structures.
    for key, value in obj.items():
        if key not in {"time", "confidence", "score"} and isinstance(value, (dict, list)):
            candidate = _find_prediction_in_object(value)
            if candidate is not None:
                return candidate
    return None

def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Roboflow classification Workflow on images.")
    parser.add_argument("--input-dir", default="data/test50", help="Folder containing images, optionally in class subfolders")
    parser.add_argument("--output", default="outputs/predictions.csv", help="CSV output path")
    parser.add_argument("--limit", type=int, default=50, help="Maximum images to process; use 0 for all")
    parser.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE", DEFAULT_WORKSPACE))
    parser.add_argument("--workflow-id", default=os.getenv("ROBOFLOW_WORKFLOW_ID", DEFAULT_WORKFLOW_ID))
    parser.add_argument("--summary", default="outputs/roboflow_summary.json", help="Metrics/timing summary JSON")
    parser.add_argument("--raw-json", default="outputs/roboflow_raw_responses.json", help="Raw Workflow responses JSON")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY. Set it in the terminal; do not put it in this file or share it in chat.")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir.resolve()}")

    image_paths = list_images(input_dir)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit(f"No supported image files found in {input_dir.resolve()}")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    raw_json_path = Path(args.raw_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    ground_truth: list[str] = []
    predictions: list[str] = []
    raw_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_name", "predicted_label", "confidence_score"],
        )
        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            start_time = time.perf_counter()
            try:
                response = client.run_workflow(
                    workspace_name=args.workspace,
                    workflow_id=args.workflow_id,
                    images={"image": str(image_path.resolve())},
                    use_cache=False,
                )
                elapsed = time.perf_counter() - start_time
                prediction = _find_prediction_in_object(response)
                if prediction is None:
                    raise ValueError(
                        "Could not find a class/confidence in the Workflow response. "
                        "Inspect outputs/roboflow_raw_responses.json."
                    )

                label, confidence = prediction
                writer.writerow({
                    "image_name": image_path.name,
                    "predicted_label": label,
                    "confidence_score": f"{confidence:.6f}",
                })
                csv_file.flush()
                latencies.append(elapsed)
                predictions.append(label)

                true_label = image_path.parent.name if image_path.parent != input_dir else None
                if true_label is not None:
                    ground_truth.append(true_label)

                raw_results.append({
                    "image_name": image_path.name,
                    "true_label_from_folder": true_label,
                    "response": response,
                })
                print(f"[{index}/{len(image_paths)}] {image_path.name}: {label} ({confidence:.2%}) | {elapsed:.2f}s")

            except Exception as exc:
                failures.append({"image_name": image_path.name, "error": str(exc)})
                raw_results.append({"image_name": image_path.name, "error": str(exc)})
                print(f"[{index}/{len(image_paths)}] ERROR {image_path.name}: {exc}")

    summary: dict[str, Any] = {
        "workspace_name": args.workspace,
        "workflow_id": args.workflow_id,
        "input_dir": str(input_dir),
        "requested_images": len(image_paths),
        "successful_predictions": len(latencies),
        "failed_predictions": len(failures),
        "mean_client_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_client_latency_seconds": statistics.median(latencies) if latencies else None,
        "min_client_latency_seconds": min(latencies) if latencies else None,
        "max_client_latency_seconds": max(latencies) if latencies else None,
        "latency_note": "Client latency includes network/API round-trip time and is not pure model compute time.",
        "failures": failures,
    }

    # Folder names give ground-truth labels for the supplied test50 layout.
    if ground_truth and len(ground_truth) == len(predictions):
        correct = sum(
            normalize_label(true) == normalize_label(pred)
            for true, pred in zip(ground_truth, predictions)
        )
        summary["ground_truth_images"] = len(ground_truth)
        summary["accuracy_on_processed_foldered_images"] = correct / len(ground_truth)
        summary["accuracy_note"] = "Computed from parent-folder names; confirm these folder names represent the true class labels."
    else:
        summary["ground_truth_images"] = 0
        summary["accuracy_on_processed_foldered_images"] = None
        summary["accuracy_note"] = "Images must be arranged in class-named subfolders to calculate accuracy from folder labels."

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw_json_path.write_text(json.dumps(raw_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nPredictions CSV: {output_path.resolve()}")
    print(f"Summary JSON:    {summary_path.resolve()}")
    print(f"Raw responses:   {raw_json_path.resolve()}")
    if latencies:
        print(f"Mean client latency: {statistics.mean(latencies):.2f} seconds/image")
    if failures:
        print(f"WARNING: {len(failures)} image(s) failed; review the summary JSON.")


if __name__ == "__main__":
    main()
