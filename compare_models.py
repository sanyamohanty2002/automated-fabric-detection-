"""Compare Roboflow and PyTorch predictions on the same labelled image subset.

Expected inputs:
  outputs/roboflow_comparison.csv
  outputs/torch_test50_predictions.csv

Run from the project root:
  python compare_models.py

Writes:
  outputs/model_comparison_50.csv
  outputs/model_comparison_summary.json
  outputs/misclassified_images.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

OUT = Path("outputs")
ROBOFLOW_CSV = OUT / "roboflow_comparison.csv"
TORCH_CSV = OUT / "torch_test50_predictions.csv"
COMBINED_CSV = OUT / "model_comparison_50.csv"
SUMMARY_JSON = OUT / "model_comparison_summary.json"
ERRORS_CSV = OUT / "misclassified_images.csv"


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def load_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path.resolve()}")
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("image_name") or "").strip()
            if not name:
                continue
            if name in rows:
                raise SystemExit(f"Duplicate image_name '{name}' in {path}")
            rows[name] = row
    return rows


def safe_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred) if y_true else None,
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if y_true else None,
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0) if y_true else {},
    }


def main() -> None:
    roboflow = load_csv(ROBOFLOW_CSV)
    torch = load_csv(TORCH_CSV)
    common = sorted(set(roboflow) & set(torch))
    only_rf = sorted(set(roboflow) - set(torch))
    only_torch = sorted(set(torch) - set(roboflow))

    if not common:
        raise SystemExit("No matching filenames between Roboflow and PyTorch prediction files.")

    combined: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    true_labels: list[str] = []
    rf_preds: list[str] = []
    torch_preds: list[str] = []
    labels_seen: set[str] = set()
    label_name_by_norm: dict[str, str] = {}

    for name in common:
        r = roboflow[name]
        t = torch[name]
        true_r = (r.get("true_label") or "").strip()
        true_t = (t.get("true_label") or "").strip()
        true_label = true_t or true_r
        if true_r and true_t and normalize_label(true_r) != normalize_label(true_t):
            raise SystemExit(
                f"Ground-truth mismatch for {name}: Roboflow='{true_r}', PyTorch='{true_t}'. "
                "Stop and verify folder labels before comparing."
            )
        rf_label = (r.get("predicted_label") or "").strip()
        torch_label = (t.get("predicted_label") or "").strip()
        if not true_label or not rf_label or not torch_label:
            raise SystemExit(f"Missing label in one of the prediction files for {name}.")

        labels_seen.add(normalize_label(true_label))
        label_name_by_norm[normalize_label(true_label)] = true_label
        labels_seen.add(normalize_label(rf_label))
        label_name_by_norm[normalize_label(rf_label)] = rf_label
        labels_seen.add(normalize_label(torch_label))
        label_name_by_norm[normalize_label(torch_label)] = torch_label
        rf_ok = normalize_label(rf_label) == normalize_label(true_label)
        torch_ok = normalize_label(torch_label) == normalize_label(true_label)
        agree = normalize_label(rf_label) == normalize_label(torch_label)

        row = {
            "image_name": name,
            "true_label": true_label,
            "roboflow_predicted_label": rf_label,
            "roboflow_confidence": r.get("confidence_score", ""),
            "roboflow_correct": str(rf_ok),
            "pytorch_predicted_label": torch_label,
            "pytorch_confidence": t.get("confidence_score", ""),
            "pytorch_correct": str(torch_ok),
            "models_agree": str(agree),
        }
        combined.append(row)
        if not rf_ok or not torch_ok:
            errors.append(row.copy())

        true_labels.append(normalize_label(true_label))
        rf_preds.append(normalize_label(rf_label))
        torch_preds.append(normalize_label(torch_label))

    labels = sorted(set(true_labels) | set(rf_preds) | set(torch_preds))
    common_count = len(common)
    rf_correct = sum(normalize_label(row["roboflow_predicted_label"]) == normalize_label(row["true_label"]) for row in combined)
    torch_correct = sum(normalize_label(row["pytorch_predicted_label"]) == normalize_label(row["true_label"]) for row in combined)
    both_correct = sum(row["roboflow_correct"] == "True" and row["pytorch_correct"] == "True" for row in combined)
    both_wrong = sum(row["roboflow_correct"] == "False" and row["pytorch_correct"] == "False" for row in combined)
    rf_only_correct = sum(row["roboflow_correct"] == "True" and row["pytorch_correct"] == "False" for row in combined)
    torch_only_correct = sum(row["roboflow_correct"] == "False" and row["pytorch_correct"] == "True" for row in combined)
    agreement = sum(row["models_agree"] == "True" for row in combined)

    OUT.mkdir(exist_ok=True)
    with COMBINED_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_name", "true_label", "roboflow_predicted_label", "roboflow_confidence", "roboflow_correct", "pytorch_predicted_label", "pytorch_confidence", "pytorch_correct", "models_agree"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    with ERRORS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image_name", "true_label", "roboflow_predicted_label", "roboflow_confidence", "roboflow_correct", "pytorch_predicted_label", "pytorch_confidence", "pytorch_correct", "models_agree"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    # sklearn reports normalize labels; readable class names are retained in summary separately.
    summary = {
        "images_matched_and_compared": common_count,
        "roboflow_rows_not_matched": only_rf,
        "pytorch_rows_not_matched": only_torch,
        "roboflow": {
            "correct": rf_correct,
            "accuracy": rf_correct / common_count,
            "macro_precision": safe_metrics(true_labels, rf_preds, labels)["macro_precision"],
            "macro_recall": safe_metrics(true_labels, rf_preds, labels)["macro_recall"],
            "macro_f1": safe_metrics(true_labels, rf_preds, labels)["macro_f1"],
            "classification_report": safe_metrics(true_labels, rf_preds, labels)["classification_report"],
        },
        "pytorch": {
            "correct": torch_correct,
            "accuracy": torch_correct / common_count,
            "macro_precision": safe_metrics(true_labels, torch_preds, labels)["macro_precision"],
            "macro_recall": safe_metrics(true_labels, torch_preds, labels)["macro_recall"],
            "macro_f1": safe_metrics(true_labels, torch_preds, labels)["macro_f1"],
            "classification_report": safe_metrics(true_labels, torch_preds, labels)["classification_report"],
        },
        "paired_comparison": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "roboflow_only_correct": rf_only_correct,
            "pytorch_only_correct": torch_only_correct,
            "predicted_label_agreement_count": agreement,
            "predicted_label_agreement_rate": agreement / common_count,
        },
        "classes_in_test_subset": [label_name_by_norm[label] for label in labels],
        "misclassified_image_count_union": len(errors),
        "latency_interpretation": "Roboflow mean client latency includes upload/network/API round-trip and remote processing (reported earlier as 2.45 s/image). PyTorch latency reported earlier is batched local forward-pass time only (50.73 ms/image), excluding file I/O/preprocessing/network. These are different measurement scopes and must be labelled as such in the report.",
        "source_files": [str(ROBOFLOW_CSV), str(TORCH_CSV)],
        "output_files": [str(COMBINED_CSV), str(ERRORS_CSV), str(SUMMARY_JSON)],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nComparison complete: same filenames matched across both models.")
    print(f"Images compared:              {common_count}")
    print(f"Roboflow accuracy:            {rf_correct}/{common_count} = {rf_correct/common_count:.2%}")
    print(f"PyTorch accuracy:             {torch_correct}/{common_count} = {torch_correct/common_count:.2%}")
    print(f"Both correct:                 {both_correct}")
    print(f"Both wrong:                   {both_wrong}")
    print(f"Only Roboflow correct:        {rf_only_correct}")
    print(f"Only PyTorch correct:         {torch_only_correct}")
    print(f"Predictions agree:            {agreement}/{common_count} = {agreement/common_count:.2%}")
    print(f"Rows needing failure review:  {len(errors)}")
    print(f"Combined CSV:                 {COMBINED_CSV.resolve()}")
    print(f"Misclassified images CSV:     {ERRORS_CSV.resolve()}")
    print(f"Summary JSON:                 {SUMMARY_JSON.resolve()}")
    if only_rf or only_torch:
        print("WARNING: filename coverage differs between models; review unmatched filenames in summary JSON.")
    print("Latency note: Roboflow client latency and local PyTorch forward-pass latency measure different scopes.")


if __name__ == "__main__":
    main()
