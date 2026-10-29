# Fabric Defect Detection: Roboflow vs PyTorch

This starter project is for the assignment's **single-label binary image classification** task. The target labels are exactly `defect_free` and `defective`. This is different from object detection: do not draw bounding boxes or use the old object-detection workflow for this assignment.

## 1. First confirm your assigned data

The email says groups use their allocated folder only:
- Group 1: Folder 1
- Group 2: Folder 2
- Group 3: Folder 3

Do not merge folders unless the instructor says otherwise. The message's S.No. ranges assign students to groups; ask your instructor/group if you are unsure of your own allocation. Keep the original labels `defect_free` and `defective`.

## 2. Recommended workflow

1. Inspect the assigned folder and confirm it contains 5,000 images labelled `defect_free` or `defective`.
2. Create a **single-label classification** project in Roboflow and upload the assigned folder with its class labels preserved.
3. Set the split to **70% train / 20% validation / 10% test** (for 5,000 images: 3,500 / 1,000 / 500 if no images are excluded).
4. Add at least two training augmentations in Roboflow. Reasonable starting points for fabric classification are horizontal flip and mild brightness/contrast changes. You may also use a small rotation if appropriate. Apply augmentation to training only; do not augment validation/test images. Record the exact settings and check the class counts.
5. Generate/train the Roboflow model. Record its project slug, version, training duration, and final validation/test metrics.
6. Download/export the **same Roboflow dataset version and split** in a classification folder structure for local PyTorch training. Keep the split identical across both approaches.
7. Fine-tune the ResNet18 baseline with `train_torch.py` on the same train/valid/test folders.
8. Put 50 held-out test images into `data/test50/`. Do not use images from the train or validation split for this requirement. For accuracy calculations, either preserve class folders under `data/test50/defect_free/` and `data/test50/defective/`, or make a labels CSV as described below.
9. Run `infer.py` to produce `predictions.csv`. Compare both models on the same test images for the report.
10. Complete `reports/report_template.md` with measured results and 2–4 real misclassified examples. Export the final report to PDF and submit the repository source code.

## 3. Dataset folder structure for PyTorch

Arrange the downloaded dataset as:

```text
data/split/
  train/
    defect_free/
    defective/
  valid/
    defect_free/
    defective/
  test/
    defect_free/
    defective/

data/test50/
  defect_free/   # optional but recommended for calculating API accuracy
  defective/     # optional but recommended for calculating API accuracy
```

Keep the 50-image inference subset separate from training and validation. When using `ImageFolder`, each split folder must have the same two class folders. If the Roboflow export uses different folder names, rename the split folder to `valid` or update `train_torch.py` accordingly; do not rename the class folders to unrelated labels.

## 4. Environment setup

Use Python 3.10–3.12 (Python 3.11 or 3.12 is a reasonable starting point). A Google Colab GPU runtime is useful for the PyTorch step, but CPU can work more slowly.

```bash
python -m venv .venv
```

Activate it, then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set credentials as environment variables. Do not put a real key in Python code or commit it to GitHub.

**Windows PowerShell:**

```powershell
$env:ROBOFLOW_API_KEY="YOUR_PRIVATE_ROBOFLOW_KEY"
$env:ROBOFLOW_MODEL_ID="your-project-slug/1"
```

**macOS/Linux:**

```bash
export ROBOFLOW_API_KEY="YOUR_PRIVATE_ROBOFLOW_KEY"
export ROBOFLOW_MODEL_ID="your-project-slug/1"
```

Get the model ID from the Roboflow project's deployed model/API instructions. The usual format is `project-slug/version`, for example `fabric-classification/1`; replace this example with your actual project and version.

## 5. Run Roboflow inference on 50 images

```bash
python infer.py --input-dir data/test50 --output predictions.csv --limit 50
```

Outputs:
- `predictions.csv`: exactly `image_name,predicted_label,confidence_score`.
- `benchmark_summary.json`: counts and mean/median client latency; when ground-truth labels are available via class subfolders or a labels CSV, it also reports accuracy.

To use a labels CSV, create a file with these columns:

```csv
image_name,true_label
sample_001.jpg,defect_free
sample_002.jpg,defective
```

Then run:

```bash
python infer.py --input-dir data/test50 --limit 50 --labels-csv data/test50_labels.csv
```

The inference script uses Roboflow's hosted **classification** endpoint via `inference-sdk`. `ROBOFLOW_MODEL_ID` must be the trained classification model ID, not a Workflow ID.

## 6. Train the PyTorch ResNet18 baseline

```bash
python train_torch.py --data-dir data/split --epochs 10 --batch-size 32 --learning-rate 0.0001
```

The script:
- uses pretrained ResNet18 and replaces its last layer with two outputs;
- uses horizontal flip and mild brightness/contrast augmentation on training images only;
- selects the best checkpoint by validation accuracy;
- evaluates the held-out test split once at the end;
- saves `outputs/resnet18_best.pth`, `outputs/training_history.csv`, `outputs/test_predictions.csv`, and `outputs/torch_metrics.json`.

A first run may download pretrained ImageNet weights. If memory is limited, reduce the batch size (for example, `--batch-size 16`). Record the actual hardware, epoch count, time, and any changes in the report.

## 7. Fair comparison and report

- Use the same 70/20/10 split and the same held-out images.
- Do not train or tune on the test split.
- For an apples-to-apples accuracy comparison, evaluate both models on the same 50 labelled unseen images (or on the full held-out test set if you run both models on all of it).
- Roboflow latency from `infer.py` is **client round-trip latency** (network + server response). PyTorch timing is the local model forward pass. State this difference in the report rather than claiming the measures are perfectly equivalent.
- Report actual measured training time and test metrics. Do not invent metrics.
- Inspect the actual misclassified files and explain likely reasons grounded in the images.

## 8. Suggested repository layout

```text
fabric-defect-detection/
  README.md
  requirements.txt
  .gitignore
  .env.example
  infer.py
  train_torch.py
  predictions.csv             # generated after running inference
  benchmark_summary.json      # generated after running inference
  outputs/                    # generated locally; large checkpoints are gitignored
  reports/
    report_template.md
    assignment_report.pdf     # export your completed report here
```

Do not commit the full image dataset, private API key, or large checkpoints unless the instructor specifically requests them. Include source code and the required report/results.

## 9. Official links

- Roboflow classification hosted API: https://docs.roboflow.com/inference-classification/hosted-api
- Roboflow Workflows: https://docs.roboflow.com/workflows
- Roboflow Inference SDK: https://inference.roboflow.com/
- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Streamlit Community Cloud is not required for this assignment unless you choose to deploy a demo.
