# Benchmarking Roboflow and PyTorch for Fabric Defect Classification

**Student/group:** [fill in]  
**Dataset folder:** [Folder 1 / Folder 2 / Folder 3]  
**Roboflow project/version:** [fill in]  

## 1. Objective and Dataset (approx. 1/2 page)
The objective was to classify fabric images into `defect_free` and `defective`. We used [number] images from the assigned folder. The split was 70% training ([n]), 20% validation ([n]) and 10% test ([n]). We used the same split for both approaches to make the comparison fair. Training-only augmentation: (1) horizontal flip, (2) mild brightness/contrast change [and any Roboflow augmentations actually used]. No augmentation was applied to validation or test images.

## 2. Approaches (approx. 1/2 page)
**Roboflow:** Describe the classification project, training settings, augmentation choices, project/version and hosted inference API. `infer.py` ran on 50 held-out images and generated `predictions.csv` with image name, predicted label and confidence score.

**PyTorch:** Describe the fine-tuned ResNet18 model, input size (224×224), training epochs, batch size, learning rate, device (CPU/GPU), and checkpoint selection using validation accuracy. State any deviations from the Roboflow experiment.

## 3. Results and Comparison (approx. 3/4 page)
Fill from the actual generated `benchmark_summary.json` and `outputs/torch_metrics.json`. Do not invent or estimate values.

| Metric | Roboflow | PyTorch ResNet18 | Measurement notes |
|---|---:|---:|---|
| Test accuracy | [value] | [value] | Use the same held-out images for both |
| Macro precision | [value] | [value] | Same test labels and class order |
| Macro recall | [value] | [value] | Same test labels and class order |
| Macro F1 | [value] | [value] | Same test labels and class order |
| Training time | [value] | [value] | Record Roboflow-reported elapsed time; PyTorch value from metrics JSON |
| Mean inference latency | [value] ms/image | [value] ms/image | Roboflow client round-trip includes network; PyTorch forward-pass excludes network. Explain this difference. |

Discuss the trade-off between ease of use, model control, repeatability, training time and inference latency. Be clear if latency measurements are not directly equivalent.

## 4. Failure Analysis and Conclusion (approx. 3/4 page)
Inspect misclassified test images from both approaches. Include 2–4 representative edge cases with the image filename, true label, predicted label, confidence, and a short explanation. Consider subtle defects, tiny defects, shadows, wrinkles, unusual lighting, blur, similar-looking textures, crop/scale, and class imbalance—but only claim causes supported by the inspected images. Conclude which approach performed better on the shared test subset and what should be improved next.

## Reproducibility
List the Roboflow project/version, Python version, package versions, split sizes, augmentation settings, random seed, training hyperparameters, hardware, and command used for inference.
