"""Generate the 3-page fabric-defect assignment PDF from saved experiment outputs.

Run from the project root:
    python generate_report.py

Required input: outputs/model_comparison_summary.json
Optional inputs: outputs/misclassified_images.csv, outputs/failure_analysis_contact_sheet.jpg,
                 outputs/torch_metrics.json
Output: reports/assignment_report.pdf
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepTogether, HRFlowable
)

ROOT = Path.cwd()
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = OUT / "model_comparison_summary.json"
FAILURES_PATH = OUT / "misclassified_images.csv"
CONTACT_SHEET_PATH = OUT / "failure_analysis_contact_sheet.jpg"
TORCH_FULL_METRICS_PATH = OUT / "torch_metrics.json"
PDF_PATH = REPORTS / "assignment_report.pdf"

# Recorded from the completed runs shown in the experiment terminal output.
ROBOFLOW_MEAN_CLIENT_SECONDS = 2.45
PYTORCH_FORWARD_MS_PER_IMAGE = 50.73
ROBOFLOW_TRAIN_TIME_NOTE = "Roboflow UI estimated 1 h 48 min before training; actual wall-clock duration was not recorded."
PYTORCH_TRAIN_TIME_NOTE = "Four epochs completed in 101.8 minutes; training was interrupted during Epoch 5. The planned 10 epochs were not completed."
PYTORCH_FULL_TEST_FALLBACK = {
    "accuracy": 0.9638,
    "macro_precision": 0.9376,
    "macro_recall": 0.9168,
    "macro_f1": 0.9267,
    "test_images": 497,
}

NAVY = colors.HexColor("#17253B")
BLUE = colors.HexColor("#2878B5")
PALE_BLUE = colors.HexColor("#EAF3FA")
TEAL = colors.HexColor("#147D79")
PALE_TEAL = colors.HexColor("#E8F5F3")
INK = colors.HexColor("#253247")
MUTED = colors.HexColor("#5D6B7D")
LIGHT = colors.HexColor("#F4F7FA")
BORDER = colors.HexColor("#D9E1EA")
RED = colors.HexColor("#B33A3A")
PALE_RED = colors.HexColor("#FCEEEE")


def pct(v):
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_failures(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clean_label(value):
    return str(value or "").strip()


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    # Header band and section line.
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 11 * mm, width, 11 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(17 * mm, height - 7.2 * mm, "FABRIC DEFECT CLASSIFICATION | ASSIGNMENT 1")
    # Footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(17 * mm, 14 * mm, width - 17 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(17 * mm, 8.5 * mm, "Experimental report - measured outputs and stated limitations")
    canvas.drawRightString(width - 17 * mm, 8.5 * mm, f"Page {doc.page} of 3")
    canvas.restoreState()


def make_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="ReportTitle", parent=ss["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, alignment=TA_LEFT, textColor=NAVY,
        spaceAfter=5 * mm,
    ))
    ss.add(ParagraphStyle(
        name="SubTitle", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10, leading=14, textColor=MUTED, spaceAfter=4 * mm,
    ))
    ss.add(ParagraphStyle(
        name="H1x", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=3 * mm,
    ))
    ss.add(ParagraphStyle(
        name="H2x", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
    ))
    ss.add(ParagraphStyle(
        name="Bodyx", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=8.6, leading=12, textColor=INK, spaceAfter=2.2 * mm,
    ))
    ss.add(ParagraphStyle(
        name="Smallx", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=7.3, leading=9.3, textColor=MUTED, spaceAfter=1.2 * mm,
    ))
    ss.add(ParagraphStyle(
        name="Calloutx", parent=ss["BodyText"], fontName="Helvetica-Bold",
        fontSize=10, leading=14, textColor=NAVY,
    ))
    ss.add(ParagraphStyle(
        name="Cellx", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=7.3, leading=9.2, textColor=INK,
    ))
    ss.add(ParagraphStyle(
        name="CellBoldx", parent=ss["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.3, leading=9.2, textColor=NAVY,
    ))
    ss.add(ParagraphStyle(
        name="CellHeaderx", parent=ss["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.3, leading=9.2, textColor=colors.white,
    ))
    ss.add(ParagraphStyle(
        name="CenterSmall", parent=ss["BodyText"], fontName="Helvetica",
        fontSize=7.3, leading=9, alignment=TA_CENTER, textColor=INK,
    ))
    return ss


def P(text, style):
    return Paragraph(text, style)


def styled_table(data, col_widths, header=True, font_size=7.5, row_bgs=None):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    for r, bg in (row_bgs or {}).items():
        commands.append(("BACKGROUND", (0, r), (-1, r), bg))
    t.setStyle(TableStyle(commands))
    return t


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}. Run 'python compare_models.py' first, then rerun this script."
        )

    summary = load_json(SUMMARY_PATH, {})
    rf = summary.get("roboflow", {})
    pt = summary.get("pytorch", {})
    paired = summary.get("paired_comparison", {})
    failures = load_failures(FAILURES_PATH)
    torch_full = load_json(TORCH_FULL_METRICS_PATH, PYTORCH_FULL_TEST_FALLBACK) or PYTORCH_FULL_TEST_FALLBACK

    n = summary.get("images_matched_and_compared", 50)
    rf_correct = rf.get("correct", 45)
    pt_correct = pt.get("correct", 45)
    rf_acc = rf.get("accuracy", rf_correct / n if n else 0)
    pt_acc = pt.get("accuracy", pt_correct / n if n else 0)
    both_correct = paired.get("both_correct", min(rf_correct, pt_correct))
    both_wrong = paired.get("both_wrong", max(0, n - both_correct))
    agree_count = paired.get("predicted_label_agreement_count", n)
    mismatch_union = summary.get("misclassified_image_count_union", len(failures))

    REPORTS.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm,
        title="Automated Fabric Defect Classification: Roboflow vs PyTorch",
        author="Fabric Defect Detection Assignment",
    )
    s = make_styles()
    story = []

    # PAGE 1: objective and methods
    story.append(Spacer(1, 2 * mm))
    story.append(P("Automated Fabric Defect Classification", s["ReportTitle"]))
    story.append(P("Roboflow-hosted ResNet18 vs locally fine-tuned PyTorch ResNet18", s["SubTitle"]))
    hero = Table([[P(
        f"<font color='#2878B5'>COMMON 50-IMAGE TEST SET</font><br/>"
        f"Roboflow accuracy: <b>{pct(rf_acc)}</b>  |  PyTorch accuracy: <b>{pct(pt_acc)}</b>", s["Calloutx"]
    )]], colWidths=[176 * mm])
    hero.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(hero)
    story.append(Spacer(1, 3 * mm))
    story.append(P("1. Objective and methodology", s["H1x"]))
    story.append(P(
        "The objective was to build and compare a hosted image-classification model in Roboflow with a local PyTorch baseline for fabric categories. The Roboflow model used a ResNet18 classification architecture. The custom baseline fine-tuned a pretrained ResNet18 in PyTorch. Both models were evaluated on the same labelled subset of 50 held-out test images.",
        s["Bodyx"]
    ))
    facts = [
        [P("Item", s["CellHeaderx"]), P("Recorded configuration", s["CellHeaderx"])],
        [P("Classes", s["Cellx"]), P("Defect free, Defected, Waste (Waste retained after instructor clarification)", s["Cellx"])],
        [P("Initial dataset split", s["Cellx"]), P("3,499 train / 999 validation / 497 test = 4,995 counted images; 5 fewer than the stated target of 5,000.", s["Cellx"])],
        [P("Augmentations", s["Cellx"]), P("Horizontal flip and brightness. Exported training folder contained 10,445 images after augmented training variants were included; validation 999 and test 497.", s["Cellx"])],
        [P("Preprocessing", s["Cellx"]), P("Roboflow Version 1, 224 x 224 resize shown in the training settings; classification workflow.", s["Cellx"])],
        [P("Common comparison subset", s["Cellx"]), P("50 images: 5 Defect free, 44 Defected, 1 Waste.", s["Cellx"])],
    ]
    story.append(styled_table(facts, [39 * mm, 137 * mm], header=True, row_bgs={2: LIGHT, 4: LIGHT}))
    story.append(Spacer(1, 3 * mm))
    story.append(P("Evaluation approach", s["H2x"]))
    story.append(P(
        "Roboflow predictions were obtained through its hosted Workflow API and saved to CSV. The local model's saved best checkpoint was evaluated without retraining. The comparison script matched rows by image filename, then computed accuracy and class-averaged precision, recall and F1-score.",
        s["Bodyx"]
    ))
    story.append(P(
        "Dataset caveat: the 50-image subset is highly imbalanced. In particular, Waste has only one test example, so its class-specific metrics cannot support a reliable general conclusion.", s["Smallx"]
    ))
    story.append(PageBreak())

    # PAGE 2: results and training / latency caveats
    story.append(Spacer(1, 2 * mm))
    story.append(P("2. Benchmark results", s["H1x"]))
    results = [
        [P("Metric on same 50 images", s["CellHeaderx"]), P("Roboflow", s["CellHeaderx"]), P("PyTorch", s["CellHeaderx"])],
        [P("Correct predictions", s["Cellx"]), P(f"{rf_correct}/{n}", s["Cellx"]), P(f"{pt_correct}/{n}", s["Cellx"])],
        [P("Accuracy", s["Cellx"]), P(pct(rf_acc), s["Cellx"]), P(pct(pt_acc), s["Cellx"])],
        [P("Macro precision", s["Cellx"]), P(pct(rf.get("macro_precision")), s["Cellx"]), P(pct(pt.get("macro_precision")), s["Cellx"])],
        [P("Macro recall", s["Cellx"]), P(pct(rf.get("macro_recall")), s["Cellx"]), P(pct(pt.get("macro_recall")), s["Cellx"])],
        [P("Macro F1-score", s["Cellx"]), P(pct(rf.get("macro_f1")), s["Cellx"]), P(pct(pt.get("macro_f1")), s["Cellx"])],
        [P("Mean inference latency", s["Cellx"]), P("2.45 s/image (client/API round trip)", s["Cellx"]), P("50.73 ms/image (batched local forward pass)", s["Cellx"])],
    ]
    story.append(styled_table(results, [64 * mm, 56 * mm, 56 * mm], header=True, row_bgs={2: PALE_BLUE, 4: LIGHT, 6: PALE_TEAL}))
    story.append(Spacer(1, 3 * mm))
    story.append(P("Interpretation", s["H2x"]))
    story.append(P(
        f"On the same subset, both systems achieved {pct(rf_acc)} accuracy and made the same predicted class decision on {agree_count}/{n} images. Each was correct on {both_correct} images and incorrect on {both_wrong}. The result does not show an accuracy winner on this small sample. Identical class predictions should be treated as an observed result for this subset, not proof that the models are equivalent on other data.",
        s["Bodyx"]
    ))
    story.append(P("Additional PyTorch full-test result", s["H2x"]))
    story.append(P(
        f"The saved PyTorch checkpoint was also evaluated on the full exported test folder ({torch_full.get('test_images', 497)} images): accuracy {pct(torch_full.get('accuracy'))}, macro precision {pct(torch_full.get('macro_precision'))}, macro recall {pct(torch_full.get('macro_recall'))}, and macro F1 {pct(torch_full.get('macro_f1'))}. This result uses a different evaluation size than the 50-image paired comparison and should not be compared directly with Roboflow's 50-image score.",
        s["Bodyx"]
    ))
    story.append(P("Training time and latency limitations", s["H2x"]))
    timing = [
        [P("Roboflow", s["CellBoldx"]), P(ROBOFLOW_TRAIN_TIME_NOTE, s["Cellx"])],
        [P("PyTorch", s["CellBoldx"]), P(PYTORCH_TRAIN_TIME_NOTE, s["Cellx"])],
        [P("Latency scope", s["CellBoldx"]), P("Roboflow's 2.45 seconds includes client-side upload/network/API round trip and remote processing. PyTorch's 50.73 ms is the batched local forward-pass measurement. These are different scopes; they should not be presented as a like-for-like speed test.", s["Cellx"])],
    ]
    story.append(styled_table(timing, [31 * mm, 145 * mm], header=False, row_bgs={1: LIGHT}))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        "Reproducibility note: the PyTorch run completed four epochs and was interrupted during epoch 5 because CPU training was slow. The saved best checkpoint was evaluated, but the planned 10-epoch run was not completed. Roboflow's exact wall-clock training time was not captured; its UI showed an estimate of 1 hour 48 minutes before training.",
        s["Smallx"]
    ))
    story.append(PageBreak())

    # PAGE 3: failure analysis and contact sheet
    story.append(Spacer(1, 2 * mm))
    story.append(P("3. Failure analysis and conclusion", s["H1x"]))
    story.append(P(
        f"The paired comparison identified {mismatch_union} misclassified images. Both models made the same five errors: three images labelled Defected were predicted as Defect free, and two images labelled Defect free were predicted as Defected. No Waste error occurred in this subset, but that class was represented by only one image.",
        s["Bodyx"]
    ))
    story.append(P("Misclassified examples", s["H2x"]))
    fail_data = [[P("Image (shortened)", s["CellHeaderx"]), P("True", s["CellHeaderx"]), P("Predicted", s["CellHeaderx"]), P("Roboflow confidence", s["CellHeaderx"])]]
    # Use filename/confidence from CSV when available; keep a transparent fallback from terminal output.
    fallback = [
        ("Defected-1176...", "Defected", "Defect free", "91.94%"),
        ("Defected-1232...", "Defected", "Defect free", "89.76%"),
        ("Defected-1263...", "Defected", "Defect free", "94.02%"),
        ("defect-free-15...", "Defect free", "Defected", "99.86%"),
        ("defect-free-229...", "Defect free", "Defected", "98.50%"),
    ]
    if failures:
        for row in failures[:5]:
            name = clean_label(row.get("image_name", ""))
            if len(name) > 27:
                name = name[:24] + "..."
            conf = row.get("roboflow_confidence", "")
            try:
                conf = f"{float(conf) * 100:.2f}%"
            except (TypeError, ValueError):
                conf = str(conf or "N/A")
            fail_data.append([
                P(escape(name), s["Cellx"]),
                P(escape(clean_label(row.get("true_label", "N/A"))), s["Cellx"]),
                P(escape(clean_label(row.get("roboflow_predicted_label", row.get("predicted_label", "N/A")))), s["Cellx"]),
                P(escape(conf), s["Cellx"]),
            ])
    else:
        for name, true_lab, pred_lab, conf in fallback:
            fail_data.append([P(escape(name), s["Cellx"]), P(escape(true_lab), s["Cellx"]), P(escape(pred_lab), s["Cellx"]), P(conf, s["Cellx"])])
    failures_table = styled_table(fail_data, [59 * mm, 31 * mm, 39 * mm, 47 * mm], header=True, row_bgs={2: LIGHT, 4: LIGHT})
    story.append(failures_table)
    story.append(Spacer(1, 2 * mm))

    analysis_text = (
        "Visual review of the contact sheet shows relatively uniform fabric textures at the displayed scale; the defects are not clearly visible in the small previews. Possible explanations include subtle texture differences, ambiguous labels, lighting/shading variation, or limited representation of some visual patterns. These are hypotheses, not confirmed causes: full-resolution image inspection and label verification would be needed to determine the reason. Both models were highly confident on several wrong predictions, so confidence alone did not guarantee correctness."
    )
    story.append(P(analysis_text, s["Smallx"]))

    # Contact sheet image with bounded dimensions; no image is embedded if missing.
    if CONTACT_SHEET_PATH.exists():
        im = RLImage(str(CONTACT_SHEET_PATH))
        max_w, max_h = 77 * mm, 89 * mm
        scale = min(max_w / im.imageWidth, max_h / im.imageHeight)
        im.drawWidth = im.imageWidth * scale
        im.drawHeight = im.imageHeight * scale
        # Put image and conclusion beside each other.
        conclusion = [
            P("Conclusion", s["H2x"]),
            P(
                "On the paired 50-image subset, Roboflow and PyTorch both scored 90.00% and had identical predicted labels. This subset is small and imbalanced, so the result is preliminary. The full PyTorch test score was 96.38% on 497 images, but a comparable Roboflow score on the same 497 images was not generated. Future work should use more labelled Waste examples, verify borderline labels, preserve identical test splits, and record actual end-to-end training and inference times.",
                s["Smallx"]
            ),
            Spacer(1, 1.5 * mm),
            P("Final limitations", s["H2x"]),
            P(
                "The 3-class setup follows the instructor's clarification to retain Waste, although the brief initially described two classes. The PyTorch run was stopped before all 10 planned epochs; full training time is therefore not available. No unmeasured timing or causal explanation has been invented.",
                s["Smallx"]
            ),
        ]
        side = Table([[im, conclusion]], colWidths=[79 * mm, 97 * mm], hAlign="LEFT")
        side.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(side)
    else:
        story.append(P("Conclusion", s["H2x"]))
        story.append(P(
            "Both models scored 90.00% on the paired 50-image subset. The result is preliminary because the subset is small and imbalanced. More labelled Waste examples, label verification, and consistent timing measurements are recommended.",
            s["Bodyx"]
        ))

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(f"PDF report created: {PDF_PATH.resolve()}")
    print("Report is designed to be 3 pages and uses saved experiment outputs.")


if __name__ == "__main__":
    main()
