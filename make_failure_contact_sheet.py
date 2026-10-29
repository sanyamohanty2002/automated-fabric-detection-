"""Create a contact sheet for the misclassified test50 images.

Run from the project root:
    python make_failure_contact_sheet.py --data-dir data/test50 --csv outputs/misclassified_images.csv

Output:
    outputs/failure_analysis_contact_sheet.jpg
"""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/test50")
    parser.add_argument("--csv", default="outputs/misclassified_images.csv")
    parser.add_argument("--output", default="outputs/failure_analysis_contact_sheet.jpg")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")
    if not csv_path.exists():
        raise SystemExit(f"Misclassified CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("No rows found in misclassified CSV.")

    for required in ("image_name", "true_label", "roboflow_predicted_label", "pytorch_predicted_label"):
        if required not in rows[0]:
            raise SystemExit(f"Expected column '{required}' is missing from {csv_path}.")

    files_by_name = {p.name: p for p in data_dir.rglob("*") if p.is_file()}
    font_title = load_font(20)
    font_body = load_font(16)

    card_w, card_h = 580, 470
    cols = 2
    rows_count = (len(rows) + cols - 1) // cols
    margin, gap, header_h = 24, 20, 46
    canvas = Image.new("RGB", (margin * 2 + cols * card_w + gap * (cols - 1), margin * 2 + rows_count * card_h + gap * (rows_count - 1)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, row in enumerate(rows):
        file_name = row["image_name"]
        image_path = files_by_name.get(file_name)
        if image_path is None:
            # Skip path-missing images, but keep a visible placeholder card.
            image = Image.new("RGB", (card_w - 24, card_h - header_h - 32), "#dddddd")
            img_note = "Image file not found in data/test50"
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                img_note = ""
            except Exception as exc:
                image = Image.new("RGB", (card_w - 24, card_h - header_h - 32), "#dddddd")
                img_note = f"Unable to open image: {exc}"

        x = margin + (i % cols) * (card_w + gap)
        y = margin + (i // cols) * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=12, outline="#999999", width=2, fill="#f7f7f7")
        short_name = file_name if len(file_name) <= 57 else file_name[:54] + "..."
        draw.text((x + 12, y + 10), short_name, fill="black", font=font_title)

        max_w, max_h = card_w - 24, 300
        image = ImageOps.contain(image, (max_w, max_h))
        image_x = x + (card_w - image.width) // 2
        image_y = y + header_h
        canvas.paste(image, (image_x, image_y))

        true_label = row.get("true_label", "")
        rf_label = row.get("roboflow_predicted_label", "")
        pt_label = row.get("pytorch_predicted_label", "")
        confidence = row.get("roboflow_confidence", "")
        text_y = y + header_h + max_h + 12
        draw.text((x + 12, text_y), f"True: {true_label}", fill="#222222", font=font_body)
        draw.text((x + 12, text_y + 25), f"Roboflow: {rf_label}  (confidence: {confidence})", fill="#9b1c1c", font=font_body)
        draw.text((x + 12, text_y + 50), f"PyTorch: {pt_label}", fill="#173e8c", font=font_body)
        if img_note:
            draw.text((x + 12, text_y + 75), img_note[:68], fill="#666666", font=font_body)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    print(f"Contact sheet saved: {output_path.resolve()}")
    print(f"Failure images included: {len(rows)}")
    if any(row["image_name"] not in files_by_name for row in rows):
        print("Warning: at least one image was not found under data/test50; check folder names/paths.")


if __name__ == "__main__":
    main()
