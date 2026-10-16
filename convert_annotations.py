import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
SPLIT_DIR = SCRIPT_DIR / "split_dataset"

# 1. Define ALL unique labels present across your dataset
CLASSES = [
    "aframeladder",
    "blockade",
    "boomlift",
    "boot",
    "bumpcap",
    "bus",
    "car",
    "cone",
    "crane",
    "excavator",
    "fire",
    "forklift",
    "glove",
    "hardhat",
    "lifeline",
    "lifevest",
    "loader",
    "longvest",
    "mixer",
    "mobile_phone",
    "motorcycle",
    "overalls",
    "pilier",
    "roller",
    "safetyharness",
    "scissorlift",
    "smoke",
    "truck",
    "vest",
    "worker",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_labelme_json(json_path: Path, output_txt_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")

    if not img_w or not img_h:
        return

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape.get("label")

        # Skip if label isn't in our defined list
        if label not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[label]
        points = shape.get("points", [])

        if not points:
            continue

        # Extract min and max x, y coordinates from "points"
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Calculate bounding box width, height, and center coordinates
        box_w = x_max - x_min
        box_h = y_max - y_min
        x_center = x_min + (box_w / 2.0)
        y_center = y_min + (box_h / 2.0)

        # Normalize relative to image dimensions (0.0 to 1.0)
        norm_x = max(0.0, min(1.0, x_center / img_w))
        norm_y = max(0.0, min(1.0, y_center / img_h))
        norm_w = max(0.0, min(1.0, box_w / img_w))
        norm_h = max(0.0, min(1.0, box_h / img_h))

        yolo_lines.append(
            f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    # Save YOLO formatted annotations
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


# 2. Process both train and val splits
for split in ["train", "val"]:
    json_dir = SPLIT_DIR / split / "annotations"
    txt_out_dir = SPLIT_DIR / split / "labels"
    txt_out_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    for j_file in json_files:
        txt_file = txt_out_dir / f"{j_file.stem}.txt"
        convert_labelme_json(j_file, txt_file)

    print(f"Converted {len(json_files)} JSON files in {split} -> {txt_out_dir}")

# 3. Create the data.yaml file needed by Ultralytics YOLO
yaml_content = f"path: {SPLIT_DIR.as_posix()}\ntrain: train/images\nval: val/images\n\nnames:\n"
for idx, name in enumerate(CLASSES):
    yaml_content += f"  {idx}: {name}\n"

yaml_path = SPLIT_DIR / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\nCreated dataset configuration at: {yaml_path}")