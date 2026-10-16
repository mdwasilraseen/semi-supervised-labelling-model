import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Source directories
split_dir = BASE_DIR / 'split_dataset'
unlabelled_dir = BASE_DIR / 'data' / 'unlabelled'
if not unlabelled_dir.exists():
    unlabelled_dir = BASE_DIR / 'unlabelled'

pseudo_labels_dir = BASE_DIR / 'pseudo_labeled_output' / 'labels'
student_dir = BASE_DIR / 'student_dataset'

def merge_datasets():
    if student_dir.exists():
        shutil.rmtree(student_dir)
        
    if split_dir.exists():
        shutil.copytree(split_dir, student_dir)
    else:
        student_dir.mkdir(parents=True, exist_ok=True)
        (student_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (student_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)

    image_map = {}
    if unlabelled_dir.exists():
        for root, _, files in os.walk(unlabelled_dir):
            for f in files:
                name_without_ext, ext = os.path.splitext(f)
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    image_map[name_without_ext] = os.path.join(root, f)

    added_count = 0
    if pseudo_labels_dir.exists():
        for txt_file in os.listdir(pseudo_labels_dir):
            if not txt_file.endswith('.txt'):
                continue
            base_name = os.path.splitext(txt_file)[0]
            if base_name in image_map:
                img_path = image_map[base_name]
                ext = os.path.splitext(img_path)[1]
                shutil.copy(img_path, student_dir / 'train' / 'images' / f"{base_name}{ext}")
                shutil.copy(pseudo_labels_dir / txt_file, student_dir / 'train' / 'labels' / txt_file)
                added_count += 1

    # Copy / update data.yaml configuration
    data_yaml_path = BASE_DIR / 'data.yaml'
    if data_yaml_path.exists():
        shutil.copy(data_yaml_path, student_dir / 'data.yaml')

    print(f'Done! Successfully added {added_count} pseudo-labeled images to the student training set.')

if __name__ == '__main__':
    merge_datasets()
