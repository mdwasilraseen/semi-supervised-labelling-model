import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    # Load trained Teacher Model weights
    teacher_weights = BASE_DIR / 'runs' / 'detect' / 'teacher_model' / 'weights' / 'best.pt'
    if not teacher_weights.exists():
        raise FileNotFoundError(f"Teacher model weights not found at: {teacher_weights}")

    model = YOLO(str(teacher_weights))
    
    unlabelled_dir = BASE_DIR / 'data' / 'unlabelled'
    output_dir = BASE_DIR / 'pseudo_labeled_output'
    
    # Predict and generate YOLO format label text files on unseen batch
    results = model.predict(
        source=str(unlabelled_dir),
        conf=0.60,
        save_txt=True,
        save_conf=False,
        project=str(BASE_DIR),
        name='pseudo_labeled_output',
        exist_ok=True
    )
    print('Pseudo-labeling complete! Labels exported to pseudo_labeled_output/labels')
