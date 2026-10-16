import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    # Initialize YOLO student model with base weights
    model_weights = BASE_DIR / 'yolo11n.pt'
    model = YOLO(str(model_weights) if model_weights.exists() else 'yolo11n.pt')
    
    student_yaml = BASE_DIR / 'student_dataset' / 'data.yaml'
    if not student_yaml.exists():
        student_yaml = BASE_DIR / 'data.yaml'

    model.train(
        data=str(student_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(BASE_DIR / 'runs' / 'detect'),
        name='student_model',
        exist_ok=True
    )
