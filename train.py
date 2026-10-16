import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    # Initialize YOLO model with pretrained weights
    model_weights = BASE_DIR / 'yolo11n.pt'
    model = YOLO(str(model_weights) if model_weights.exists() else 'yolo11n.pt')
    
    # Train teacher model on baseline annotated dataset
    data_yaml = BASE_DIR / 'data.yaml'
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(BASE_DIR / 'runs' / 'detect'),
        name='teacher_model',
        exist_ok=True
    )
