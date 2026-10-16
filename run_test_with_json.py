import os
import json
import shutil
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    model_path = BASE_DIR / 'runs' / 'detect' / 'student_model' / 'weights' / 'best.pt'
    source_dir = BASE_DIR / 'data' / 'test_images'
    output_dir = BASE_DIR / 'safety_inspection'
    
    if not source_dir.exists():
        source_dir = BASE_DIR / 'test_images'
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
    model = YOLO(str(model_path))
    
    # Predict without burning drawings onto images
    results = model.predict(
        source=str(source_dir),
        conf=0.50,
        save=False
    )
    
    for result in results:
        img_path = result.path
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Copy clean original image to destination
        dest_img_path = output_dir / img_name
        shutil.copy(img_path, dest_img_path)
        
        json_path = output_dir / f'{base_name}.json'
        orig_h, orig_w = result.orig_shape
        
        shapes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            
            shapes.append({
                'label': class_name,
                'points': [
                    [round(xyxy[0], 2), round(xyxy[1], 2)],
                    [round(xyxy[2], 2), round(xyxy[3], 2)]
                ],
                'group_id': None,
                'description': '',
                'shape_type': 'rectangle',
                'flags': {},
                'mask': None
            })
            
        labelme_data = {
            'version': '5.0.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': img_name,
            'imageData': None,
            'imageHeight': orig_h,
            'imageWidth': orig_w
        }
        
        with open(json_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
            
    print(f'Done! Clean images and Labelme JSONs generated in {output_dir}')
