import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torch.nn.functional as F

# 1. Initialize GPU Feature Extractor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).to(device)
model.eval()
extractor = torch.nn.Sequential(*list(model.children())[:-1])

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_crop_embedding(img_path, points):
    if not os.path.exists(img_path): return None
    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])
    
    img = Image.open(img_path).convert('RGB')
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    if crop.width < 5 or crop.height < 5: return None
    
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = extractor(tensor).flatten()
        emb = F.normalize(emb, p=2, dim=0)
    return emb

BASE_DIR = Path(__file__).resolve().parent

def propagate_action(ref_image_name, target_label, action='delete', new_label=None, threshold=0.85, inspection_dir=None):
    if inspection_dir is None:
        inspection_dir = str(BASE_DIR / 'safety_inspection')
    ref_json_path = os.path.join(inspection_dir, os.path.splitext(ref_image_name)[0] + '.json')
    ref_img_path = os.path.join(inspection_dir, ref_image_name)
    
    if not os.path.exists(ref_json_path):
        print(f'Error: Could not find reference JSON {ref_json_path}')
        return
        
    with open(ref_json_path, 'r') as f:
        ref_data = json.load(f)
        
    # Find target box crop in reference image
    target_shape = None
    for shape in ref_data.get('shapes', []):
        if shape['label'] == target_label:
            target_shape = shape
            break
            
    if not target_shape:
        print(f'No shape with label "{target_label}" found in {ref_image_name}')
        return
        
    ref_emb = get_crop_embedding(ref_img_path, target_shape['points'])
    if ref_emb is None:
        print('Failed to extract embedding from reference box.')
        return
        
    print(f'Extracted embedding vector. Scanning remaining JSONs for visual similarity >= {threshold:.0%}...')
    
    total_matches = 0
    modified_files = 0
    
    for fname in os.listdir(inspection_dir):
        if not fname.endswith('.json'): continue
        json_path = os.path.join(inspection_dir, fname)
        base_name = os.path.splitext(fname)[0]
        
        # Find matching image file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            possible = os.path.join(inspection_dir, base_name + ext)
            if os.path.exists(possible):
                img_path = possible
                break
        if not img_path: continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        kept_shapes = []
        file_changed = False
        
        for shape in data.get('shapes', []):
            cand_emb = get_crop_embedding(img_path, shape['points'])
            if cand_emb is not None:
                similarity = torch.dot(ref_emb, cand_emb).item()
                if similarity >= threshold:
                    total_matches += 1
                    file_changed = True
                    print(f'  [Match {similarity:.1%}] in {fname} ({shape["label"]})')
                    
                    if action == 'delete':
                        continue  # Skip adding to kept_shapes (effectively deletes it)
                    elif action == 'relabel' and new_label:
                        shape['label'] = new_label
                        
            kept_shapes.append(shape)
            
        if file_changed:
            data['shapes'] = kept_shapes
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            modified_files += 1
            
    print(f'\nDone! Processed action "{action}" across {modified_files} files ({total_matches} total boxes updated).')
