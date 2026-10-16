import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import io
import base64
import json
import shutil
import subprocess
import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw
import torch.nn.functional as F
from ultralytics import YOLO

# --- STREAMLIT CANVAS BASE64 PATCH ---
import streamlit.elements.image as st_image

def base64_image_to_url(image, width=-1, clamp=False, channels="RGB", output_format="PNG", image_id=""):
    if isinstance(image, str):
        return image
    buffered = io.BytesIO()
    if hasattr(image, 'save'):
        image.save(buffered, format="PNG")
    else:
        Image.fromarray(image).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

st_image.image_to_url = base64_image_to_url

from streamlit_drawable_canvas import st_canvas

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(BASE_DIR / 'runs' / 'detect' / 'student_model' / 'weights' / 'best.pt')
DEFAULT_TEST_DIR = str(BASE_DIR / 'test_images')
DEFAULT_INSPECT_DIR = str(BASE_DIR / 'safety_inspection')

@st.cache_resource
def load_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return extractor, device

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def draw_unified_overlay(img_pil, shapes):
    """Draws existing detections and labels onto the image in green."""
    overlay_img = img_pil.copy()
    draw = ImageDraw.Draw(overlay_img)
    for s in shapes:
        pts = s['points']
        x1, y1 = min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])
        x2, y2 = max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])
        lbl = s.get('label', '')
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
        draw.text((x1 + 4, y1 + 4), f" {lbl} ", fill="#FFFFFF")
    return overlay_img

def get_crop_and_embedding(img_path, points, extractor, device):
    if not os.path.exists(img_path): return None, None
    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])
    
    img = Image.open(img_path).convert('RGB')
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    if crop.width < 5 or crop.height < 5: return None, None
    
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = extractor(tensor).flatten()
        emb = F.normalize(emb, p=2, dim=0)
    return crop, emb

def scan_for_new_missing_boxes(img_path, ref_emb, target_points, sim_threshold, existing_shapes, extractor, device, max_margin_pct=0.10):
    """Sliding-window patch scanner spatially restricted to within max_margin_pct (default 10%) of the drawn box."""
    if not os.path.exists(img_path): return []
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    
    ref_x1 = min(target_points[0][0], target_points[1][0])
    ref_y1 = min(target_points[0][1], target_points[1][1])
    ref_x2 = max(target_points[0][0], target_points[1][0])
    ref_y2 = max(target_points[0][1], target_points[1][1])
    
    ref_w = ref_x2 - ref_x1
    ref_h = ref_y2 - ref_y1
    
    if ref_w < 10 or ref_h < 10:
        return []

    # Spatial ROI boundary calculation (Strictly restricted around drawn coordinates)
    margin_w = ref_w * max_margin_pct
    margin_h = ref_h * max_margin_pct
    
    roi_x1 = max(0, int(ref_x1 - margin_w))
    roi_y1 = max(0, int(ref_y1 - margin_h))
    roi_x2 = min(W, int(ref_x2 + margin_w))
    roi_y2 = min(H, int(ref_y2 + margin_h))
    
    # Restrict scale variations to ±10%
    scales = [0.9, 1.0, 1.1]
    candidate_boxes = []
    crops_tensors = []
    
    for scale in scales:
        bw = int(ref_w * scale)
        bh = int(ref_h * scale)
        if bw < 10 or bh < 10 or bw > W or bh > H: continue
        
        step_x = max(4, int(bw * 0.15))
        step_y = max(4, int(bh * 0.15))
        
        y_start = roi_y1
        y_end = max(roi_y1 + 1, roi_y2 - bh + 1)
        x_start = roi_x1
        x_end = max(roi_x1 + 1, roi_x2 - bw + 1)
        
        for y in range(y_start, y_end, step_y):
            for x in range(x_start, x_end, step_x):
                crop = img.crop((x, y, x + bw, y + bh))
                candidate_boxes.append([x, y, x + bw, y + bh])
                crops_tensors.append(transform(crop))
                
    if not crops_tensors:
        return []
        
    batch_size = 64
    sim_scores = []
    for i in range(0, len(crops_tensors), batch_size):
        batch = torch.stack(crops_tensors[i:i+batch_size]).to(device)
        with torch.no_grad():
            embs = extractor(batch).squeeze(-1).squeeze(-1)
            embs = F.normalize(embs, p=2, dim=1)
            sims = torch.matmul(embs, ref_emb.unsqueeze(1)).squeeze(1)
            sim_scores.extend(sims.cpu().tolist())
            
    passed_boxes = []
    passed_scores = []
    for box, score in zip(candidate_boxes, sim_scores):
        if score >= sim_threshold:
            overlap_existing = False
            for s in existing_shapes:
                pts = s['points']
                ex_box = [min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1]),
                          max(pts[0][0], pts[1][0]), max(pts[0][1], pts[1][1])]
                xA = max(box[0], ex_box[0]); yA = max(box[1], ex_box[1])
                xB = min(box[2], ex_box[2]); yB = min(box[3], ex_box[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                areaA = (box[2] - box[0]) * (box[3] - box[1])
                areaB = (ex_box[2] - ex_box[0]) * (ex_box[3] - ex_box[1])
                iou = inter / float(areaA + areaB - inter + 1e-6)
                if iou > 0.35:
                    overlap_existing = True
                    break
            if not overlap_existing:
                passed_boxes.append(box)
                passed_scores.append(score)
                
    if not passed_boxes:
        return []
        
    boxes_t = torch.tensor(passed_boxes, dtype=torch.float32)
    scores_t = torch.tensor(passed_scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.3)
    
    return [passed_boxes[k] for k in keep.tolist()]

def get_indexed_files(directory):
    if not os.path.exists(directory): return {}
    files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    mapping = {}
    for idx, f_json in enumerate(files, start=1):
        base = os.path.splitext(f_json)[0]
        img_name = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            if os.path.exists(os.path.join(directory, base + ext)):
                img_name = base + ext
                break
        if img_name:
            mapping[idx] = {'json': f_json, 'image': img_name, 'base': base}
    return mapping

# --- SIDEBAR ---
st.sidebar.title("🪖 Safety MITL Control Center")
cuda_status = "🟢 GPU Active (RTX 3050)" if torch.cuda.is_available() else "⚠️ CPU Mode"
st.sidebar.caption(f"Hardware Status: **{cuda_status}**")
st.sidebar.divider()

model_path = st.sidebar.text_input("YOLO Model Weights", DEFAULT_MODEL)
inspection_dir = st.sidebar.text_input("Inspection Directory", DEFAULT_INSPECT_DIR)

st.sidebar.divider()
if st.sidebar.button("🎨 Launch External Labelme GUI"):
    try:
        subprocess.Popen(["labelme", inspection_dir])
        st.sidebar.success("Launched Labelme!")
    except Exception as e:
        st.sidebar.error(f"Failed to launch Labelme: {e}")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 1. Run Batch Prediction", "⚡ 2. Visual Embedding MITL Engine", "📊 3. Batch Analytics"])

# ==========================================
# TAB 1: BATCH INFERENCE & JSON GENERATION
# ==========================================
with tab1:
    st.header("Generate Predictions & Clean Labelme JSONs")
    col1, col2 = st.columns(2)
    with col1:
        source_dir = st.text_input("Source Image Folder", DEFAULT_TEST_DIR)
    with col2:
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05)
        
    if st.button("▶️ Run Batch Inference", type="primary"):
        if not os.path.exists(source_dir):
            st.error(f"Source folder does not exist: {source_dir}")
        else:
            with st.spinner("Processing images with YOLO..."):
                os.makedirs(inspection_dir, exist_ok=True)
                model = YOLO(model_path)
                results = model.predict(source=source_dir, conf=conf_thresh, save=False)
                
                count = 0
                for result in results:
                    img_path = result.path
                    img_name = os.path.basename(img_path)
                    base_name = os.path.splitext(img_name)[0]
                    
                    dest_img_path = os.path.join(inspection_dir, img_name)
                    shutil.copy(img_path, dest_img_path)
                    
                    orig_h, orig_w = result.orig_shape
                    shapes = []
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        xyxy = box.xyxy[0].tolist()
                        shapes.append({
                            'label': class_name,
                            'points': [[round(xyxy[0], 2), round(xyxy[1], 2)], [round(xyxy[2], 2), round(xyxy[3], 2)]],
                            'group_id': None, 'description': '', 'shape_type': 'rectangle', 'flags': {}, 'mask': None
                        })
                        
                    labelme_data = {
                        'version': '5.0.1', 'flags': {}, 'shapes': shapes,
                        'imagePath': img_name, 'imageData': None, 'imageHeight': orig_h, 'imageWidth': orig_w
                    }
                    with open(os.path.join(inspection_dir, f"{base_name}.json"), 'w') as f:
                        json.dump(labelme_data, f, indent=2)
                    count += 1
                st.success(f"Successfully processed {count} images with clean global serial numbers!")

# ==========================================
# TAB 2: INTERACTIVE BOX DRAWING & PROPAGATION
# ==========================================
with tab2:
    st.header("Visual Similarity Propagation Engine")
    st.write("Reference images and objects using sequential index numbers (**Image #1, #2...**).")
    
    indexed_files = get_indexed_files(inspection_dir)
    if not indexed_files:
        st.warning(f"No inspection files found in `{inspection_dir}`. Run Tab 1 first.")
    else:
        selected_idx = st.selectbox("Select Image Number", list(indexed_files.keys()), 
                                    format_func=lambda i: f"Image #{i} ({indexed_files[i]['image']})")
        
        file_info = indexed_files[selected_idx]
        ref_json_path = os.path.join(inspection_dir, file_info['json'])
        img_path = os.path.join(inspection_dir, file_info['image'])
        
        with open(ref_json_path, 'r') as f:
            ref_data = json.load(f)
        shapes = ref_data.get('shapes', [])
        
        box_source_mode = st.radio("Reference Box Selection Mode", 
                                   ["🖱️ Draw New Box with Mouse", "📋 Pick Existing Detection Box"], 
                                   horizontal=True)
        
        col_left, col_right = st.columns([1.2, 1])
        
        target_points = None
        current_label = "safety_harness"
        
        with col_left:
            img_pil = Image.open(img_path).convert('RGB')
            w_orig, h_orig = img_pil.size
            
            unified_bg = draw_unified_overlay(img_pil, shapes)
            
            if box_source_mode == "🖱️ Draw New Box with Mouse":
                st.subheader("1. Click & Drag to Draw Bounding Box")
                st.caption("Existing labels are highlighted in green. Draw a box around any un-labeled object:")
                
                max_canvas_w = 650
                scale = min(1.0, max_canvas_w / float(w_orig))
                canvas_w = int(w_orig * scale)
                canvas_h = int(h_orig * scale)
                
                bg_resized = unified_bg.resize((canvas_w, canvas_h))
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.2)",
                    stroke_color="#00FF00",
                    stroke_width=2,
                    background_image=bg_resized,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="rect",
                    key=f"canvas_draw_{selected_idx}"
                )
                
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    last_box = canvas_result.json_data["objects"][-1]
                    left = last_box["left"] / scale
                    top = last_box["top"] / scale
                    width = last_box["width"] / scale
                    height = last_box["height"] / scale
                    
                    target_points = [[left, top], [left + width, top + height]]
                    st.success(f"Box Captured: ({left:.0f}, {top:.0f}) to ({left+width:.0f}, {top+height:.0f})")
                else:
                    st.info("Draw a box on the image to capture visual features.")
                    
            else:
                st.subheader("1. Existing Detections Preview")
                st.image(unified_bg, caption=f"Image #{selected_idx}: {file_info['image']}", use_column_width=True)
                
                if shapes:
                    box_labels = [f"Box {i}: '{s['label']}' at ({s['points'][0][0]:.0f}, {s['points'][0][1]:.0f})" for i, s in enumerate(shapes)]
                    sel_box_idx = st.selectbox("Select Bounding Box", range(len(box_labels)), format_func=lambda i: box_labels[i])
                    target_points = shapes[sel_box_idx]['points']
                    current_label = shapes[sel_box_idx]['label']
                else:
                    st.warning("No detection boxes exist for this image yet. Switch to 'Draw New Box' above.")

        with col_right:
            st.subheader("2. Crop Feature & Propagation Control")
            
            extractor, device = load_feature_extractor()
            if target_points:
                crop_img, ref_emb = get_crop_and_embedding(img_path, target_points, extractor, device)
                if crop_img:
                    st.write("**Captured Object Visual Feature:**")
                    st.image(crop_img, width=180)
            
            target_class_label = st.text_input("Assign Target Class Label Name", value=current_label)
            action_type = st.radio("Action Type", ["✨ Add Box & Propagate to Similar Objects", "Relabel Existing Similar Boxes", "Delete Similar Boxes"])
            
            sim_threshold = st.slider("Visual Similarity Threshold", 0.50, 0.98, 0.80, 0.02,
                                      help="Higher = Strict match. Lower = Catch variations across lighting and angles.")
            
            max_drift_pct = st.slider("Max Spatial Search Boundary Drift (%)", 5, 25, 10, 5,
                                      help="Strictly limits box creation to within this percentage margin of the drawn box.") / 100.0
            
            save_to_current = st.checkbox("Also save this drawn box into current image JSON", value=True)
            
            st.divider()
            if st.button("⚡ Save & Execute Batch Propagation", type="primary"):
                if target_points is None:
                    st.error("Please draw or select a bounding box first!")
                else:
                    crop_img, ref_emb = get_crop_and_embedding(img_path, target_points, extractor, device)
                    if ref_emb is None:
                        st.error("Selected box region is too small. Draw a larger bounding box.")
                    else:
                        if save_to_current and box_source_mode == "🖱️ Draw New Box with Mouse":
                            new_shape = {
                                'label': target_class_label,
                                'points': target_points,
                                'group_id': None, 'description': '', 'shape_type': 'rectangle', 'flags': {}, 'mask': None
                            }
                            ref_data['shapes'].append(new_shape)
                            with open(ref_json_path, 'w') as f:
                                json.dump(ref_data, f, indent=2)
                            st.info(f"Saved drawn box '{target_class_label}' to Image #{selected_idx} JSON!")

                        act = 'add' if "Add" in action_type else ('relabel' if "Relabel" in action_type else 'delete')
                        modified_files = 0
                        total_new_boxes = 0
                        total_relabels = 0
                        log_text = []
                        
                        all_indices = list(indexed_files.keys())
                        progress_bar = st.progress(0)
                        
                        for idx, file_id in enumerate(all_indices):
                            f_info = indexed_files[file_id]
                            j_path = os.path.join(inspection_dir, f_info['json'])
                            cur_img_path = os.path.join(inspection_dir, f_info['image'])
                            
                            with open(j_path, 'r') as f:
                                data = json.load(f)
                                
                            file_changed = False
                            kept_shapes = []
                            
                            for s in data.get('shapes', []):
                                _, cand_emb = get_crop_and_embedding(cur_img_path, s['points'], extractor, device)
                                if cand_emb is not None:
                                    sim = torch.dot(ref_emb, cand_emb).item()
                                    if sim >= sim_threshold:
                                        if act == 'delete':
                                            file_changed = True
                                            log_text.append(f"Image #{file_id} [Deleted] box match ({sim:.1%})")
                                            continue
                                        elif act in ['relabel', 'add'] and s['label'] != target_class_label:
                                            s['label'] = target_class_label
                                            file_changed = True
                                            total_relabels += 1
                                            log_text.append(f"Image #{file_id} [Relabeled] existing box ({sim:.1%}) ➔ {target_class_label}")
                                kept_shapes.append(s)
                               
                            if act == 'add':
                                new_boxes = scan_for_new_missing_boxes(
                                    cur_img_path, ref_emb, target_points, 
                                    sim_threshold, kept_shapes, extractor, device, max_margin_pct=max_drift_pct
                                )
                                for nb in new_boxes:
                                    kept_shapes.append({
                                        'label': target_class_label,
                                        'points': [[round(nb[0], 2), round(nb[1], 2)], [round(nb[2], 2), round(nb[3], 2)]],
                                        'group_id': None, 'description': '', 'shape_type': 'rectangle', 'flags': {}, 'mask': None
                                    })
                                    file_changed = True
                                    total_new_boxes += 1
                                    log_text.append(f"Image #{file_id} ✨ [Created Local Box] at ({nb[0]}, {nb[1]})")

                            if file_changed:
                                data['shapes'] = kept_shapes
                                with open(j_path, 'w') as f:
                                    json.dump(data, f, indent=2)
                                modified_files += 1
                                
                            progress_bar.progress((idx + 1) / len(all_indices))
                            
                        st.success(f"Done! Modified **{modified_files} images** (Created **{total_new_boxes} spatially constrained boxes**, updated **{total_relabels} existing boxes**).")
                        with st.expander("View Detailed Execution Logs"):
                            st.text("\n".join(log_text))

# ==========================================
# TAB 3: BATCH ANALYTICS
# ==========================================
with tab3:
    st.header("Dataset & Class Distribution Analytics")
    indexed_files = get_indexed_files(inspection_dir)
    if indexed_files:
        class_counts = {}
        total_boxes = 0
        for f_id in indexed_files:
            with open(os.path.join(inspection_dir, indexed_files[f_id]['json']), 'r') as f:
                data = json.load(f)
                for s in data.get('shapes', []):
                    lbl = s['label']
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1
                    total_boxes += 1
                    
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Indexed Images", len(indexed_files))
        m2.metric("Total Bounding Boxes", total_boxes)
        m3.metric("Unique Classes", len(class_counts))
        
        st.subheader("Class Distribution Breakdown")
        if class_counts:
            st.bar_chart(class_counts)