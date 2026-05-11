import streamlit as st
import torch
import torch.nn.functional as F
import open_clip
import hnswlib
import json
import os
from PIL import Image
from ultralytics import YOLO

# --- APP CONFIG ---
st.set_page_config(page_title="AI Fashion Search", layout="wide")

# Automatically use GPU if available, otherwise CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CACHED AI LOADING ---
# @st.cache_resource prevents reloading the heavy models every time the user clicks
@st.cache_resource
def load_ai_pipeline():
    # 1. Load Fine-Tuned CLIP
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.load_state_dict(torch.load("models/clip_ft_C_seed546_best.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    # 2. Load YOLO (will auto-download yolov8n.pt if missing)
    yolo = YOLO('yolov8n.pt')
    
    # 3. Load HNSW Index & Metadata
    idx = hnswlib.Index(space='cosine', dim=512)
    idx.load_index("index/config_C/hnsw.bin")
    idx.set_ef(150)
    
    meta = json.load(open("index/config_C/config.json"))
    gallery_paths = meta['paths']
    
    return model, preprocess, yolo, idx, gallery_paths

# --- MAIN UI ---
st.title("👗 AI Fashion Search Engine")
st.write("Upload a photo of an outfit to find visually similar items in our database.")

with st.spinner("Warming up AI Models..."):
    model, preprocess, yolo, idx, gallery_paths = load_ai_pipeline()

# File Uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. The original YOLO toggle
    use_yolo = st.checkbox("Use YOLO Auto-Cropping", value=True, help="Turn this off if your image is already a clean photo.")
    
    # 2. >>> NEW: The TA's Requirement - Radio Buttons <<<
    search_focus = st.radio(
        "What do you want to search for?",
        ["Full Body (Default)", "Upper Body (Shirts/Jackets)", "Lower Body (Pants/Skirts)"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Target")
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, use_container_width=True)
        
    with col2:
        st.subheader("Top Matches")
        with st.spinner("Analyzing and searching..."):
            
            crop = query_image # Default to the full image
            
            # 1. Optional YOLO Cropping
            if use_yolo:
                results = yolo(query_image, verbose=False)
                boxes = results[0].boxes
                best_box, best_area = None, 0
                
                # Find the largest person in the photo
                for i, cls in enumerate(boxes.cls.cpu().numpy()):
                    if int(cls) == 0:  # Class 0 is 'person'
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area = area
                            best_box = (int(x1), int(y1), int(x2), int(y2))
                
                # >>> NEW: Geometric Math to slice the box <<<
                if best_box:
                    x1, y1, x2, y2 = best_box
                    box_height = y2 - y1
                    
                    if "Upper Body" in search_focus:
                        # Keep the top 60% of the box (Head to waist)
                        y2 = y1 + (box_height * 0.60)
                    elif "Lower Body" in search_focus:
                        # Keep the bottom 55% of the box (Waist to feet)
                        y1 = y1 + (box_height * 0.45)
                    
                    # Apply the final crop!
                    crop = query_image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            # Show the user exactly what the AI is about to search for
            with col1:
                st.write("**What the AI is actually searching for:**")
                st.image(crop, width=150)
            
            # 2. Extract Features
            x = preprocess(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = F.normalize(model.encode_image(x), dim=-1).cpu().numpy().squeeze()
            
            # 3. Search the Database
            labels, distances = idx.knn_query(emb, k=6)
            
            # 4. Display Results
            result_cols = st.columns(3)
            for i, label in enumerate(labels[0]):
                original_path = gallery_paths[label]
                filename = original_path.replace("/", "__")
                local_image_path = os.path.join("gallery_images", filename)
                
                with result_cols[i % 3]:
                    try:
                        res_img = Image.open(local_image_path)
                        st.image(res_img, use_container_width=True)
                        st.caption(f"Match {i+1} (Score: {1 - distances[0][i]:.2f})")
                    except FileNotFoundError:
                        st.error(f"Missing: {filename}")