import os

# HuggingFace cache
os.environ["HF_HOME"] = "D:/VR_Final_Proj/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/VR_Final_Proj/hf_cache"
os.environ["TORCH_HOME"] = "D:/VR_Final_Proj/torch_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/VR_Final_Proj/hf_datasets_cache"

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import json
import open_clip
import hnswlib
from transformers import pipeline # <-- NEW: Hugging Face Pipeline
from lavis.models import load_model_and_preprocess

# --- 1. SETUP & DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HF pipelines use 0 for the first GPU, and -1 for CPU
HF_DEVICE = 0 if DEVICE == "cuda" else -1

# --- 2. PIPELINE LOADER ---
@st.cache_resource
def load_ai_pipeline():
    """
    Loads all models, databases, and captions into memory once.
    """
    # ---------------------------------------------------------
    # 1. LOAD FINE-TUNED CLIP (Stage 1 Encoder)
    # ---------------------------------------------------------
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.load_state_dict(torch.load("models/clip_ft_C_seed546_best.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    # ---------------------------------------------------------
    # 2. LOAD HUGGING FACE FASHION DETECTOR (Replaces YOLO)
    # ---------------------------------------------------------
    # Fine-tuned on Fashionpedia - recognizes 40+ distinct garment types
    fashion_detector = pipeline("object-detection", model="valentinafeve/yolos-fashionpedia", device=HF_DEVICE)
    
    # ---------------------------------------------------------
    # 3. LOAD HNSW INDEX & METADATA (Stage 1 Database)
    # ---------------------------------------------------------
    idx = hnswlib.Index(space='cosine', dim=512)
    idx.load_index("index/config_C/hnsw.bin")
    idx.set_ef(150)
    
    meta = json.load(open("index/config_C/config.json"))
    gallery_paths = meta['paths']
    
    # ---------------------------------------------------------
    # 4. LOAD BLIP-2 ITM (Lightweight Re-ranker)
    # ---------------------------------------------------------
    blip_model, vis_processors, text_processors = load_model_and_preprocess(
        name="blip2_image_text_matching",
        model_type="pretrain",
        is_eval=True,
        device=DEVICE
    )
    
    # ---------------------------------------------------------
    # 5. LOAD CAPTIONS
    # ---------------------------------------------------------
    captions_path = os.path.join("safe_captions", "captions", "captions.json")
    with open(captions_path, "r") as f:
        raw_captions = json.load(f)
        
    gallery_captions = []
    for path in gallery_paths:
        clean_path = path.replace("\\", "/") 
        caption = raw_captions.get(clean_path, "A piece of clothing") 
        gallery_captions.append(caption)

    # Return fashion_detector instead of yolo
    return model, preprocess, fashion_detector, idx, gallery_paths, blip_model, vis_processors, text_processors, gallery_captions

# --- 3. MAIN UI ---
st.set_page_config(layout="wide", page_title="AI Fashion Search")
st.title("👗 AI Fashion Search Engine (Two-Stage Retrieval)")
st.write("Upload a photo of an outfit to find visually and semantically similar items.")

with st.spinner("Warming up AI Models... (This takes a minute on startup)"):
    # Unpack the new fashion_detector
    model, preprocess, fashion_detector, idx, gallery_paths, blip_model, vis_processors, text_processors, gallery_captions = load_ai_pipeline()

# File Uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Target")
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, use_container_width=True)
        
    with col2:
        st.subheader("Garment Selection")
        
        # --- 1. RUN THE FASHION DETECTOR ---
        with st.spinner("Scanning image for specific clothing items..."):
            detections = fashion_detector(query_image)
            
        # --- 2. BUILD THE OPTIONS LIST ---
        # These 3 manual options will ALWAYS be in the list
        options = [
            "Full Body (Default)", 
            "Upper Body (Top 60%)", 
            "Lower Body (Bottom 55%)"
        ]
        
        # If the AI found anything, add ALL of them to the list!
        if detections:
            st.success(f"Found {len(detections)} distinct garment(s) via AI!")
            ai_options = [f"AI Detected: {d['label']} ({int(d['score']*100)}% confidence)" for d in detections]
            options.extend(ai_options) # Add AI options to the end of the manual options
        else:
            st.warning("No distinct clothing items detected by AI. Please use a manual selection.")

        st.write("### Which item would you like to search for?")
        
        # --- 3. DISPLAY THE RADIO BUTTONS ---
        selected_option = st.radio("Select Target:", options)
        
        # --- 4. EXECUTE THE CROP BASED ON SELECTION ---
        W, H = query_image.size
        
        if selected_option == "Full Body (Default)":
            crop = query_image
            
        elif selected_option == "Upper Body (Top 60%)":
            crop = query_image.crop((0, 0, W, int(H * 0.60)))
            
        elif selected_option == "Lower Body (Bottom 55%)":
            crop = query_image.crop((0, int(H * 0.45), W, H))
            
        else:
            # If it's none of the manual ones, it MUST be an AI option.
            # We subtract 3 because the first 3 items in the 'options' list are the manual ones.
            ai_index = options.index(selected_option) - 3 
            best_box = detections[ai_index]['box']
            crop = query_image.crop((best_box['xmin'], best_box['ymin'], best_box['xmax'], best_box['ymax']))
        
        # --- 5. SHOW THE RESULT TO THE USER ---
        with col1:
            st.write("**Isolated Search Target:**")
            st.image(crop, width=200)
            
    # ==========================================
    # TWO-STAGE RETRIEVAL PIPELINE
    # ==========================================
    st.subheader("Top Matches")
    
    with st.spinner("Stage 1: Fast HNSW Retrieval..."):
        # 1. Extract CLIP Features
        x = preprocess(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = F.normalize(model.encode_image(x), dim=-1).cpu().numpy().squeeze()
        
        # 2. Grab Top 15
        labels, distances = idx.knn_query(emb, k=15) 
        
    with st.spinner("Stage 2: Semantic Re-ranking... (Fast ITM)"):
        reranked_results = []
        
        for label in labels[0]:
            candidate_path = gallery_paths[label]
            candidate_caption = gallery_captions[label]
            
            # Use BLIP-2 specifically for Image-Text Matching
            image = vis_processors["eval"](crop).unsqueeze(0).to(DEVICE)
            text = text_processors["eval"](candidate_caption)

            sample = {
                "image": image,
                "text_input": [text]
            }

            with torch.no_grad():
                itm_output = blip_model(sample, match_head="itm")
                itm_probs = torch.nn.functional.softmax(itm_output, dim=1)
                itm_match_probability = itm_probs[:, 1].item()
            
            reranked_results.append({
                "path": candidate_path,
                "caption": candidate_caption,
                "score": itm_match_probability
            })
        
        # Sort Top 15 by BLIP-2 Score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_top_10 = reranked_results[:10]
            
    # --- DISPLAY RESULTS (THE FLAT FOLDER WAY) ---
    # Notice this is lined up with 'col1, col2 = st.columns([1, 2])'
    # It sits outside the 'with col2:' block so it spans the whole screen!
    
    st.markdown("---") # Adds a nice visual line break
    st.subheader("Visual Matches (Original Dataset Images)")
    
    result_cols = st.columns(5) # 5 columns for the Top 10
    for i, result in enumerate(final_top_10):
        
        # 1. Convert database path to the double-underscore flat name
        safe_filename = result["path"].replace("/", "__").replace("\\", "__")
        
        # Add 'img__' if your flat folder uses it
        if not safe_filename.startswith("img__"):
            safe_filename = "img__" + safe_filename
        
        # 2. Point directly to your NEW flat folder of full images!
        local_image_path = os.path.join("gallery_images_full", safe_filename)
        
        with result_cols[i % 5]:
            try:
                res_img = Image.open(local_image_path)
                st.image(res_img, use_container_width=True)
                st.caption(f"**Rank {i+1}** (Score: {result['score']:.3f})")
            except FileNotFoundError:
                st.error(f"Missing: {safe_filename}")