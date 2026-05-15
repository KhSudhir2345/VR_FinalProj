import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
import open_clip
import hnswlib
import math
from transformers import pipeline
from lavis.models import load_model_and_preprocess

# --- CACHE SETUP (Same as app.py) ---
os.environ["HF_HOME"] = "D:/VR_Final_Proj/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/VR_Final_Proj/hf_cache"
os.environ["TORCH_HOME"] = "D:/VR_Final_Proj/torch_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/VR_Final_Proj/hf_datasets_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_DEVICE = 0 if DEVICE == "cuda" else -1

# --- EVALUATION METRICS ---
def calculate_recall_at_k(retrieved_paths, gt_path, k):
    top_k = retrieved_paths[:k]
    return 1.0 if gt_path in top_k else 0.0

def calculate_map_at_k(retrieved_paths, gt_path, k):
    top_k = retrieved_paths[:k]
    if gt_path in top_k:
        rank = top_k.index(gt_path) + 1
        return 1.0 / rank
    return 0.0

def calculate_ndcg_at_k(retrieved_paths, gt_path, k):
    top_k = retrieved_paths[:k]
    if gt_path in top_k:
        rank = top_k.index(gt_path) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0

# --- PIPELINE LOADER ---
def load_ai_pipeline():
    print("\n[SYSTEM] Loading AI Models into Memory... Please wait.")
    
    # 1. CLIP
    print(" -> Loading Fine-Tuned CLIP (Stage 1)...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.load_state_dict(torch.load("models/clip_ft_C_seed546_best.pt", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    # 2. Fashion Detector
    print(" -> Loading Hugging Face Fashion Detector...")
    fashion_detector = pipeline("object-detection", model="valentinafeve/yolos-fashionpedia", device=HF_DEVICE)
    
    # 3. HNSW
    print(" -> Loading HNSW Index & Metadata...")
    idx = hnswlib.Index(space='cosine', dim=512)
    idx.load_index("index/config_C/hnsw.bin")
    idx.set_ef(150)
    meta = json.load(open("index/config_C/config.json"))
    gallery_paths = meta['paths']
    
    # 4. BLIP-2
    print(" -> Loading BLIP-2 ITM Re-ranker (Stage 2)...")
    blip_model, vis_processors, text_processors = load_model_and_preprocess(
        name="blip2_image_text_matching", model_type="pretrain", is_eval=True, device=DEVICE
    )
    
    # 5. Captions
    captions_path = os.path.join("safe_captions", "captions", "captions.json")
    with open(captions_path, "r") as f:
        raw_captions = json.load(f)
    gallery_captions = [raw_captions.get(p.replace("\\", "/"), "A piece of clothing") for p in gallery_paths]

    print("[SYSTEM] All models loaded successfully!\n")
    return model, preprocess, fashion_detector, idx, gallery_paths, blip_model, vis_processors, text_processors, gallery_captions

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("="*60)
    print("👗 VIVA COMMAND LINE EVALUATION TOOL")
    print("="*60)
    
    model, preprocess, fashion_detector, idx, gallery_paths, blip_model, vis_processors, text_processors, gallery_captions = load_ai_pipeline()
    
    # Start an infinite loop so the TA can test as many images as they want
    while True:
        print("\n" + "="*60)
        query_input = input("Enter path to TA's test image (or type 'exit' to quit): ").strip()
        
        if query_input.lower() == 'exit':
            break
            
        if not os.path.exists(query_input):
            print(f"[ERROR] Could not find image at {query_input}")
            continue
            
        gt_input = input("Enter the Ground Truth (GT) target path (e.g., img/MEN/Denim/id_123/01.jpg): ").strip()
        
        # --- PROCESS QUERY ---
        print("\n[1/4] Processing Image...")
        query_image = Image.open(query_input).convert("RGB")
        
        # --- CROP LOGIC ---
        print("[2/4] Running Fashion Detector...")
        detections = fashion_detector(query_image)
        valid_detections = [d for d in detections if d['score'] > 0.60]
        
        if valid_detections:
            # For the CLI, we automatically pick the highest confidence garment to speed things up
            best_box = valid_detections[0]['box']
            crop = query_image.crop((best_box['xmin'], best_box['ymin'], best_box['xmax'], best_box['ymax']))
            print(f" -> Auto-cropped highest confidence item: {valid_detections[0]['label']} ({int(valid_detections[0]['score']*100)}%)")
        else:
            # Fallback to full image if nothing is found
            crop = query_image
            print(" -> No garments found >60% confidence. Using Full Image.")
            
        # --- STAGE 1: HNSW ---
        print("[3/4] Running Stage 1 HNSW Retrieval...")
        x = preprocess(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = F.normalize(model.encode_image(x), dim=-1).cpu().numpy().squeeze()
        labels, distances = idx.knn_query(emb, k=15)
        
        # --- STAGE 2: RE-RANKING ---
        print("[4/4] Running Stage 2 BLIP-2 Semantic Re-ranking...")
        reranked_results = []
        for label in labels[0]:
            candidate_path = gallery_paths[label]
            candidate_caption = gallery_captions[label]
            
            image = vis_processors["eval"](crop).unsqueeze(0).to(DEVICE)
            text = text_processors["eval"](candidate_caption)
            sample = {"image": image, "text_input": [text]}

            with torch.no_grad():
                itm_output = blip_model(sample, match_head="itm")
                itm_probs = torch.nn.functional.softmax(itm_output, dim=1)
                itm_match_probability = itm_probs[:, 1].item()
            
            reranked_results.append({
                "path": candidate_path,
                "score": itm_match_probability
            })
            
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        final_top_10 = reranked_results[:10]
        
        # --- METRICS OUTPUT ---
        retrieved_paths = [res["path"] for res in final_top_10]
        
        recall = calculate_recall_at_k(retrieved_paths, gt_input, 10)
        map_score = calculate_map_at_k(retrieved_paths, gt_input, 10)
        ndcg = calculate_ndcg_at_k(retrieved_paths, gt_input, 10)
        
        print("\n" + "-"*30)
        print(" 🏆 FINAL METRICS (K=10)")
        print("-"*30)
        print(f" Target GT : {gt_input}")
        print(f" Recall@10 : {recall:.2f}")
        print(f" mAP@10    : {map_score:.4f}")
        print(f" NDCG@10   : {ndcg:.4f}")
        print("-"*30)
        
        if recall > 0:
            rank = retrieved_paths.index(gt_input) + 1
            print(f" ✅ SUCCESS! Target found at Rank {rank}.")
        else:
            print(" ❌ Target NOT found in Top 10.")