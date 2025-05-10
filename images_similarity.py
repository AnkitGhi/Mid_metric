import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_ROOT      = "./Mid_metric"           # where your model folders live
OUTPUT_ROOT    = "./images_generated"     # where your ref + pred PNGs live
REF_DIR        = os.path.join(OUTPUT_ROOT, "references_images")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = [
    "Salesforce_blip-image-captioning-base_Salesforce_blip-image-captioning-base",
    "Salesforce_blip2-opt-2.7b_Salesforce_blip2-opt-2.7b",
    "microsoft_git-base_microsoft_git-base",
    "nlpconnect_vit-gpt2-image-captioning_nlpconnect_vit-gpt2-image-captioning",
    "meta-llama_Llama-3.2-11B-Vision-Instruct_meta-llama_Llama-3.2-11B-Vision-Instruct",
    "Ertugrul_Qwen2-VL-7B-Captioner-Relaxed_Ertugrul_Qwen2-VL-7B-Captioner-Relaxed",
    "Qwen_Qwen2.5-VL-7B-Instruct_Qwen_Qwen2.5-VL-7B-Instruct"
]

# ─── SETUP CLIP ──────────────────────────────────────────────────────────
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def embed_image(path: str):
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    emb = clip_model.get_image_features(**inputs)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

# ─── MAIN LOOP ───────────────────────────────────────────────────────────
for model_name in MODEL_NAMES:
    folder          = os.path.join(DATA_ROOT, model_name)
    samples_path    = os.path.join(folder, "samples.json")
    results_path    = os.path.join(folder, "metric_results.json")
    pred_images_dir = os.path.join(OUTPUT_ROOT, f"{model_name}_predict_images")

    # skip if required files/folders are missing
    if not os.path.isfile(samples_path):
        print(f"⚠ samples.json not found for {model_name}, skipping.")
        continue
    if not os.path.isfile(results_path):
        print(f"⚠ metric_results.json not found for {model_name}, skipping.")
        continue
    if not os.path.isdir(pred_images_dir):
        print(f"⚠ prediction folder not found for {model_name}, skipping.")
        continue

    # load metric_results.json
    try:
        with open(results_path, "r") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠ invalid JSON in {results_path}, skipping: {e}")
        continue

    # for each entry, extract img_id from its image_path
    for entry in metrics:
        img_path = entry.get("image_path", "")
        # basename e.g. "15.jpg"
        fname = os.path.basename(img_path)
        img_id = os.path.splitext(fname)[0]         # "15"

        ref_file  = os.path.join(REF_DIR,  f"{img_id}_reference.png")
        pred_file = os.path.join(pred_images_dir, f"{img_id}_prediction.png")

        sim = None
        if os.path.isfile(ref_file) and os.path.isfile(pred_file):
            try:
                ref_emb  = embed_image(ref_file)
                pred_emb = embed_image(pred_file)
                sim = float(cosine_similarity(ref_emb, pred_emb).item())
            except Exception as e:
                print(f"⚠ embedding error for {model_name} image_id={img_id}: {e}")

        # update or add the field
        entry["ImageSimilarity"] = sim

    # write back updated metric_results.json
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Updated ImageSimilarity in {model_name}/metric_results.json")
