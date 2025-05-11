import os
import json
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# ─── CONFIGURATION ────────────────────────────────────────────────────────
SAMPLES_JSON = "./Mid_metric/samples.json"
OUTPUT_DIR   = "./generated_images"
MODEL_NAMES = [
    "Salesforce_blip-image-captioning-base_Salesforce_blip-image-captioning-base",
    "Salesforce_blip2-opt-2.7b_Salesforce_blip2-opt-2.7b",
    "microsoft_git-base_microsoft_git-base",
    "nlpconnect_vit-gpt2-image-captioning_nlpconnect_vit-gpt2-image-captioning",
    "Ertugrul_Qwen2-VL-7B-Captioner-Relaxed_Ertugrul_Qwen2-VL-7B-Captioner-Relaxed",
]
SD_MODEL      = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ─── SETUP ─────────────────────────────────────────────────────────────────
# load your SD pipeline once
pipeline = DiffusionPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
# disable safety checker if needed
pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# make output folders
refs_dir = os.path.join(OUTPUT_DIR, "references")
os.makedirs(refs_dir, exist_ok=True)
model_dirs = {}
for m in MODEL_IDS:
    d = os.path.join(OUTPUT_DIR, m)
    os.makedirs(d, exist_ok=True)
    model_dirs[m] = d

# ─── LOAD MERGED samples.json ─────────────────────────────────────────────
with open(SAMPLES_JSON, "r") as f:
    model_data = json.load(f)       # now a dict: model_name -> list of samples

# get your model names (will match MODEL_IDS)
models = list(model_data.keys())
if not models:
    raise ValueError("no models found in JSON")

# assume every model produced the same number of samples
samples = model_data[models[0]]     # pick the first model’s list to drive the loop
print(f"Loaded {len(samples)} samples (via model '{models[0]}')")

print(f"Generating images on {DEVICE}...")

# ─── 1) generate *reference* images once ───────────────────────────────────
for sample in samples:
    img_id       = os.path.splitext(os.path.basename(sample["image_path"]))[0]
    ref_caption  = sample.get("reference", "").strip()
    if not ref_caption:
        continue

    try:
        img = pipeline(ref_caption).images[0]
        img.save(os.path.join(refs_dir, f"{img_id}.png"))
        print(f"[ref ] {img_id}")
    except Exception as e:
        print(f"✖ ref  {img_id}: {e}")

# ─── 2) for each model, generate its *prediction* images ───────────────────
for i, sample in enumerate(samples):
    img_id = os.path.splitext(os.path.basename(sample["image_path"]))[0]

    for model_name in models:
        out_dir = model_dirs[model_name]
        # each model’s list is parallel to samples
        pred_caption = model_data[model_name][i].get("prediction", "").strip()
        if not pred_caption:
            continue

        try:
            img = pipeline(pred_caption).images[0]
            img.save(os.path.join(out_dir, f"{img_id}.png"))
            print(f"[{model_name[:8]}] {img_id}")
        except Exception as e:
            print(f"✖ [{model_name[:8]}] {img_id}: {e}")

print("✔ Image generation complete.")
