import os
from PIL import Image
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity

# --- CONFIG ---
BASE_DIR  = "./images_generated"
REFS_DIR  = os.path.join(BASE_DIR, "references_images")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# discover all sub‐folders under BASE_DIR except refs
model_dirs = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
    and d != "references_images"
]

if not os.path.isdir(REFS_DIR):
    raise RuntimeError(f"Reference folder not found: {REFS_DIR}")

if not model_dirs:
    print("⚠️  No model prediction folders found under images_generated/ — nothing to compare.")
    exit()

print(f"🔍 Found model folders: {model_dirs}")

# --- LOAD CLIP ---
clip_model_name = "openai/clip-vit-base-patch32"
model     = CLIPModel.from_pretrained(clip_model_name).to(DEVICE)
processor = CLIPProcessor.from_pretrained(clip_model_name)

@torch.no_grad()
def get_image_emb(path):
    img = Image.open(path).convert("RGB")
    inp = processor(images=img, return_tensors="pt").to(DEVICE)
    emb = model.get_image_features(**inp)
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

# --- COLLECT SIMS ---
records = []
for fn in os.listdir(REFS_DIR):
    if not fn.lower().endswith(".png"):
        continue
    img_id  = os.path.splitext(fn)[0]
    ref_path = os.path.join(REFS_DIR, fn)

    try:
        ref_emb = get_image_emb(ref_path)
    except Exception as e:
        print(f"⚠️  Couldn't embed reference {fn}: {e}")
        continue

    for mdl in model_dirs:
        pred_path = os.path.join(BASE_DIR, mdl, fn)
        if not os.path.isfile(pred_path):
            sim = None
        else:
            try:
                pred_emb = get_image_emb(pred_path)
                sim = cosine_similarity(ref_emb, pred_emb).item()
            except Exception as e:
                print(f"⚠️  Error embedding {mdl}/{fn}: {e}")
                sim = None

        records.append({
            "image_id":   img_id,
            "model_dir":  mdl,
            "similarity": sim
        })

# --- MAKE DATAFRAME ---
df = pd.DataFrame(records)
df_pivot = df.pivot(index="image_id", columns="model_dir", values="similarity")
print(df_pivot)

# optional: save
df_pivot.to_csv("clip_image_similarities.csv")
