import torch
import clip
from PIL import Image
from metrics.mid import MutualInformationDivergence

import json

def load_data(json_path):
    """
    Reads a JSON file of the form:
    [
      {
        "image_path": "...",
        "caption": "...",
        "average_score": ...
      },
      ...
    ]
    Returns the raw list of dicts, plus parallel lists of image_paths & captions.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_paths = [entry["image_path"] for entry in data]
    gen_captions = [entry["caption"]    for entry in data]
    return data, image_paths, gen_captions

# 1) Load your JSON
data, image_paths, gen_captions = load_data("new_pairs.json")
assert len(image_paths) == len(gen_captions)

# 2) Load images
real_images = [Image.open(p).convert("RGB") for p in image_paths]

# 3) Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 4) Preprocess & tokenize
imgs = torch.stack([preprocess(img) for img in real_images]).to(device)
txts = clip.tokenize(gen_captions, truncate=True).to(device)

# 5) Extract embeddings
with torch.no_grad():
    x = model.encode_image(imgs)
    y = model.encode_text(txts)

# 6) Normalize & cast to double
x = (x / x.norm(dim=1, keepdim=True)).double()
y = (y / y.norm(dim=1, keepdim=True)).double()

# 7) Build & update MID metric
D   = x.shape[1]
mid = MutualInformationDivergence(feature=D, limit=30000).to(device)
mid.update(x, y, x)

# 8) Compute per-pair scores
scores = mid.compute(reduction=False).cpu().numpy()

# ————— Add JSON-writing here —————
# 9) Merge scores back into your original data and write a new file:
for entry, score in zip(data, scores):
    entry["mid_pmi_score"] = float(score)

with open("new_pairs_with_mid.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
# ——————————————————————————————

# 10) (Optional) Print them out
for i, s in enumerate(scores):
    print(f"Pair {i}: MID/PMI score = {s:.4f}")
