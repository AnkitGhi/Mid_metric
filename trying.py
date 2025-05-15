import os
import json
import torch
import clip
from PIL import Image
from metrics.mid import MutualInformationDivergence

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paths    = [e["image_path"] for e in data]
    captions = [e["caption"]    for e in data]
    return data, paths, captions

def main():
    # 1) Load
    data, image_paths, gen_captions = load_data("./Mid_metric/new_pairs.json")
    assert len(image_paths) == len(gen_captions), "Mismatch lengths"

    # 2) Load images
    real_images = [Image.open(p).convert("RGB") for p in image_paths]

    # 3) CLIP setup
    device, cache = ("cuda", True) if torch.cuda.is_available() else ("cpu", False)
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 4) Preprocess/tokenize
    imgs = torch.stack([preprocess(img) for img in real_images]).to(device)
    txts = clip.tokenize(gen_captions, truncate=True).to(device)

    # 5) Encode
    with torch.no_grad():
        x = model.encode_image(imgs)
        y = model.encode_text(txts)

    # 6) Normalize & to float64
    x = (x / x.norm(dim=1, keepdim=True)).double()
    y = (y / y.norm(dim=1, keepdim=True)).double()

    # 7) Build & update MID
    D   = x.shape[1]
    mid = MutualInformationDivergence(feature=D, limit=30000).to(device)
    mid.update(x, y, x)

    # 8) Compute per-pair scores
    scores = mid.compute(reduction=False).cpu().numpy()

    # 9) Inject into JSON
    out_path = "new_pairs_with_mid.json"
    print(f"[+] Writing scores into {out_path} …")
    for entry, score in zip(data, scores):
        entry["mid_pmi_score"] = float(score)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✔] Done! Check {os.path.abspath(out_path)}")

    # 10) Optional print
    for i, s in enumerate(scores):
        print(f"Pair {i:3d}: MID/PMI = {s:.4f}")

if __name__ == "__main__":
    main()
