"""
    mid.metric 
    Adapted for samples.json format, multi‐model runner
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
import os
import json
from typing import *
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

import clip
from metrics.mid           import MutualInformationDivergence
from metrics.clips         import ClipScore
from metrics.r_precision   import RPrecision
from metrics.soa           import SemanticObjectAccuracy
from metrics.infonce       import InfoNCE
from metrics.caption_clips import CaptionClipScore

# ─── GLOBAL CONFIG ────────────────────────────────────────────────────────
MODEL_NAMES = [
    "Salesforce_blip-image-captioning-base_Salesforce_blip-image-captioning-base",
    "Salesforce_blip2-opt-2.7b_Salesforce_blip2-opt-2.7b",
    "microsoft_git-base_microsoft_git-base",
    "nlpconnect_vit-gpt2-image-captioning_nlpconnect_vit-gpt2-image-captioning",
    "Ertugrul_Qwen2-VL-7B-Captioner-Relaxed_Ertugrul_Qwen2-VL-7B-Captioner-Relaxed",
]

# The set of metrics to compute per sample
METRICS = [
    MutualInformationDivergence,  # Ours
    ClipScore,                    # CLIP-S
    RPrecision,                   # CLIP-R-Precision
    # SemanticObjectAccuracy,     # (optional) SOA
    InfoNCE,                      # Negative InfoNCE loss
    CaptionClipScore              # OFA-Large + CLIP-S
]

# Maximum number of samples to process per model
LIMIT = 30000


# ─── CUSTOM DATASET ──────────────────────────────────────────────────────
class CustomImageTextDataset(Dataset):
    def __init__(self, json_path: str, transform=None):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        self.transform = transform
        print(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 224, 224))

        reference_caption = sample['reference']
        predicted_caption = sample['prediction']
        dataset_index     = sample.get('dataset_index', idx)
        label = [torch.tensor(1.0), torch.tensor(1.0)]
        return image, reference_caption, str(dataset_index), "", image, label, "custom"


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────
def get_clip(eval_model: str, device: torch.device):
    """Load CLIP model by name."""
    clip_model, _ = clip.load(eval_model, device=device)
    clip_prep = T.Compose([
        T.Resize(224),
        T.Normalize((0.48145466, 0.4578275,  0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
    ])
    return clip_model.cuda(device), clip_prep


def init_metric(root: str, metric: Type[torch.nn.Module], eval_model: str,
                limit: int, device: torch.device) -> torch.nn.Module:
    """Initialize a metric class properly."""
    if metric is SemanticObjectAccuracy:
        m = metric(limit=limit)
    elif metric is CaptionClipScore:
        m = metric(limit=limit, gen_json=os.path.join(root, "ofa_caption"))
    else:
        dim = 768 if eval_model == 'ViT-L/14' else 512
        m = metric(dim, limit=limit)
    m.cuda(device)
    m._debug = False
    return m


@torch.no_grad()
def populate_metrics(dataloader: DataLoader, metrics: List[torch.nn.Module],
                     clip_model: torch.nn.Module, clip_prep) -> torch.Tensor:
    device = next(clip_model.parameters()).device
    all_labels = []
    for real, gt, iid, cid, fake, label, gen_type in tqdm(dataloader):
        real = real.cuda(device)
        fake = fake.cuda(device)
        all_labels.append(torch.stack(label, dim=1))

        txt = clip.tokenize(gt, truncate=True).cuda(device)
        txt_features = clip_model.encode_text(txt).float()
        real_im = clip_prep(real)
        fake_im = clip_prep(fake)
        real_im_features = clip_model.encode_image(real_im).float()
        fake_im_features = clip_model.encode_image(fake_im).float()

        # normalize
        txt_features     = F.normalize(txt_features, dim=-1)
        real_im_features = F.normalize(real_im_features, dim=-1)
        fake_im_features = F.normalize(fake_im_features, dim=-1)

        X_ref = real_im_features
        Y_ref = txt_features
        X     = fake_im_features

        for m in metrics:
            if isinstance(m, SemanticObjectAccuracy):
                m.update(real, gt, is_real=True)
                m.update(fake, gt, is_real=False)
            elif isinstance(m, CaptionClipScore):
                cap = clip.tokenize(gt, truncate=True).cuda(device)
                cf  = clip_model.encode_text(cap).float()
                cf  = F.normalize(cf, dim=-1)
                m.update(X_ref, Y_ref, cf)
            else:
                m.update(X_ref, Y_ref, X)

        if (len(all_labels) * real.shape[0]) >= metrics[0].limit:
            break

    return torch.cat(all_labels, dim=0).to(device)


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle

    EVAL_MODEL = os.getenv('EVAL_MODEL', "ViT-B/32")
    DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_name in MODEL_NAMES:
        root = os.path.join("./", model_name)
        samples_json = os.path.join(root, "samples.json")
        results_json = os.path.join(root, "metric_results.json")
        cache_dir    = os.path.join(root, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file   = os.path.join(cache_dir, f"cache_{len(METRICS)}.pth")
        pred_dir     = os.path.join("./images_generated", f"{model_name}_predict_images")

        # skip missing
        if not os.path.isdir(root):
            print(f"⚠ Model folder missing: {root}")
            continue
        if not os.path.isfile(samples_json):
            print(f"⚠ samples.json missing for {model_name}")
            continue

        print(f"\n▶ Processing '{model_name}'")

        # load or compute
        force = True
        if os.path.exists(cache_file) and not force:
            label, results = torch.load(cache_file)
            print(f"  • Loaded cache from {cache_file}")
        else:
            print(f"  • Loading CLIP ({EVAL_MODEL}) → {DEVICE}")
            clip_model, clip_prep = get_clip(EVAL_MODEL, DEVICE)

            print("  • Initializing metrics")
            metrics = [
                init_metric(root, m, EVAL_MODEL, LIMIT, DEVICE)
                for m in METRICS
            ]

            print("  • Preparing DataLoader")
            transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor()
            ])
            ds = CustomImageTextDataset(samples_json, transform=transform)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

            print("  • Populating metrics…")
            label   = populate_metrics(dl, metrics, clip_model, clip_prep)
            results = [m.compute(reduction=False) for m in metrics]

            print(f"  • Caching to {cache_file}")
            torch.save((label, results), cache_file)

        n_samples = min(len(results[0]), LIMIT, label.shape[0])
        print(f"  • Using {n_samples} samples")

        with open(samples_json, 'r') as f:
            samples = json.load(f)

        # build metric_results.json
        output = []
        for i in range(min(n_samples, len(samples))):
            s = samples[i]
            entry = {
                "dataset_index": s.get("dataset_index", i),
                "image_path":    s["image_path"],
                "reference":     s["reference"],
                "prediction":    s["prediction"]
            }
            # append each metric
            for j, m_cls in enumerate(METRICS):
                name = m_cls.__name__
                entry[name] = float(results[j][i].cpu().numpy())
            output.append(entry)

        # write metric_results.json
        with open(results_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {results_json}")
