import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from metrics.mid import MutualInformationDivergence
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import slogdet, inv

from tqdm import tqdm

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paths    = [e["image_path"] for e in data]
    captions = [e["caption"]    for e in data]
    return data, paths, captions

def process_image_text_file(file_path):
    image_paths = []
    captions = []
    base_path = "./drive/MyDrive/Flickr8k/"

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip() # Remove leading/trailing whitespace, including newline
                if not line: # Skip empty lines
                    continue

                parts = line.split('\t', 1) # Split only on the first tab
                if len(parts) == 2:
                    image_info = parts[0]
                    caption = parts[1]

                    # Get the image filename (part before '#')
                    image_filename = image_info.split('#')[0]

                    # Construct the full image path
                    full_image_path = f"{base_path}{image_filename}"
                    image_paths.append(full_image_path)

                    # Add the caption
                    captions.append(caption)
                else:
                    print(f"Skipping malformed line: {line}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

    return image_paths, captions

class ClipFeatureDataset(Dataset):
    def __init__(self, image_paths: list[str], captions: list[str], device="cuda"):
        self.image_paths = image_paths
        self.captions    = captions
        self.device      = device
        self.model       = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.captions[idx]

        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)

        pixel_vals = inputs["pixel_values"].to(self.device)
        input_ids  = inputs["input_ids"].to(self.device)
        attention  = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            img_feat  = self.model.get_image_features(pixel_vals)
            txt_feat  = self.model.get_text_features(input_ids, attention_mask=attention)

        return img_feat.double().squeeze(0), txt_feat.double().squeeze(0)

class GaussianEstimator:
    def __init__(self, D, reg=1e-6, device="cpu"):
        self.D = D
        self.reg = reg
        self.device = device
        # counters
        self.count = 0
        # 1st moments
        self.sum_x = torch.zeros(D, dtype=torch.float64, device=device)
        self.sum_y = torch.zeros(D, dtype=torch.float64, device=device)
        self.sum_z = torch.zeros(2*D, dtype=torch.float64, device=device)
        # 2nd moments
        self.sum_xx = torch.zeros(D, D, dtype=torch.float64, device=device)
        self.sum_yy = torch.zeros(D, D, dtype=torch.float64, device=device)
        self.sum_zz = torch.zeros(2*D, 2*D, dtype=torch.float64, device=device)

    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        x_batch, y_batch: both (B, D), dtype float64
        """
        B = x_batch.shape[0]
        z_batch = torch.cat([x_batch, y_batch], dim=1)  # (B, 2D)

        self.count += B
        self.sum_x += x_batch.sum(dim=0)
        self.sum_y += y_batch.sum(dim=0)
        self.sum_z += z_batch.sum(dim=0)

        self.sum_xx += x_batch.t() @ x_batch
        self.sum_yy += y_batch.t() @ y_batch
        self.sum_zz += z_batch.t() @ z_batch

    def finalize(self):
        """Call after all updates. Returns the dict of mus, invΣs, and log-dets."""
        N = self.count
        μ_x = self.sum_x / N
        μ_y = self.sum_y / N
        μ_z = self.sum_z / N

        # unbiased cov = (Σ x xᵀ - N μ μᵀ)/(N−1)
        Σ_x = (self.sum_xx - N * torch.outer(μ_x, μ_x)) / (N - 1)
        Σ_y = (self.sum_yy - N * torch.outer(μ_y, μ_y)) / (N - 1)
        Σ_z = (self.sum_zz - N * torch.outer(μ_z, μ_z)) / (N - 1)

        # regularize
        Σ_x += self.reg * torch.eye(self.D, device=self.device)
        Σ_y += self.reg * torch.eye(self.D, device=self.device)
        Σ_z += self.reg * torch.eye(2*self.D, device=self.device)

        # invert & slogdet
        invΣ_x = torch.linalg.inv(Σ_x)
        invΣ_y = torch.linalg.inv(Σ_y)
        invΣ_z = torch.linalg.inv(Σ_z)
        _, logdet_x = torch.linalg.slogdet(Σ_x)
        _, logdet_y = torch.linalg.slogdet(Σ_y)
        _, logdet_z = torch.linalg.slogdet(Σ_z)

        return {
            "mu_x": μ_x,    "mu_y": μ_y,    "mu_z": μ_z,
            "invΣ_x": invΣ_x, "invΣ_y": invΣ_y, "invΣ_z": invΣ_z,
            "logdet_x": logdet_x, "logdet_y": logdet_y, "logdet_z": logdet_z
        }

if __name__ == "__main__":
    
    actual_file_path = "./Mid_metric/annotations.txt"
    image_list, caption_list = process_image_text_file(actual_file_path)

    json_path = "./Mid_metric/new_pairs.json"
    test_image, test_captions = load_data(json_path)
    
    dataset = ClipFeatureDataset(image_list, caption_list, device="cuda")
    real_dataloader = DataLoader(
    dataset,
    batch_size=64,    
    shuffle=False,
    num_workers=2
)
    
    est = GaussianEstimator(D=512, device="cuda")
    for x_batch, y_batch in tqdm(real_dataloader, desc="training"):
                
        x_batch = x_batch.double().to("cuda")
        y_batch = y_batch.double().to("cuda")
        est.update(x_batch, y_batch)

    dists = est.finalize()

