# datasets.py
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReferenceImageCaptionDataset(Dataset):
    """
    JSON entries must have:
      - "real_image_path"
      - "caption"
      - "generated_image_path"
    """
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.entries = json.load(f)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        real_img = Image.open(e["real_image_path"]).convert("RGB")
        gen_img  = Image.open(e["generated_image_path"]).convert("RGB")
        caption  = e["caption"]

        img_inputs = self.processor(images=[real_img, gen_img], return_tensors="pt").to(DEVICE)
        txt_inputs = self.processor(text=[caption],   return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            img_feats = self.model.get_image_features(**img_inputs)  # [2, D]
            txt_feats = self.model.get_text_features(**txt_inputs)   # [1, D]

        # return (real_feat, text_feat, generated_feat)
        return img_feats[0], txt_feats[0], img_feats[1]


class PairImageCaptionDataset(Dataset):
    """
    JSON entries must have:
      - "image_path"
      - "caption"
    """
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.entries = json.load(f)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img     = Image.open(e["image_path"]).convert("RGB")
        caption = e["caption"]

        img_inputs = self.processor(images=[img], return_tensors="pt").to(DEVICE)
        txt_inputs = self.processor(text=[caption], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            img_feat = self.model.get_image_features(**img_inputs)[0]  # [D]
            txt_feat = self.model.get_text_features(**txt_inputs)[0]   # [D]

        return img_feat, txt_feat
