import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from metrics.mid import MutualInformationDivergence
from transformers import CLIPProcessor, CLIPModel

from numpy.linalg import slogdet, inv

def fit_gaussians(image_feats: np.ndarray,
                  text_feats: np.ndarray,
                  reg: float = 1e-6):
    """
    Given:
      image_feats: shape (N, d_i)
      text_feats : shape (N, d_t)
    Returns:
      mu_img, inv_cov_img, logdet_img,
      mu_txt, inv_cov_txt, logdet_txt,
      mu_joint, inv_cov_joint, logdet_joint
    """
    # 1) Marginal image Gaussian
    mu_img = image_feats.mean(axis=0)  # (d_i,)
    cov_img = np.cov(image_feats, rowvar=False)  # (d_i, d_i)
    cov_img += reg * np.eye(cov_img.shape[0])    # regularize
    inv_cov_img = inv(cov_img)
    sign_i, logdet_img = slogdet(cov_img)

    # 2) Marginal text Gaussian
    mu_txt = text_feats.mean(axis=0)   # (d_t,)
    cov_txt = np.cov(text_feats, rowvar=False)  # (d_t, d_t)
    cov_txt += reg * np.eye(cov_txt.shape[0])
    inv_cov_txt = inv(cov_txt)
    sign_t, logdet_txt = slogdet(cov_txt)

    # 3) Joint Gaussian
    joint = np.concatenate([image_feats, text_feats], axis=1)  # (N, d_i+d_t)
    mu_joint = joint.mean(axis=0)  
    cov_joint = np.cov(joint, rowvar=False)  # (d_i+d_t, d_i+d_t)
    cov_joint += reg * np.eye(cov_joint.shape[0])
    inv_cov_joint = inv(cov_joint)
    sign_j, logdet_joint = slogdet(cov_joint)

    return (mu_img, inv_cov_img, logdet_img,
            mu_txt, inv_cov_txt, logdet_txt,
            mu_joint, inv_cov_joint, logdet_joint)

def pmi_scores_batch(
    img_feats: np.ndarray,      # (N, d_i)
    txt_feats: np.ndarray,      # (N, d_t)
    mu_img, inv_cov_img, logdet_img,
    mu_txt, inv_cov_txt, logdet_txt,
    mu_joint, inv_cov_joint, logdet_joint
) -> np.ndarray:
    N = img_feats.shape[0]
    d_i = img_feats.shape[1]
    d_t = txt_feats.shape[1]
    const_i = d_i * np.log(2 * np.pi)
    const_t = d_t * np.log(2 * np.pi)
    const_j = (d_i + d_t) * np.log(2 * np.pi)

    # compute img log-probs
    di = img_feats - mu_img
    lp_img = -0.5 * np.einsum('ni,ij,nj->n', di, inv_cov_img, di) - 0.5 * (const_i + logdet_img)

    # compute txt log-probs
    dt = txt_feats - mu_txt
    lp_txt = -0.5 * np.einsum('ni,ij,nj->n', dt, inv_cov_txt, dt) - 0.5 * (const_t + logdet_txt)

    # compute joint log-probs
    joint_feats = np.concatenate([img_feats, txt_feats], axis=1)  # (N, d_i+d_t)
    dj = joint_feats - mu_joint
    lp_joint = -0.5 * np.einsum('ni,ij,nj->n', dj, inv_cov_joint, dj) - 0.5 * (const_j + logdet_joint)

    return lp_joint - (lp_img + lp_txt)

def pmi_scores_in_batches(
    img_feats: np.ndarray,      # shape (N, d_i)
    txt_feats: np.ndarray,      # shape (N, d_t)
    batch_size: int,
    mu_img, inv_cov_img, logdet_img,
    mu_txt, inv_cov_txt, logdet_txt,
    mu_joint, inv_cov_joint, logdet_joint
) -> np.ndarray:
    """
    Compute PMI scores in batches to avoid memory spikes.
    Returns an array of shape (N,) of PMI values.
    """
    N = img_feats.shape[0]
    scores = []
    for start in range(0, N, batch_size):
        end = start + batch_size
        bi = img_feats[start:end]
        bt = txt_feats[start:end]
        # vectorized pmi for this batch
        batch_pmi = pmi_scores_batch(
            bi, bt,
            mu_img, inv_cov_img, logdet_img,
            mu_txt, inv_cov_txt, logdet_txt,
            mu_joint, inv_cov_joint, logdet_joint
        )
        scores.append(batch_pmi)
    return np.concatenate(scores, axis=0)


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


def embed_dataset_batch(
    image_paths: list[str],
    captions:    list[str],
    batch_size:  int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed images and captions in batches using CLIP.
    Returns:
      image_feats: array shape (N, d_i)
      text_feats:  array shape (N, d_t)
    """
    assert len(image_paths) == len(captions), "Length mismatch!"
    all_img_feats = []
    all_txt_feats = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_caps  = captions   [i : i + batch_size]
        
        # Load & preprocess
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(
            text=batch_caps,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            img_feats = model.get_image_features(pixel_values=inputs.pixel_values)       # (B, d_i)
            txt_feats = model.get_text_features(input_ids=inputs.input_ids,
                                                attention_mask=inputs.attention_mask)    # (B, d_t)
        
        # Collect
        all_img_feats.append(img_feats.cpu().numpy())
        all_txt_feats.append(txt_feats.cpu().numpy())
    
    # Concatenate batches
    image_feats = np.concatenate(all_img_feats, axis=0)
    text_feats  = np.concatenate(all_txt_feats,  axis=0)
    return image_feats, text_feats

# --- Example usage ---

if __name__ == "__main__":
    # Your lists of image paths and captions:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    actual_file_path = "./Mid_metric/annotations.txt"
    image_list, caption_list = process_image_text_file(actual_file_path)

    img_feats_ref, txt_feats_ref = embed_dataset_batch(image_list, caption_list, batch_size=32)
    print("Image feats shape:", img_feats_ref.shape)  # e.g. (N_ref, 512)
    print("Text  feats shape:", txt_feats_ref.shape)  # e.g. (N_ref, 512)

    json_path = "./Mid_metric/new_pairs.json"
    test_image, test_captions = load_data(json_path)
    test_image_feat, test_cap_feat = embed_dataset_batch(test_image, test_captions, batch_size=32)
    
    (mu_img, inv_cov_img, logdet_img,
     mu_txt, inv_cov_txt, logdet_txt,
     mu_joint, inv_cov_joint, logdet_joint) = fit_gaussians(
        img_feats_ref, txt_feats_ref
    )
     
    pmi_all = pmi_scores_in_batches(
    test_image_feat,
    test_cap_feat,
    batch_size=128,
    mu_img=mu_img, inv_cov_img=inv_cov_img, logdet_img=logdet_img,
    mu_txt=mu_txt, inv_cov_txt=inv_cov_txt, logdet_txt=logdet_txt,
    mu_joint=mu_joint, inv_cov_joint=inv_cov_joint, logdet_joint=logdet_joint
)

    with open("pmi_all.txt", "w") as f:
        for score in pmi_all:
            f.write(f"{score:.6f}\n")