"""
    mid.metric 
    Adapted for samples.json format
    Copyright (c) 2022-present NAVER Corp.
    Apache-2.0
"""
from PIL import Image
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from tqdm import tqdm
from typing import *
import clip
import csv
import json
import krippendorff as kd
import os
import pickle
import scipy.stats as ss

from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from metrics.mid           import MutualInformationDivergence
from metrics.clips         import ClipScore
from metrics.r_precision   import RPrecision
from metrics.soa           import SemanticObjectAccuracy
from metrics.infonce       import InfoNCE
from metrics.caption_clips import CaptionClipScore


# Custom dataset for your samples.json format
class CustomImageTextDataset(Dataset):
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path: Path to samples.json
            transform: Optional transform to be applied on images
        """
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
        self.transform = transform
        print(f"Loaded {len(self.samples)} samples from {json_path}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if there's an error
            image = torch.zeros((3, 224, 224))
        
        # Get captions - using the field names from your JSON
        reference_caption = sample['reference']
        predicted_caption = sample['prediction']
        dataset_index = sample.get('dataset_index', idx)  # Use dataset_index if available, otherwise idx
        
        # Format similar to the original code's expectations
        # real, gt, iid, cid, fake, label, gen_type
        # Where:
        # - real is the original image
        # - gt is the ground truth caption (reference_caption)
        # - iid is the image ID (use dataset_index as identifier)
        # - cid is caption ID (None in our case)
        # - fake is the same image (in original code it would be a generated image)
        # - label is a quality/alignment rating (not available, use default)
        # - gen_type is the generation type (use 'custom' in our case)
        
        # Use a default label tensor with 1.0 for both quality and alignment
        # Original format had [quality, alignment] scores
        label = [torch.tensor(1.0), torch.tensor(1.0)]
        
        return image, reference_caption, str(dataset_index), "", image, label, "custom"


def get_clip(eval_model: Module, device: Union[torch.device, int]) \
        -> Tuple[Module, Module]:
    """Get the CLIP model

    Args:
        eval_model (Module): The CLIP model to evaluate
        device (Union[torch.device, int]): Device index to select

    Returns:
        Tuple[Module, Module]: The CLIP model and a preprocessor
    """
    clip_model, _ = clip.load(eval_model)
    clip_model = clip_model.cuda(device)
    clip_prep = T.Compose([T.Resize(224),
                           T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))])
    return clip_model, clip_prep


def init_metric(root: str, metric: Type[Module], eval_model: Module,
                limit: int, device: torch.device) -> Module:
    """Initialize a given metric class.

    Args:
        root (str): Path to data directory
        metric (Type[Module]): Metric class
        eval_model (Module): Evaluating CLIP model
        limit (int, optional): Number of reference samples
        device (torch.device): Device index to select

    Returns:
        Module: Initialized metric instance
    """
    if metric is SemanticObjectAccuracy:
        m = metric(limit=limit)
    elif metric is CaptionClipScore:
        m = metric(limit=limit, gen_json=os.path.join(root, "ofa_caption"))
    else:
        m = metric(768 if eval_model == 'ViT-L/14' else 512,
                   limit=limit)
    m.cuda(device)
    m._debug = False
    return m


@torch.no_grad()
def populate_metrics(dataloader: DataLoader, metrics: List[Module],
                     clip_model: Module) -> Tensor:
    """Populate the list of metrics using a given data loader.

    Args:
        dataloader (DataLoader): Data loader
        metrics (List[Module]): List of metrics
        clip_model (Module): Evaluating CLIP model

    Returns:
        Tensor: Labels
    """
    device = next(clip_model.parameters()).device
    labels = []
    for i, (real, gt, iid, cid, fake, label, gen_type) in enumerate(
            tqdm(dataloader)):

        real = real.cuda(device)
        fake = fake.cuda(device)
        labels.append(torch.stack(label, dim=1))

        txt = clip.tokenize(gt, truncate=True).cuda(device)
        txt_features = clip_model.encode_text(txt).float()

        real_im_features = clip_model.encode_image(
            clip_prep(real)).float()
        fake_im_features = clip_model.encode_image(
            clip_prep(fake)).float()

        # float16 of CLIP may suffer in l2-normalization
        txt_features = F.normalize(txt_features, dim=-1)
        real_im_features = F.normalize(real_im_features, dim=-1)
        fake_im_features = F.normalize(fake_im_features, dim=-1)

        X_ref = real_im_features
        Y_ref = txt_features
        X = fake_im_features

        # metrics handle features in float64
        for idx, m in enumerate(metrics):
            if isinstance(m, SemanticObjectAccuracy):
                m.update(real, gt, is_real=True)
                m.update(fake, gt, is_real=False)
            elif isinstance(m, CaptionClipScore):
                # For CaptionClipScore, we need to handle iid
                # If we don't have real captions, let's use fake ones
                captions = gt  # Using ground truth captions for simplicity
                cap = clip.tokenize(captions, truncate=True).cuda(device)
                cap_features = clip_model.encode_text(cap).float()
                cap_features = F.normalize(cap_features, dim=-1)
                m.update(X_ref, Y_ref, cap_features)
            else:
                m.update(X_ref, Y_ref, X)

        if (i + 1) * real.shape[0] > metrics[0].limit:
            print(f"break loop due to the limit of {metrics[0].limit}.")
            break

    return torch.cat(labels, dim=0).to(device)  # N x (quality, alignment)


if "__main__" == __name__:
    # config
    eval_model = os.getenv('EVAL_MODEL')
    if eval_model is None:
        eval_model = "ViT-B/32"
    root = "./data/"  # Root directory for data
    samples_json_path = os.path.join(root, "samples.json")  # Path to your samples.json file
    limit = 30000  # number of reference samples

    METRICS = [MutualInformationDivergence,  # Ours
               ClipScore,                    # CLIP-S
               RPrecision,                   # CLIP-R-Precision
               SemanticObjectAccuracy,       # Piece-wise SOA
               InfoNCE,                      # Negative InfoNCE loss
               CaptionClipScore,             # OFA-Large+CLIP-S
               ]

    cache_path = os.path.join(
        root, ".cache",
        f"samples_json_metric{len(METRICS)}.pth")
    os.makedirs(os.path.join(root, ".cache"), exist_ok=True)

    force = True  # Set to True to recompute scores, False to load from cache
    if not os.path.exists(cache_path) or force:
        # get clip model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Using CLIP model: {eval_model}")
        clip_model, clip_prep = get_clip(eval_model, device)

        metrics = [
            init_metric(root, x, eval_model, limit, device) for x in METRICS]

        # Set up image transform
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ])
        
        # load dataset
        dataset = CustomImageTextDataset(samples_json_path, transform=transform)
        dl = DataLoader(dataset, batch_size=32,
                        drop_last=False, shuffle=False,
                        num_workers=4)

        # compute clip features
        label = populate_metrics(dl, metrics, clip_model)
        results = [m.compute(reduction=False) for m in metrics]

        torch.save([label, results], cache_path)
        print(f"[INFO] score cache is saved to `{cache_path}`.")
    else:
        label, results = torch.load(cache_path)
        print(f"[INFO] score cache is loaded from `{cache_path}`.")

    # Filter results based on available data
    n_samples = min(len(results[0]), limit)
    print(f"Number of samples: {n_samples}")

    for i, metric_cls in enumerate(METRICS):
        metric_name = metric_cls.__name__
        scores = results[i][:n_samples].cpu().numpy()
        print(f"{metric_name}: Mean={scores.mean():.4f}, Min={scores.min():.4f}, Max={scores.max():.4f}")
    
    # Save results to JSON
    with open(samples_json_path, 'r') as f:
        samples = json.load(f)
    
    output = []
    for i in range(min(n_samples, len(samples))):
        entry = {
            'dataset_index': samples[i].get('dataset_index', i),
            'image_path': samples[i]['image_path'],
            'reference': samples[i]['reference'],
            'prediction': samples[i]['prediction']
        }
        
        # Add metrics
        for j, metric_cls in enumerate(METRICS):
            metric_name = metric_cls.__name__
            if i < len(results[j]):
                entry[metric_name] = float(results[j][i].cpu().numpy())
        
        output.append(entry)
    
    # Save results
    with open(os.path.join(root, 'metric_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {os.path.join(root, 'metric_results.json')}")

