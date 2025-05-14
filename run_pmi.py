# run_pmi.py
from torch.utils.data import DataLoader
import torch

from datasets import ReferenceImageCaptionDataset, PairImageCaptionDataset
from mid_extension import MIDWithBatchPMI

def main():
    # 1) DataLoaders
    ref_loader = DataLoader(
        ReferenceImageCaptionDataset("./Mid_metric/reference_data.json"),
        batch_size=32, shuffle=True
    )
    pair_loader = DataLoader(
        PairImageCaptionDataset("./Mid_metric/new_pairs.json"),
        batch_size=32, shuffle=False
    )

    # 2) Initialize MID
    mid = MIDWithBatchPMI(feature=512, limit=30000)

    # 3) Fit reference Gaussians
    for x_ref, y_ref, x0_ref in ref_loader:
        mid.update(x_ref, y_ref, x0_ref)

    # 4a) Overall MID
    overall = mid.compute()
    print(f"Overall MID score: {overall.item():.4f}")

    # 4b) Per-sample PMI on new pairs
    all_pmis = []
    for x_new, y_new in pair_loader:
        pmis = mid.batch_pmi(x_new, y_new)
        all_pmis.append(pmis)

    all_pmis = torch.cat(all_pmis, dim=0)
    print("Per-sample PMI:", all_pmis)

if __name__ == "__main__":
    main()
