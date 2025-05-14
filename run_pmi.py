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

    pmi_list = all_pmis.tolist()  

    # 2) Access by index
    first  = all_pmis[0].item()
    second = all_pmis[1].item()
    print("First PMI:", first)
    print("Second PMI:", second)

    # 3) Iterate with your original entries
    import json
    with open("new_pairs.json", "r") as f:
        entries = json.load(f)

    for idx, (entry, pmi) in enumerate(zip(entries, pmi_list)):
        print(f"Pair #{idx}:")
        print("  Image:  ", entry["image_path"])
        print("  Caption:", entry["caption"])
        print(f"  PMI = {pmi:.4f}\n")
if __name__ == "__main__":
    main()
