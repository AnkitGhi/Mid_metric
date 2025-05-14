# run_pmi.py

import json
import torch
from torch.utils.data import DataLoader

from datasets import ReferenceImageCaptionDataset, PairImageCaptionDataset
from mid_extension import MIDWithBatchPMI

def main():
    # 1) DataLoaders
    ref_loader = DataLoader(
        ReferenceImageCaptionDataset("./Mid_metric/reference_data.json"),
        batch_size=32, shuffle=False  # shuffle=False for reproducibility
    )
    pair_loader = DataLoader(
        PairImageCaptionDataset("./Mid_metric/new_pairs.json"),
        batch_size=32, shuffle=False
    )

    # 2) Initialize MID (with a small epsilon for stability)
    mid = MIDWithBatchPMI(feature=512, limit=1000000, eps=5e-4)

    # 3) Fit reference Gaussians
    for x_ref, y_ref, x0_ref in ref_loader:
        mid.update(x_ref, y_ref, x0_ref)

    # 4a) Overall MID on reference set
    overall = mid.compute()
    print(f"Overall MID score (reference set): {overall.item():.4f}")

    # 4b) Per-sample PMI on new pairs
    all_pmis = []
    for x_new, y_new in pair_loader:
        pmis = mid.batch_pmi(x_new, y_new)  # Tensor of shape [batch_size]
        all_pmis.append(pmis)
    all_pmis = torch.cat(all_pmis, dim=0)  # shape [N_new]

    # Convert to Python list
    pmi_list = all_pmis.tolist()

    # Print first two as a sanity check
    print("First PMI:", pmi_list[0])
    print("Second PMI:", pmi_list[1])

    # 5) Load original new-pairs entries
    with open("./Mid_metric/new_pairs.json", "r") as f:
        entries = json.load(f)

    # 6) Build results and print
    results = []
    for entry, pmi in zip(entries, pmi_list):
        print(f"Image:   {entry['image_path']}")
        print(f"Caption: {entry['caption']}")
        print(f"PMI = {pmi:.4f}\n")
        results.append({
            "image_path": entry["image_path"],
            "caption":    entry["caption"],
            "PMI":        pmi
        })

    # 7) Save to a new JSON file
    with open("pmi_results.json", "w") as out:
        json.dump(results, out, indent=2)
    print(f"Wrote {len(results)} entries to pmi_results.json")

if __name__ == "__main__":
    main()
