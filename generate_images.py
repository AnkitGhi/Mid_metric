import json
import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import logging as transformers_logging

# Silence diffusers/transformers warnings
transformers_logging.set_verbosity_error()

# Configuration
MODEL_NAMES        = [
    "Salesforce_blip-image-captioning-base_Salesforce_blip-image-captioning-base",
    "Salesforce_blip2-opt-2.7b_Salesforce_blip2-opt-2.7b",
    "microsoft_git-base_microsoft_git-base",
    "nlpconnect_vit-gpt2-image-captioning_nlpconnect_vit-gpt2-image-captioning",
    "meta-llama_Llama-3.2-11B-Vision-Instruct_meta-llama_Llama-3.2-11B-Vision-Instruct",
    "Ertugrul_Qwen2-VL-7B-Captioner-Relaxed_Ertugrul_Qwen2-VL-7B-Captioner-Relaxed",
    "Qwen_Qwen2.5-VL-7B-Instruct_Qwen_Qwen2.5-VL-7B-Instruct"
]
DATA_ROOT          = "./Mid_metric" #Change this when not using google collab
SAMPLES_FILENAME   = "samples.json"
BASE_OUTPUT_FOLDER = "./images_generated"
SD_MODEL_ID        = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Create base output directory
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)
print(f"▶ Images will be written under {BASE_OUTPUT_FOLDER}/")

def load_pipeline(model_id, device):
    """
    Load the SD XL pipeline with fp16 on GPU or fp32 on CPU,
    attention slicing, CPU offload, and a more efficient scheduler.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    # switch to DPM‐Solver for fewer timesteps/memory
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # reduce attention memory
    pipe.enable_attention_slicing()
    # optionally offload to CPU to save VRAM
    try:
        pipe.enable_model_cpu_offload()
    except:
        pass
    pipe = pipe.to(device)
    # disable safety checker (avoids None returns)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    return pipe

# Load the diffusion pipeline once
print(f"▶ Loading Stable Diffusion XL pipeline on {DEVICE}...")
pipeline = load_pipeline(SD_MODEL_ID, DEVICE)
print("✅ Pipeline ready.")

# Path for reference images
refs_dir = os.path.join(BASE_OUTPUT_FOLDER, "references_images")

for model_name in MODEL_NAMES:
    # Samples live under Mid_metric/<model_name>/samples.json
    samples_path = os.path.join(DATA_ROOT, model_name, SAMPLES_FILENAME)
    pred_dir     = os.path.join(BASE_OUTPUT_FOLDER, f"{model_name}_predict_images")
    os.makedirs(pred_dir, exist_ok=True)

    # Load samples.json for this model
    try:
        with open(samples_path, "r") as f:
            samples = json.load(f)
        print(f"\n▶ [{model_name}] Loaded {len(samples)} samples")
    except Exception as e:
        print(f"❌ Could not load {samples_path}: {e}")
        continue

    # Generate reference images if not done yet
    if not os.path.exists(refs_dir):
        os.makedirs(refs_dir, exist_ok=True)
        print(f"▶ Generating reference images → {refs_dir}/")
        for sample in samples:
            img_id = os.path.splitext(os.path.basename(sample["image_path"]))[0]
            ref_caption = sample.get("reference","") or ""
            if not ref_caption.strip():
                print(f"  ↩ Skipping {img_id}: empty reference")
                continue
            try:
                img = pipeline(ref_caption).images[0]
                path = os.path.join(refs_dir, f"{img_id}_reference.png")
                img.save(path)
                print(f"  🖼 Saved ref: {img_id}")
            except Exception as e:
                print(f"  ⚠ Error ref {img_id}: {e}")
    else:
        print(f"▶ Skipping reference images (already in {refs_dir}/)")

    # Generate this model's prediction images
    print(f"▶ Generating {model_name} predictions → {pred_dir}/")
    for sample in samples:
        img_id = os.path.splitext(os.path.basename(sample["image_path"]))[0]
        pred_caption = sample.get("prediction","") or ""
        if not pred_caption.strip():
            print(f"  ↩ Skipping {img_id}: empty prediction")
            continue
        try:
            img = pipeline(pred_caption).images[0]
            path = os.path.join(pred_dir, f"{img_id}_prediction.png")
            img.save(path)
            print(f"  🖼 Saved pred: {img_id}")
        except Exception as e:
            print(f"  ⚠ Error pred {img_id}: {e}")

print("\n🎉 All done!")
