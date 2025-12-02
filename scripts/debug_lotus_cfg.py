import torch
import sys
import os
import numpy as np
from PIL import Image

# --- Setup Paths (Same as depth_estimator.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
lotus_repo_path = os.path.join(project_root, "checkpoints", "Lotus")

if lotus_repo_path not in sys.path:
    sys.path.insert(0, lotus_repo_path)
# ------------------------------------------------

try:
    from pipeline import DirectDiffusionPipeline
except ImportError:
    print(f"CRITICAL: Could not import pipeline from {lotus_repo_path}")
    sys.exit(1)

def debug_pipeline_shapes():
    print(f"--- Debugging Lotus Pipeline Tensor Shapes ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.path.join(project_root, "checkpoints", "lotus-depth")

    # 1. Load Pipeline
    print(f"Loading from: {model_id}")
    pipe = DirectDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # 2. Inspect Default Configuration
    print(f"\n[Configuration Check]")
    # Check if the pipeline has a default guidance scale
    # Many pipelines default to 7.5 or 5.0 in their __init__ or config
    default_guidance = getattr(pipe, "guidance_scale", "Not Set")
    print(f"Pipeline Default Guidance Scale: {default_guidance}")

    # 3. Simulate Inputs
    print(f"\n[Tensor Shape Analysis]")
    dummy_image = torch.randn(1, 3, 1024, 1024).to(device, dtype=torch.float16)
    print(f"Input Image Tensor Shape: {dummy_image.shape} (Batch Size: {dummy_image.shape[0]})")

    # 4. Test Text Encoding (The source of Batch Size expansion)
    # We simulate what happens inside the pipe when guidance_scale > 1
    print(f"\n--- Simulating CFG (Guidance Scale > 1.0) ---")
    try:
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        combined_embeds = torch.cat([negative_embeds, prompt_embeds])
        print(f"Text Embeddings (Positive): {prompt_embeds.shape}")
        print(f"Text Embeddings (Negative): {negative_embeds.shape}")
        print(f"Combined Text Batch Size:   {combined_embeds.shape[0]}")

        if combined_embeds.shape[0] != dummy_image.shape[0]:
            print(f"❌ MISMATCH DETECTED: Text Batch ({combined_embeds.shape[0]}) != Image Batch ({dummy_image.shape[0]})")
            print("   This confirms that the pipeline expects the image latents to be duplicated, but they likely aren't.")
    except Exception as e:
        print(f"Error during encoding simulation: {e}")

    # 5. Test Fix (Guidance Scale = 1.0)
    print(f"\n--- Simulating Fix (Guidance Scale = 1.0) ---")
    try:
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False # CFG Disabled
        )
        print(f"Text Embeddings (No CFG):   {prompt_embeds.shape}")
        print(f"Text Batch Size:            {prompt_embeds.shape[0]}")

        if prompt_embeds.shape[0] == dummy_image.shape[0]:
            print(f"✅ MATCH CONFIRMED: Text Batch ({prompt_embeds.shape[0]}) == Image Batch ({dummy_image.shape[0]})")
            print("   Setting guidance_scale=1.0 solves the batch size collision.")

    except Exception as e:
        print(f"Error during fix simulation: {e}")

if __name__ == "__main__":
    debug_pipeline_shapes()