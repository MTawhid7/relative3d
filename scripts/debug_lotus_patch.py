import torch
import sys
import os
import numpy as np

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
lotus_repo_path = os.path.join(project_root, "checkpoints", "Lotus")
if lotus_repo_path not in sys.path:
    sys.path.insert(0, lotus_repo_path)

try:
    from pipeline import DirectDiffusionPipeline
except ImportError:
    print("CRITICAL: Could not import pipeline.")
    sys.exit(1)

def debug_and_patch():
    print("--- Debugging Lotus Runtime Attributes ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.path.join(project_root, "checkpoints", "lotus-depth")

    print(f"Loading from: {model_id}")
    pipe = DirectDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # 1. Inspect the ACTUAL attribute causing the crash
    # The config might say None, but the attribute might be set.
    actual_num_embeds = getattr(pipe.unet, "num_class_embeds", "Missing")
    print(f"\n[Runtime Check]")
    print(f"pipe.unet.config.num_class_embeds: {pipe.unet.config.num_class_embeds}")
    print(f"pipe.unet.num_class_embeds:        {actual_num_embeds}")

    # 2. Inspect Class Embedding Layer
    if hasattr(pipe.unet, "class_embedding"):
        print(f"pipe.unet.class_embedding:         {type(pipe.unet.class_embedding)}")
    else:
        print(f"pipe.unet.class_embedding:         None")

    # 3. Attempt Patching
    # If the model thinks it needs labels but the pipeline doesn't support them,
    # and this is a depth model (unconditional), we can likely disable this requirement.
    if actual_num_embeds is not None and actual_num_embeds > 0:
        print(f"\n[PATCHING] Force-disabling class embeddings on UNet...")
        pipe.unet.num_class_embeds = None
        pipe.unet.config.num_class_embeds = None
        pipe.unet.class_embedding = None
        print("Patch applied.")

    # 4. Test Inference
    print(f"\n[Testing Inference Post-Patch]")
    dummy_tensor = torch.randn(1, 3, 1024, 1024).to(device, dtype=torch.float16)

    try:
        output = pipe(
            dummy_tensor,
            prompt="",
            guidance_scale=1.0,
            num_inference_steps=1,
            output_type="np"
        )
        print("✅ SUCCESS: Inference worked after patching!")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    debug_and_patch()