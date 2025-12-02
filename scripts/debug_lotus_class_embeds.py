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

def verify_and_fix():
    print("--- Verifying Lotus UNet Class Embeddings ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.path.join(project_root, "checkpoints", "lotus-depth")

    print(f"Loading from: {model_id}")
    pipe = DirectDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # 1. Inspect Configuration
    config = pipe.unet.config
    print(f"\n[UNet Configuration]")
    print(f"config.num_class_embeds: {config.num_class_embeds}")
    print(f"config.class_embed_type: {config.class_embed_type}") # <--- The Suspect

    # 2. Inspect Runtime Layer
    print(f"\n[Runtime Layer]")
    print(f"pipe.unet.class_embedding: {type(pipe.unet.class_embedding)}")

    # 3. Verify the Crash (Baseline)
    print(f"\n[Test 1: Baseline Inference]")
    dummy_tensor = torch.randn(1, 3, 1024, 1024).to(device, dtype=torch.float16)
    try:
        pipe(dummy_tensor, prompt="", guidance_scale=1.0, num_inference_steps=1, output_type="np")
        print("✅ Baseline worked (Unexpected).")
    except ValueError as e:
        print(f"❌ Baseline Failed as expected: {e}")
    except Exception as e:
        print(f"❌ Baseline Failed with other error: {e}")

    # 4. Verify the Fix
    print(f"\n[Test 2: Applying Patch]")
    # We disable the class embedding requirement
    pipe.unet.config.class_embed_type = None
    pipe.unet.class_embedding = None

    try:
        pipe(dummy_tensor, prompt="", guidance_scale=1.0, num_inference_steps=1, output_type="np")
        print("✅ SUCCESS: Inference worked after patching class_embed_type!")
    except Exception as e:
        print(f"❌ FAILED after patching: {e}")

if __name__ == "__main__":
    verify_and_fix()