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

def debug_unet():
    print("--- Debugging Lotus UNet Configuration ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.path.join(project_root, "checkpoints", "lotus-depth")

    # 1. Load Pipeline
    print(f"Loading from: {model_id}")
    pipe = DirectDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)

    # 2. Inspect UNet Config
    unet_config = pipe.unet.config
    print(f"\n[UNet Config]")
    print(f"num_class_embeds: {unet_config.num_class_embeds}")

    if unet_config.num_class_embeds is None:
        print("Model does NOT expect class labels.")
    else:
        print(f"Model EXPECTS class labels! (Count: {unet_config.num_class_embeds})")

    # 3. Test Inference with Dummy Labels
    if unet_config.num_class_embeds:
        print(f"\n[Attempting Fix: Passing Dummy Class Labels]")
        dummy_tensor = torch.randn(1, 3, 1024, 1024).to(device, dtype=torch.float16)

        # Create dummy class labels (Batch Size 1)
        # Usually 0 is a safe default
        class_labels = torch.zeros(1, device=device, dtype=torch.long)

        try:
            # We try passing it via cross_attention_kwargs or directly if the pipe accepts kwargs
            # The error came from self.unet(), so we need to see if pipe passes kwargs to unet.
            output = pipe(
                dummy_tensor,
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=1,
                output_type="np",
                class_labels=class_labels # Try injecting this
            )
            print("✅ SUCCESS: Inference worked with class_labels!")
        except TypeError as e:
            print(f"❌ FAILED: Pipeline does not accept 'class_labels' argument. Error: {e}")
            print("   We might need to use 'cross_attention_kwargs' or patch the pipeline.")
        except Exception as e:
            print(f"❌ FAILED with other error: {e}")

if __name__ == "__main__":
    debug_unet()