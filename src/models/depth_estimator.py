import torch
import sys
import os
import numpy as np
from PIL import Image
from loguru import logger
from .base_model import BaseModel
from typing import Dict, Any

# Add the cloned repo to sys.path
lotus_repo_path = os.path.abspath("./checkpoints/Lotus")
if lotus_repo_path not in sys.path:
    sys.path.insert(0, lotus_repo_path)

try:
    from pipeline import DirectDiffusionPipeline
except ImportError as e:
    raise ImportError(f"Could not import Lotus Pipeline from {lotus_repo_path}. Error: {e}")

class LotusDepth(BaseModel):
    def _load_model(self):
        if hasattr(self.config, "model") and hasattr(self.config.model, "depth"):
            self.depth_config = self.config.model.depth
        else:
            self.depth_config = self.config.depth

        model_id = self.depth_config.repo_id
        logger.info(f"Loading Lotus-G Depth weights from: {model_id}")

        try:
            self.pipe = DirectDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.depth_config.dtype == "fp16" else torch.float32,
                local_files_only=True
            )
            self.pipe.to(self.device)

            # Disable 'projection' class embedding to prevent crash
            if self.pipe.unet.config.class_embed_type is not None:
                self.pipe.unet.config.class_embed_type = None
                self.pipe.unet.class_embedding = None

            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()

        except Exception as e:
            logger.error(f"Failed to load Lotus Pipeline: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # Resize image to be divisible by 32 to prevent VAE rounding errors
        # This ensures the output is predictable, though we still resize back later.
        w, h = image.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32

        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)

        img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device, dtype=self.pipe.dtype)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        logger.debug("Running Lotus-G Depth Inference...")

        # Capture original size for restoring later
        original_size = image.size # (W, H)

        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            output = self.pipe(
                input_tensor,
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=self.depth_config.steps,
                output_type="np"
            )

            # Extract disparity
            disparity = output.images[0]

            # Handle channel dimensions
            if disparity.ndim == 3:
                disparity = disparity.mean(axis=-1) if disparity.shape[-1] == 3 else disparity.squeeze(-1)

            # --- FIX 1: Resize back to original resolution ---
            # disparity is (H, W). PIL resize expects (W, H).
            disp_img = Image.fromarray(disparity)
            if disp_img.size != original_size:
                disp_img = disp_img.resize(original_size, Image.BILINEAR)
            disparity = np.array(disp_img)
            # -------------------------------------------------

            # --- FIX 2: Better Normalization (Robust) ---
            # We use percentiles to ignore outliers (like black borders)
            p_min, p_max = np.percentile(disparity, 1), np.percentile(disparity, 99)
            disparity_norm = np.clip((disparity - p_min) / (p_max - p_min + 1e-6), 0, 1)

            # --- FIX 3: Do NOT invert for raw data ---
            # For relative point clouds of clothing, Disparity IS the data we want.
            # It captures the surface variation linearly.
            # We return the same map for both keys, but 'depth' implies the Z-buffer usage.
            depth_norm = disparity_norm

        return {
            "depth": depth_norm,
            "disparity": disparity_norm
        }