import torch
import sys
import os
import numpy as np
from PIL import Image
from loguru import logger
from .base_model import BaseModel
from typing import Dict, Any

# Reuse the cloned Lotus repo
lotus_repo_path = os.path.abspath("./checkpoints/Lotus")
if lotus_repo_path not in sys.path:
    sys.path.insert(0, lotus_repo_path)

try:
    from pipeline import DirectDiffusionPipeline
except ImportError as e:
    raise ImportError(f"Could not import Lotus Pipeline. Error: {e}")

class LotusNormal(BaseModel):
    def _load_model(self):
        # ... (Same as before) ...
        if hasattr(self.config, "model") and hasattr(self.config.model, "normal"):
            self.norm_config = self.config.model.normal
        else:
            self.norm_config = self.config.normal

        model_id = self.norm_config.repo_id
        logger.info(f"Loading Lotus-G Normal weights from: {model_id}")

        try:
            self.pipe = DirectDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.norm_config.dtype == "fp16" else torch.float32,
                local_files_only=True
            )
            self.pipe.to(self.device)

            # Disable 'projection' class embedding
            if self.pipe.unet.config.class_embed_type is not None:
                self.pipe.unet.config.class_embed_type = None
                self.pipe.unet.class_embedding = None

            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()

        except Exception as e:
            logger.error(f"Failed to load Lotus Pipeline: {e}")
            raise

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        # Resize to be divisible by 32
        w, h = image.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32

        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)

        img_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device, dtype=self.pipe.dtype)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        logger.debug("Running Lotus-G Normal Inference...")

        original_size = image.size
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            output = self.pipe(
                input_tensor,
                prompt="",
                guidance_scale=1.0,
                num_inference_steps=self.norm_config.steps,
                output_type="np"
            )

            # 1. Extract Output
            # Lotus with output_type="np" returns [0, 1] RGB image
            normals_rgb = output.images[0]

            # 2. Resize back to original resolution (while still in RGB mode)
            # We use uint8 for resizing to avoid float artifacts, then convert back
            norm_img = Image.fromarray((normals_rgb * 255).astype(np.uint8))
            if norm_img.size != original_size:
                norm_img = norm_img.resize(original_size, Image.BILINEAR)

            # 3. Convert [0, 255] -> [0, 1] -> [-1, 1]
            normals_np = (np.array(norm_img).astype(np.float32) / 255.0) * 2.0 - 1.0

            # 4. Re-Normalize Vectors (CRITICAL FIX)
            # Resizing changes vector lengths. We must force length=1.0.
            # Calculate L2 norm per pixel
            norms = np.linalg.norm(normals_np, axis=2, keepdims=True)
            # Avoid division by zero
            normals_np = normals_np / (norms + 1e-6)

            # 5. Create Visualization [0, 1]
            normals_vis = (normals_np + 1.0) / 2.0

        return {
            "normals": normals_np, # [-1, 1] Normalized Vectors
            "vis": normals_vis     # [0, 1] RGB for Display
        }