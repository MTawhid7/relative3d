import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from loguru import logger
from .base_model import BaseModel
from typing import Dict, Any

class SAM3Segmentor(BaseModel):
    def _load_model(self):
        # --- FIX: Handle nested config structure ---
        # The debug output shows the config is nested as cfg.model.segmentation
        if hasattr(self.config, "model") and hasattr(self.config.model, "segmentation"):
            self.seg_config = self.config.model.segmentation
        else:
            # Fallback if config structure changes to flat later
            self.seg_config = self.config.segmentation

        model_path = self.seg_config.repo_id
        # -------------------------------------------

        logger.info(f"Loading SAM 3 from local path: {model_path}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load SAM 3: {e}")
            raise

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        # Use the locally saved config reference
        prompt = self.seg_config.prompt
        logger.debug(f"Segmenting concept: '{prompt}'")

        inputs = self.processor(
            images=image,
            text_prompts=[prompt],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs.original_sizes,
            inputs.reshaped_input_sizes
        )[0]

        scores = outputs.iou_scores[0].cpu().numpy()
        masks_np = results.cpu().numpy().astype(np.uint8)

        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(0)
            scores = scores.squeeze(0)

        return {"masks": masks_np, "scores": scores}