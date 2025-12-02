import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from loguru import logger
from .base_model import BaseModel
from typing import Dict, Any

class SAM3Segmentor(BaseModel):
    def _load_model(self):
        # Handle nested config structure
        if hasattr(self.config, "model") and hasattr(self.config.model, "segmentation"):
            self.seg_config = self.config.model.segmentation
        else:
            self.seg_config = self.config.segmentation

        model_path = self.seg_config.repo_id
        logger.info(f"Loading SAM 3 from local path: {model_path}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load SAM 3: {e}")
            raise

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        prompt = self.seg_config.prompt
        logger.debug(f"Segmenting concept: '{prompt}'")

        # 1. Initialize Video Session
        try:
            inference_session = self.processor.init_video_session(
                video=[image],
                inference_device=self.device,
                dtype=self.model.dtype
            )
        except Exception as e:
            logger.error(f"Failed to init video session: {e}")
            raise

        # 2. Add Text Prompt
        # --- FIX: Remove frame_idx and obj_id ---
        try:
            self.processor.add_text_prompt(
                inference_session=inference_session,
                text=prompt
            )
        except Exception as e:
            logger.error(f"Failed to add text prompt: {e}")
            raise
        # ----------------------------------------

        # 3. Run Inference
        with torch.no_grad():
            # We still need frame_idx here to tell the model WHICH frame to decode
            outputs = self.model(
                inference_session=inference_session,
                frame_idx=0
            )

        # 4. Post-process
        original_size = (image.height, image.width)

        results = self.processor.postprocess_outputs(
            inference_session=inference_session,
            model_outputs=outputs,
            original_sizes=[original_size]
        )

        # --- DEBUG & FIX START ---
        logger.debug(f"Post-process result type: {type(results)}")
        if isinstance(results, dict):
            logger.debug(f"Result keys: {results.keys()}")

        # Robust Extraction Strategy
        res = None

        # Case A: Dictionary keyed by integer frame_idx (Standard)
        if isinstance(results, dict):
            if 0 in results:
                res = results[0]
            # Case B: Dictionary keyed by string frame_idx
            elif "0" in results:
                res = results["0"]
            # Case C: The dictionary IS the result (no frame nesting)
            elif 'pred_masks' in results or 'masks' in results:
                res = results
            # Case D: Take the first available value
            elif len(results) > 0:
                first_key = list(results.keys())[0]
                logger.warning(f"Frame 0 not found. Using key: {first_key}")
                res = results[first_key]

        # Case E: List (Standard Image Processing)
        elif isinstance(results, list):
            if len(results) > 0:
                res = results[0]

        if res is None:
            logger.error("Could not extract results from processor output.")
            # Return empty mask to prevent crash
            return {"masks": np.zeros((1, *original_size), dtype=np.uint8), "scores": [0.0]}

        # --- DEBUG & FIX END ---

        # Handle Output Keys
        if 'pred_masks' in res:
            masks_np = res['pred_masks'].cpu().numpy()
            scores = res['iou_scores'].cpu().numpy()
        elif 'masks' in res:
            masks_np = res['masks'].cpu().numpy()
            scores = res.get('scores', np.ones(masks_np.shape[0])).cpu().numpy()
        else:
            logger.warning(f"Unexpected keys in res: {res.keys()}")
            return {"masks": np.zeros((1, *original_size), dtype=np.uint8), "scores": [0.0]}

        # Threshold & Clean
        masks_np = (masks_np > 0.0).astype(np.uint8)

        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(0)
            if scores.ndim > 1: scores = scores.squeeze(0)

        return {"masks": masks_np, "scores": scores}