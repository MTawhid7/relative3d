import sys
import os

# --- FIX: Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

import hydra
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from src.models.segmentor import SAM3Segmentor

# Configure Logger
logger.add(os.path.join(project_root, "logs", "step1_segmentation.log"), rotation="10 MB")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Resolve Paths relative to project root
    input_dir = Path(project_root) / cfg.paths.input_dir
    save_dir = Path(project_root) / cfg.paths.segmentation_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    # Initialize Model
    segmentor = SAM3Segmentor(cfg)

    for img_path in image_files:
        logger.info(f"Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # Inference
        result = segmentor.predict(image)

        masks = result['masks']

        if len(masks) == 0:
            logger.warning(f"No object found in {img_path.name}")
            continue

        # Select Best Mask
        best_mask = masks[0]

        # Save Clean Outputs
        base_name = img_path.stem

        # A. Save Binary Mask
        mask_filename = save_dir / f"{base_name}_mask.png"
        Image.fromarray(best_mask * 255).save(mask_filename)

        # B. Save Visualization
        vis_img = np.array(image).copy()
        gray_bg = np.mean(vis_img, axis=2).astype(np.uint8)
        vis_img[best_mask == 0] = np.stack([gray_bg//3]*3, axis=-1)[best_mask == 0]

        vis_filename = save_dir / f"{base_name}_vis.jpg"
        Image.fromarray(vis_img).save(vis_filename)

        logger.success(f"Saved: {mask_filename}")

if __name__ == "__main__":
    main()