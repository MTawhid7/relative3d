import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from src.models.depth_estimator import LotusDepth

logger.add(os.path.join(project_root, "logs", "step2_depth.log"), rotation="10 MB")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    input_dir = Path(project_root) / cfg.paths.input_dir
    mask_dir = Path(project_root) / cfg.paths.segmentation_dir
    save_dir = Path(project_root) / cfg.paths.depth_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    depth_model = LotusDepth(cfg)

    for img_path in image_files:
        logger.info(f"Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # 1. Load Mask
        base_name = img_path.stem
        mask_path = mask_dir / f"{base_name}_mask.png"
        mask = None

        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            # Ensure strict size matching
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # 2. Inference
        result = depth_model.predict(image)
        depth_map = result['depth'] # This is now same size as image

        # 3. Apply Masking
        if mask is not None:
            # Safety check for dimensions
            if depth_map.shape != mask.shape:
                logger.warning(f"Shape mismatch handled: Depth {depth_map.shape} vs Mask {mask.shape}")
                # Resize depth to match mask exactly
                d_img = Image.fromarray(depth_map)
                d_img = d_img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
                depth_map = np.array(d_img)

            depth_map = depth_map * mask

            # Contrast Stretch on Body Only
            body_vals = depth_map[mask > 0.5]
            if len(body_vals) > 0:
                d_min, d_max = body_vals.min(), body_vals.max()
                depth_map = (depth_map - d_min) / (d_max - d_min + 1e-6)
                depth_map = np.clip(depth_map, 0, 1) * mask

        # 4. Save Outputs
        # Raw (16-bit)
        depth_uint16 = (depth_map * 65535).astype(np.uint16)
        Image.fromarray(depth_uint16).save(save_dir / f"{base_name}_depth_raw.png")

        # Vis (Magma)
        plt.figure(figsize=(10, 5))
        plt.imshow(depth_map, cmap='magma', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.savefig(save_dir / f"{base_name}_depth_vis.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

        logger.success(f"Saved: {save_dir / f'{base_name}_depth_vis.jpg'}")

if __name__ == "__main__":
    main()