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
from src.models.normal_estimator import LotusNormal

logger.add(os.path.join(project_root, "logs", "step3_normals.log"), rotation="10 MB")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Paths
    input_dir = Path(project_root) / cfg.paths.input_dir
    mask_dir = Path(project_root) / cfg.paths.segmentation_dir
    save_dir = Path(project_root) / cfg.paths.normals_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    # Init Model
    normal_model = LotusNormal(cfg)

    for img_path in image_files:
        logger.info(f"Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # Load Mask
        base_name = img_path.stem
        mask_path = mask_dir / f"{base_name}_mask.png"
        mask = None
        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32) # Binary 0/1
            # Expand mask for 3 channels (H, W, 1) -> (H, W, 3)
            mask_3c = np.stack([mask]*3, axis=-1)

        # Inference
        result = normal_model.predict(image)
        normals_np = result['normals'] # [-1, 1]
        normals_vis = result['vis']    # [0, 1]

        # Apply Masking
        if mask is not None:
            # For normals, background is usually (0,0,0) or (0,0,1)
            # We use Black (0,0,0) for the visualization to match depth style
            normals_vis = normals_vis * mask_3c

            # For raw data, we might want 0 or NaN, let's stick to 0 for now
            normals_np = normals_np * mask_3c

        # Save Raw (.npy) - Best for 3D reconstruction loading
        np.save(save_dir / f"{base_name}_normals.npy", normals_np)

        # Save Visualization (.png)
        plt.figure(figsize=(10, 5))
        plt.imshow(normals_vis)
        plt.axis('off')
        plt.title("Lotus-G Normals (Masked)")
        plt.savefig(save_dir / f"{base_name}_normals_vis.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

        logger.success(f"Saved normals to {save_dir}")

if __name__ == "__main__":
    main()