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
import open3d as o3d
from loguru import logger
from src.core.cloud_builder import create_point_cloud

logger.add(os.path.join(project_root, "logs", "step4_cloud.log"), rotation="10 MB")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    input_dir = Path(project_root) / cfg.paths.input_dir
    mask_dir = Path(project_root) / cfg.paths.segmentation_dir
    depth_dir = Path(project_root) / cfg.paths.depth_dir
    norm_dir = Path(project_root) / cfg.paths.normals_dir
    save_dir = Path(project_root) / cfg.paths.cloud_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    for img_path in image_files:
        base_name = img_path.stem
        logger.info(f"Building Point Cloud for: {base_name}")

        try:
            # Load Assets
            color_img = np.array(Image.open(img_path).convert("RGB"))

            mask_path = mask_dir / f"{base_name}_mask.png"
            mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

            depth_path = depth_dir / f"{base_name}_depth_raw.png"
            depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
            depth_norm = depth_raw / 65535.0

            norm_path = norm_dir / f"{base_name}_normals.npy"
            normals = np.load(norm_path)

        except FileNotFoundError as e:
            logger.error(f"Missing asset: {e}")
            continue

        if depth_norm.shape != mask.shape:
            d_img = Image.fromarray(depth_norm)
            d_img = d_img.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
            depth_norm = np.array(d_img)

        # --- TUNING PARAMETERS ---
        # Mode: 'inverse' usually looks rounder for humans than 'linear'
        # Scale: Increased from 0.3 to 0.8 to give volume
        pcd = create_point_cloud(
            color_img=color_img,
            depth_map=depth_norm,
            normal_map=normals,
            mask=mask,
            mode="inverse",   # <--- CHANGED to Inverse (Perspective)
            depth_scale=0.8,  # <--- INCREASED Volume
            fov_deg=50.0      # <--- Narrower FOV flattens the face less
        )

        # --- POST-PROCESSING: Outlier Removal ---
        logger.info("Cleaning up outliers...")
        # nb_neighbors: how many neighbors to check
        # std_ratio: lower = more aggressive removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        pcd_clean = pcd.select_by_index(ind)

        output_path = save_dir / f"{base_name}_tuned.ply"
        o3d.io.write_point_cloud(str(output_path), pcd_clean)
        logger.success(f"Saved Tuned Cloud: {output_path}")

if __name__ == "__main__":
    main()