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

# Configure Logger
logger.add(os.path.join(project_root, "logs", "step4_variations.log"), rotation="10 MB")

def resize_maps(color, depth, normals, mask, scale_factor=2.0):
    """Upsamples maps to increase point density."""
    if scale_factor == 1.0:
        return color, depth, normals, mask

    h, w = depth.shape
    new_size = (int(w * scale_factor), int(h * scale_factor))

    # Color (Bilinear)
    c_img = Image.fromarray(color).resize(new_size, Image.BILINEAR)

    # Depth (Bicubic for smoothness)
    d_img = Image.fromarray(depth).resize(new_size, Image.BICUBIC)

    # Normals (Bilinear - need renormalization later but ok for now)
    # Normals are (H, W, 3) [-1, 1], shifted to [0, 255] for resizing usually,
    # but here we just resize the float array via PIL if possible or just use CV2/Zoom.
    # Simpler: Resize the visual representation [0,1] then map back.
    n_vis = (normals + 1.0) / 2.0
    n_img = Image.fromarray((n_vis * 255).astype(np.uint8)).resize(new_size, Image.BILINEAR)
    n_new = (np.array(n_img).astype(np.float32) / 255.0) * 2.0 - 1.0

    # Mask (Nearest Neighbor to keep sharp edges)
    m_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(new_size, Image.NEAREST)

    return np.array(c_img), np.array(d_img), n_new, np.array(m_img).astype(np.float32) / 255.0

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    input_dir = Path(project_root) / cfg.paths.input_dir
    mask_dir = Path(project_root) / cfg.paths.segmentation_dir
    depth_dir = Path(project_root) / cfg.paths.depth_dir
    norm_dir = Path(project_root) / cfg.paths.normals_dir
    save_dir = Path(project_root) / cfg.paths.cloud_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    # Define the variations to generate
    variations = [
        {"name": "Linear_Native", "mode": "linear", "scale": 0.5, "upsample": 1.0},
        {"name": "Linear_Dense",  "mode": "linear", "scale": 0.5, "upsample": 2.0}, # Recommended
        {"name": "Linear_Thick",  "mode": "linear", "scale": 1.0, "upsample": 2.0}, # More volume
        {"name": "Inverse_Smooth","mode": "inverse", "scale": 0.5, "upsample": 2.0},
    ]

    for img_path in image_files:
        base_name = img_path.stem
        logger.info(f"Processing: {base_name}")

        try:
            # Load Base Assets
            color_raw = np.array(Image.open(img_path).convert("RGB"))
            mask_path = mask_dir / f"{base_name}_mask.png"
            mask_raw = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

            depth_path = depth_dir / f"{base_name}_depth_raw.png"
            depth_raw_img = np.array(Image.open(depth_path)).astype(np.float32)
            depth_norm_raw = depth_raw_img / 65535.0

            norm_path = norm_dir / f"{base_name}_normals.npy"
            normals_raw = np.load(norm_path)

            # Safety Resize
            if depth_norm_raw.shape != mask_raw.shape:
                d_img = Image.fromarray(depth_norm_raw)
                d_img = d_img.resize((mask_raw.shape[1], mask_raw.shape[0]), Image.BILINEAR)
                depth_norm_raw = np.array(d_img)

        except Exception as e:
            logger.error(f"Failed to load assets: {e}")
            continue

        # Generate Variations
        for v in variations:
            logger.info(f"  -> Generating: {v['name']}")

            # 1. Upsample if needed
            c, d, n, m = resize_maps(
                color_raw, depth_norm_raw, normals_raw, mask_raw,
                scale_factor=v['upsample']
            )

            # 2. Build Cloud
            pcd = create_point_cloud(
                color_img=c,
                depth_map=d,
                normal_map=n,
                mask=m,
                mode=v['mode'],
                depth_scale=v['scale'],
                fov_deg=50.0
            )

            # 3. Clean Outliers (Crucial for Dense clouds)
            # Only apply if we have enough points
            if len(pcd.points) > 100:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

            # 4. Save
            fname = f"{base_name}_{v['name']}.ply"
            o3d.io.write_point_cloud(str(save_dir / fname), pcd)

    logger.success(f"All variations saved to {save_dir}")

if __name__ == "__main__":
    main()