import sys
import os
import cv2

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

logger.add(os.path.join(project_root, "logs", "step4_final.log"), rotation="10 MB")

def create_corrected_cloud(color, depth, normals, mask, scale_factor=2.0):
    """
    Generates point cloud using relative scaling and gentle smoothing.
    """
    # 1. Calculate Target Size (Relative 2x Upscale)
    # This maintains the exact aspect ratio and projection geometry of the original
    h, w = color.shape[:2]
    target_size = (int(w * scale_factor), int(h * scale_factor))

    logger.info(f"  -> Upsampling from {w}x{h} to {target_size} (Factor: {scale_factor})")

    # 2. Resize Assets
    img_pil = Image.fromarray(color).resize(target_size, Image.BICUBIC)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(target_size, Image.NEAREST)
    depth_pil = Image.fromarray(depth).resize(target_size, Image.BICUBIC)

    # Normals (Careful resize)
    norm_vis = (normals + 1.0) / 2.0
    norm_pil = Image.fromarray((norm_vis * 255).astype(np.uint8)).resize(target_size, Image.BICUBIC)

    # Convert to Numpy
    c_high = np.array(img_pil)
    m_high = np.array(mask_pil).astype(np.float32) / 255.0
    d_high = np.array(depth_pil)
    n_high = (np.array(norm_pil).astype(np.float32) / 255.0) * 2.0 - 1.0

    # 3. GENTLE Smoothing (The "Layer Cake" Fix)
    # Kernel (9,9) is small enough to keep shape, big enough to blend steps
    d_smooth = cv2.GaussianBlur(d_high, (9, 9), 0)

    # 4. Geometry Generation (Linear Mode - The "Good" Strategy)
    height, width = d_smooth.shape
    fov_deg = 50.0
    focal_length = (width / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    cx, cy = width / 2, height / 2

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    valid = m_high > 0.5
    z_raw = d_smooth[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # --- DEPTH SCALING ---
    # 0.5 scale = 50cm thickness (approx).
    # No offset needed for relative clouds, but +1.0 keeps it in front of camera.
    z_metric = (1.0 - z_raw) * 0.5 + 1.0

    x_metric = (u_valid - cx) * z_metric / focal_length
    y_metric = (v_valid - cy) * z_metric / focal_length

    points = np.stack((x_metric, -y_metric, -z_metric), axis=1)
    colors = c_high[valid].astype(np.float64) / 255.0
    norms = n_high[valid].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(norms)

    return pcd

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
        logger.info(f"Processing: {base_name}")

        try:
            color = np.array(Image.open(img_path).convert("RGB"))
            mask_path = mask_dir / f"{base_name}_mask.png"
            mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

            depth_path = depth_dir / f"{base_name}_depth_raw.png"
            depth = np.array(Image.open(depth_path)).astype(np.float32) / 65535.0

            norm_path = norm_dir / f"{base_name}_normals.npy"
            normals = np.load(norm_path)

        except Exception as e:
            logger.error(f"Asset loading failed: {e}")
            continue

        # Generate Cloud (2x Upscale, Gentle Blur)
        pcd = create_corrected_cloud(color, depth, normals, mask, scale_factor=2.0)

        output_path = save_dir / f"{base_name}_final_v3.ply"
        o3d.io.write_point_cloud(str(output_path), pcd)
        logger.success(f"Saved: {output_path}")

if __name__ == "__main__":
    main()