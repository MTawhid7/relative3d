import torch
import numpy as np
import meshlib.mrmeshpy as mm # GPU Geometry Library
from pathlib import Path
from loguru import logger
from PIL import Image

from src.models.segmentor import SAM3Segmentor
from src.models.depth_estimator import LotusDepth
from src.models.normal_estimator import StableNormal
from src.core.cloud_builder import back_project_gpu

class PipelineRunner:
    def __init__(self, config):
        self.cfg = config
        self.device = config.project.device

        # Initialize Models
        logger.info("Loading SAM 3 (Nov 2025 Release)...")
        self.seg = SAM3Segmentor(config)

        logger.info("Loading Lotus-G v2.1...")
        self.depth = LotusDepth(config)

        logger.info("Loading StableNormal...")
        self.normal = StableNormal(config)

    def run(self, image_path: Path):
        logger.info(f"Processing: {image_path.name}")

        # 1. Load Image
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)

        # 2. Segmentation (SAM 3)
        # Returns binary mask of the user/t-shirt
        mask = self.seg.predict(img_pil)
        logger.debug(f"Mask generated: {mask.shape}")

        # 3. Depth Estimation (Lotus)
        # Returns 16-bit normalized depth map
        depth_map = self.depth.predict(img_pil)

        # 4. Normal Estimation (StableNormal)
        # Returns surface normals for crisp wrinkle rendering
        normal_map = self.normal.predict(img_pil)

        # 5. Geometry Generation (MeshLib GPU)
        # This function (to be implemented in core) converts maps to .ply
        pcd = back_project_gpu(
            color=img_np,
            depth=depth_map,
            mask=mask,
            normals=normal_map,
            fov=60.0 # Approximate FOV for Gemini portraits
        )

        return pcd