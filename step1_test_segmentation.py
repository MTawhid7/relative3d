import hydra
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Import the class we just wrote
from src.models.segmentor import SAM3Segmentor

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # --- DEBUG START ---
    print("Full Configuration Loaded by Hydra:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 20)
    # --- DEBUG END ---
    
    # Setup
    input_dir = Path(cfg.paths.input_dir)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Image
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    test_image_path = image_files[0]
    logger.info(f"Processing: {test_image_path.name}")
    image = Image.open(test_image_path).convert("RGB")

    # Run Inference
    segmentor = SAM3Segmentor(cfg)
    result = segmentor.predict(image)

    masks = result['masks']   # (3, H, W)
    scores = result['scores'] # (3,)

    # Visualization: 1 Row, 4 Columns (Original + 3 Masks)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Col 1: Original
    axes[0].imshow(image)
    axes[0].set_title("Original Input")
    axes[0].axis('off')

    # Cols 2-4: Masks
    for i in range(3):
        # Create Red Overlay
        overlay = np.array(image).copy()
        mask_layer = masks[i]

        # Darken background (where mask is 0)
        overlay[mask_layer == 0] = (overlay[mask_layer == 0] * 0.3).astype(np.uint8)

        # Add Red Tint to foreground (optional, or just keep it bright)
        # overlay[mask_layer == 1, 0] = np.clip(overlay[mask_layer == 1, 0] + 50, 0, 255)

        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"Mask {i}\nScore: {scores[i]:.3f}")
        axes[i+1].axis('off')

        # Save individual raw mask for debugging
        Image.fromarray(masks[i] * 255).save(output_dir / f"mask_{i}.png")

    # Save Comparison
    save_path = output_dir / "step1_mask_comparison.jpg"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.success(f"Saved comparison to: {save_path}")
    logger.info("Download this image to select the best mask index.")

if __name__ == "__main__":
    main()