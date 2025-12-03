import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def inspect_normals(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"--- Inspecting: {os.path.basename(file_path)} ---")

    # 1. Load Raw Data
    normals = np.load(file_path) # Expected shape (H, W, 3)

    print(f"Shape:      {normals.shape}")
    print(f"Data Type:  {normals.dtype}")
    print(f"Min Value:  {normals.min():.4f} (Should be near -1.0)")
    print(f"Max Value:  {normals.max():.4f} (Should be near 1.0)")

    # 2. Check Vector Normalization (Crucial for 3D)
    # Every pixel vector length should be approx 1.0 (excluding background)
    # We calculate the L2 norm of the vectors
    norms = np.linalg.norm(normals, axis=2)

    # Filter out background (where norm is likely 0 or very small)
    valid_mask = norms > 0.1
    valid_norms = norms[valid_mask]

    if len(valid_norms) > 0:
        print(f"Mean Vector Length: {valid_norms.mean():.4f} (Target: 1.0)")
        print(f"Min Vector Length:  {valid_norms.min():.4f}")
        print(f"Max Vector Length:  {valid_norms.max():.4f}")

        if abs(valid_norms.mean() - 1.0) > 0.05:
            print("⚠️ WARNING: Normal vectors are not normalized! Lighting may look wrong.")
        else:
            print("✅ SUCCESS: Vectors are properly normalized.")
    else:
        print("❌ CRITICAL: No valid normal vectors found.")

    # 3. Visualize Components (X, Y, Z) separately
    # This helps see wrinkles that are hidden in the purple mix
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X-Channel (Left/Right slopes) - Good for vertical folds
    axes[0].imshow(normals[:, :, 0], cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title("X Component (Left/Right Slopes)")
    axes[0].axis('off')

    # Y-Channel (Up/Down slopes) - Good for horizontal folds
    axes[1].imshow(normals[:, :, 1], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title("Y Component (Up/Down Slopes)")
    axes[1].axis('off')

    # Z-Channel (Depth facing)
    axes[2].imshow(normals[:, :, 2], cmap='gray', vmin=-1, vmax=1)
    axes[2].set_title("Z Component (Facing Camera)")
    axes[2].axis('off')

    save_path = "logs/debug_normals_components.png"
    plt.savefig(save_path)
    print(f"Saved component visualization to {save_path}")

if __name__ == "__main__":
    # UPDATE THIS PATH to your actual output file
    target_file = "output/normals/test_body_normals.npy"
    inspect_normals(target_file)