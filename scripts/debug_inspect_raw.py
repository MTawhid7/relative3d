import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

def inspect_raw(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"--- Inspecting: {os.path.basename(file_path)} ---")

    # 1. Load as 16-bit integer
    img = Image.open(file_path)
    data = np.array(img)

    print(f"Dimensions: {data.shape}")
    print(f"Data Type:  {data.dtype}")
    print(f"Min Value:  {data.min()} (0 is black)")
    print(f"Max Value:  {data.max()} (65535 is white)")
    print(f"Mean Value: {data.mean():.2f}")

    if data.max() == 0:
        print("❌ CRITICAL: Image is truly empty (all zeros).")
    else:
        print("✅ SUCCESS: Image contains data.")

    # 2. Generate a Histogram to show data distribution
    plt.figure(figsize=(10, 4))
    plt.hist(data.flatten(), bins=100, color='gray', log=True)
    plt.title("Depth Value Distribution (Log Scale)")
    plt.xlabel("Pixel Intensity (0-65535)")
    plt.ylabel("Frequency")
    plt.savefig("logs/debug_depth_histogram.png")
    print("Saved histogram to logs/debug_depth_histogram.png")

    # 3. Save a 'Human Readable' version (Contrast Stretched)
    # We ignore 0 (background) for contrast stretching if possible
    valid_pixels = data[data > 0]
    if len(valid_pixels) > 0:
        p5, p95 = np.percentile(valid_pixels, 5), np.percentile(valid_pixels, 95)
        # Clip and normalize to 0-255
        stretched = np.clip(data, p5, p95)
        stretched = ((stretched - p5) / (p95 - p5) * 255).astype(np.uint8)

        Image.fromarray(stretched).save("logs/debug_depth_visible.png")
        print("Saved human-readable check to logs/debug_depth_visible.png")

if __name__ == "__main__":
    # Point this to your actual output file
    # UPDATE THIS PATH based on where your last run saved the file
    target_file = "output/depth/test_body_depth_raw.png"
    inspect_raw(target_file)