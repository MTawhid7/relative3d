# Relative-3D: High-Fidelity Point Cloud Generation

This project implements a state-of-the-art AI pipeline (as of Dec. 2025) to generate a high-fidelity relative 3D point cloud from a single 2D image. The goal is to capture detailed geometry, including fabric wrinkles and folds, to serve as a high-quality asset for visualization and virtual try-on applications.


## The Pipeline

The system is a four-stage modular pipeline, where each stage produces a specific data asset.

1.  **Stage 1: Segmentation (SAM 3)**
    *   **Goal:** Isolate the subject from the background to prevent noise.
    *   **Tool:** Meta's **SAM 3** with "Promptable Concept Segmentation" (`"person wearing t-shirt"`).
    *   **Output:** A clean, high-precision binary mask.

2.  **Stage 2: Depth Estimation (Lotus-G Depth)**
    *   **Goal:** Infer the global geometry and volume of the subject.
    *   **Tool:** **Lotus-G v2.1**, a generative diffusion model that excels at capturing surface texture.
    *   **Output:** A 16-bit disparity map representing the scene's relative depth.

3.  **Stage 3: Normal Estimation (Lotus-G Normal)**
    *   **Goal:** Capture the high-frequency surface details (wrinkles, seams) that depth maps smooth over.
    *   **Tool:** **Lotus-G v1.1 Normal**, which shares the same generative architecture as the depth model for perfect alignment.
    *   **Output:** A `.npy` file containing 3D unit vectors (`[-1, 1]`) for every pixel, describing the surface orientation.

4.  **Stage 4: Point Cloud Assembly**
    *   **Goal:** Combine all data into a final, high-quality 3D asset.
    *   **Tools:** **OpenCV** (for smoothing), **NumPy** (for math), and **Open3D** (for `.ply` generation).
    *   **Process:**
        1.  **Upsample** all maps by 2x to create a dense point cloud.
        2.  Apply a gentle **Gaussian Blur** to the depth map to eliminate quantization artifacts ("Layer Cake" effect).
        3.  Back-project pixels into 3D space using a stable **Linear Mapping** ("Bas-Relief") strategy.
        4.  Attach color and normal data to each point.
        5.  Save the final asset as a `.ply` file.

## Technology Stack

*   **Python:** 3.12
*   **Core AI:** PyTorch 2.9, Diffusers, Transformers
*   **Configuration:** Hydra
*   **Geometry:** Open3D, OpenCV
*   **Models:** SAM 3, Lotus-G Depth v2.1, Lotus-G Normal v1.1

## Project Architecture

The codebase is modular, allowing individual stages to be run, tested, and upgraded independently.

```text
relative3d/
├── checkpoints/         # Local model weights (SAM3, Lotus, etc.)
├── config/             # Hydra configuration files
├── input/              # Source images
├── output/             # Generated assets (masks, depth, clouds)
├── scripts/            # Runner scripts for each pipeline stage
├── src/
│   ├── models/         # Wrappers for AI models
│   └── core/           # Core geometry processing logic
├── logs/               # Log files
└── environment.yaml    # Conda environment
```

## Development Journey & Key Challenges

This project involved significant debugging due to the use of bleeding-edge research models. Key issues and their solutions are documented here:

*   **Challenge 1: The "Layer Cake" Effect.**
    *   **Problem:** The final point cloud was stratified into distinct, flat layers.
    *   **Cause:** The Lotus depth model outputted slightly quantized values. When scaled, these microscopic steps became macroscopic gaps.
    *   **Solution:** Apply a gentle **Gaussian Blur** (`(9,9)` kernel) to the upsampled depth map to "melt" the steps into a smooth gradient.

*   **Challenge 2: The "Backward Lean".**
    *   **Problem:** The 3D model was consistently leaning backward.
    *   **Cause:** An incorrect assumption that the monocular depth estimate was slanted. Applying a corrective rotation to the stable **Linear Mapping** mode introduced the tilt.
    *   **Solution:** Remove all artificial rotations. The Linear Mapping strategy is naturally aligned with the camera plane.

*   **Challenge 3: "Holes in the Face".**
    *   **Problem:** The subject's face had large holes.
    *   **Cause:** Overly aggressive `remove_statistical_outlier` filtering, which misidentified the complex geometry of the face as noise.
    *   **Solution:** Disable the outlier removal step entirely. The SAM 3 mask is clean enough to prevent most stray points.

*   **Challenge 4: Model API Instability.**
    *   **Problem:** The SOTA models (SAM 3, Lotus) used custom, undocumented, or non-standard APIs within the `transformers` and `diffusers` libraries.
    *   **Solutions:**
        *   **SAM 3:** Switched to the "Session-Based" video API, as it treats all inputs as frames.
        *   **Lotus:** Manually downloaded the pipeline code from the official GitHub, patched the UNet config to disable unused `class_embed_type`, and explicitly disabled Classifier-Free Guidance (`guidance_scale=1.0`) to resolve tensor batch size mismatches.

## Installation

1.  **Clone the Repository & Submodules:**
    ```bash
    git clone https://github.com/mtawhid7/relative3d.git
    cd relative3d
    git clone https://github.com/EnVision-Research/Lotus.git checkpoints/Lotus
    ```

2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate relative3d
    ```

3.  **Download Model Checkpoints:**
    ```bash
    # Download SAM 3
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/sam3', local_dir='./checkpoints/sam3')"

    # Download Lotus-G Depth
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jingheya/lotus-depth-g-v2-1-disparity', local_dir='./checkpoints/lotus-depth')"

    # Download Lotus-G Normal
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jingheya/lotus-normal-g-v1-1', local_dir='./checkpoints/lotus-normal')"
    ```

## Usage

1.  Place your input image(s) (e.g., `my_photo.jpg`) into the `input/` folder.
2.  Run the pipeline stages sequentially:
    ```bash
    python scripts/step1_test_segmentation.py
    python scripts/step2_test_depth.py
    python scripts/step3_test_normals.py
    python scripts/step4_final_build.py
    ```
3.  The final asset will be saved in `output/pointclouds/my_photo_final_v3.ply`.
