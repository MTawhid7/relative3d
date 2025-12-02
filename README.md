# Relative-3D: High-Fidelity Garment Reconstruction

A modular AI pipeline for generating high-resolution relative 3D point clouds from single images. This project leverages state-of-the-art models (late 2025) to capture intricate fabric details, wrinkles, and folds for virtual try-on and 3D visualization.

## 🚀 Tech Stack (2025)

- **Segmentation:** [SAM 3 (Meta)](https://github.com/facebookresearch/sam3) - Promptable concept segmentation.
- **Depth Estimation:** Lotus-G (v2.1) - Diffusion-based generative depth.
- **Surface Details:** StableNormal - Crisp surface normal estimation.
- **Geometry Processing:** MeshLib (GPU) - Real-time point cloud generation.
- **Orchestration:** Hydra + PyTorch 2.9.

## 📂 Project Structure

```text
relative3d/
├── config/             # Hydra configuration files
├── src/
│   ├── models/         # Wrappers for SAM3, Lotus, StableNormal
│   ├── core/           # Geometry processing logic
│   └── pipeline/       # Execution orchestration
├── environment.yaml    # Conda environment spec
└── step1_test_...py    # Unit test scripts
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:mtawhid7/relative3d.git
   cd relative3d
   ```

2. **Create Conda Environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate relative3d
   ```

3. **Download Model Weights:**
   Weights are not included in the repo. Download them to `checkpoints/`:
   ```bash
   huggingface-cli download facebook/sam3 --local-dir ./checkpoints/sam3
   ```

## ⚡ Usage

Run the segmentation test:
```bash
python step1_test_segmentation.py
```

## 📝 License
Private Research Project.
```