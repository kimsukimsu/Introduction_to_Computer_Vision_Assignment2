# MVTec-Shift: A Synthetic Benchmark for Industrial Anomaly Detection under Environmental Domain Shifts

> **Project for Introduction to Computer Vision 2025-2 (Assignment #2)**

### [Report](https://drive.google.com/file/d/1w7fHMe6SER9b6y6vDmaps3oFbhORqZkY/view?usp=drivesdk)

## 1. Project Overview

**MVTec-Shift** is a novel synthetic dataset designed to benchmark the robustness of unsupervised anomaly detection models against environmental domain shifts.

Existing datasets like MVTec AD are captured under controlled studio lighting. However, real-world industrial environments often suffer from **low light, grease, noise, and oxidation (rust)**. Standard models trained on clean data often fail in these conditions, misclassifying functional but environmentally affected parts as defects.

This project utilizes **Generative AI (Stable Diffusion - InstructPix2Pix)** to transform the environmental "mood" of the MVTec AD *Metal Nut* category while preserving structural integrity.

### 📊 Dataset Comparison (Uniqueness)

| Feature | MVTec AD (Original) | **MVTec-Shift (Ours)** |
| :--- | :--- | :--- |
| **Environment** | Controlled Studio Lighting | **Real-world Factory Conditions** |
| **Domain Shifts** | None (Clean Background) | **Dark, Greasy, Rusty, Noisy** |
| **Data Type** | Real Photography | **Synthetic (Generative AI)** |
| **Goal** | Defect Detection | **Robustness against Domain Shift** |

## 📂 2. Dataset Structure & Statistics

We extended the *Metal Nut* class from MVTec AD.
* **Total Classes:** 1 (Metal Nut) with 4 Environmental Domains
* **Total Images:** 220 (Original Train) + 880 (Generated Train) = 1,100 Training Images

```bash
MVTec-Shift/
├── train/
│   └── good/              # Original Training Data (220 images)
├── test/                  # Original Defect Data (115 images)
│   ├── bent/
│   ├── color/
│   ├── flip/
│   └── scratch/           
└── generated_data/        # [NEW] Synthetic Data (880 images total)
    ├── dark/              # Low-light condition (220 images)
    ├── greasy/            # Oil stains & specular highlights (220 images)
    ├── rusty/             # Oxidation & weathering (220 images)
    └── noisy/             # Sensor noise & dust (220 images)

```
# 🚀3. Installation

Ensure you have Python installed (3.8+ recommended). Install the required dependencies:

```
pip install torch torchvision diffusers transformers accelerate PIL tqdm matplotlib seaborn scikit-learn
```

# 💻4. Usage

Step 1: Data Generation
Run data_generation.py to create synthetic images. This script uses InstructPix2Pix to alter the image style based on text prompts.


```
# Generate 'Dark' mood images from the training set
python data_generation.py \
    --source ./mvtec_ad/metal_nut/train/good \
    --output ./Result_MetalNut \
    --mood dark

# Generate all moods (Dark, Greasy, Rusty, Noisy)
python data_generation.py \
    --source ./mvtec_ad/metal_nut/train/good \
    --output ./Result_MetalNut \
    --mood all
```

Key Arguments:

--source: Path to the input image directory.

--mood: Target environment (dark, greasy, rusty, noisy, or all).

image_guidance_scale (in code): Set to 1.5 to ensure structural preservation of defects.

Step 2: Distribution Analysis (Evaluation)

Run distribution_anomaly.py to visualize how the environmental shift affects the anomaly scores (CLIP Cosine Distance).

```
python distribution_anomaly.py \
    --orig_root ./mvtec_ad/metal_nut/test \
    --gen_root ./Result_MetalNut_Test \
    --output analysis_result.png
```










