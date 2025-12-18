## MVTec-Shift: A Synthetic Benchmark for Industrial Anomaly Detection under Environmental Domain Shifts

This project is for "Assignment# 2: Dataset Creation and Benchmarking" in Introduction_to_computer_vision 2025-2


Task: Each student is required to make a comprehensive plan of a new dataset, including its title, main description of the dataset, number of classes, images count and a table listing existing datasets with their key information and details of the new dataset, showing how the new dataset is different and unique etc. You can include all possible details. I will explain via example in the class or using a separate video. The evaluation will be done based on these points:

#### [[Report]](https://drive.google.com/file/d/1w7fHMe6SER9b6y6vDmaps3oFbhORqZkY/view?usp=drive_link) 

-Dataset Understanding & Problem Analysis (5 points)

-Novel Dataset / Extension Proposal (5 points)

-Practical Demonstration/Feasibility (5 points)

---Option A (Coding track): run a prototype experiment, test a model, do possible ablation and report results.
    
---Option B (Non-coding track): provide a feasibility analysis, workflow diagram, collection plan, and manual annotation sample for the dataset you are proposing.
    
---Option C: Create a small/mini dataset (what is proposed in point 2)

# 1. Project Overview
MVTec-Shift is a novel synthetic dataset designed to benchmark the robustness of unsupervised anomaly detection models against environmental domain shifts.

Existing datasets like MVTec AD  are captured under controlled studio lighting. However, real-world industrial environments often suffer from low light, grease, noise, and oxidation (rust). Standard models trained on clean data often fail in these conditions, misclassifying functional but environmentally affected parts as defects.

This project utilizes Generative AI (Stable Diffusion - InstructPix2Pix) to transform the environmental "mood" of the MVTec AD Metal Nut category while preserving structural integrity. This allows for the evaluation of anomaly detection performance under realistic factory conditions.

# 📂2. Dataset Structure
```
MVTec-Shift/
├── train/
│   └── good/              # Original Training Data
├── test/
│   ├── bent/
│   ├── color/
│   ├── flip/
│   └── scratch/           # Original Defect Data
└── generated_data/        # [NEW] Synthetic Data
    ├── dark/              # Low-light condition
    ├── greasy/            # Oil stains & specular highlights
    ├── rusty/             # Oxidation & weathering
    └── noisy/             # Sensor noise & dust
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




