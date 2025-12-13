import os
import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 색상 팔레트용 (없으면 pip install seaborn)
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

def extract_features(path, model, processor, device, label=""):
    # jpg, png, jpeg 등 다양한 확장자 지원
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(path, ext)))
    
    if not files: 
        print(f"[Warning] No images found in: {path}")
        return None

    features = []
    print(f"[Info] Extracting features for '{label}' ({len(files)} images)...")

    # tqdm으로 진행률 표시
    for f in tqdm(files, unit="img"):
        try:
            image = Image.open(f).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                # CLIP 이미지 임베딩 추출
                feat = model.get_image_features(**inputs).squeeze().cpu().numpy()
                features.append(feat)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    return np.array(features) if features else None

def main():
    parser = argparse.ArgumentParser(description="Visualize CLIP features using PCA based on Original data.")
    
    # [설정] 앞서 작업한 경로들을 기본값으로 지정
    parser.add_argument('--original_dir', type=str, default="./mvtec_ad/metal_nut/train/good", 
                        help="Path to original (normal) images used for PCA basis")
    parser.add_argument('--generated_root', type=str, default="./Result_MetalNut", 
                        help="Root directory containing generated mood subfolders (e.g., dark, rusty)")
    parser.add_argument('--output', default="pca_analysis_result.png", help="Output filename")
    
    args = parser.parse_args()

    # Device & Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    print(f">>> Loading CLIP Model ({model_id}) on {device}...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    data_map = {}

    # 1. Load Original Data (Basis)
    if os.path.exists(args.original_dir):
        feats = extract_features(args.original_dir, model, processor, device, label="Original (Train)")
        if feats is not None:
            data_map["Original"] = feats
    else:
        print(f"[Error] Original directory not found: {args.original_dir}")
        return

    # 2. Load Generated Data (Automatically scan subfolders)
    if os.path.exists(args.generated_root):
        subfolders = sorted([f.path for f in os.scandir(args.generated_root) if f.is_dir()])
        
        for folder in subfolders:
            mood_name = os.path.basename(folder) # 폴더명(dark, rusty 등)을 라벨로 사용
            feats = extract_features(folder, model, processor, device, label=mood_name)
            if feats is not None:
                data_map[mood_name] = feats
    else:
        print(f"[Warning] Generated root directory not found: {args.generated_root}")

    # 3. PCA Analysis
    if "Original" in data_map and len(data_map["Original"]) > 1:
        print("\n[Info] Fitting PCA on 'Original' data...")
        pca = PCA(n_components=2)
        pca.fit(data_map["Original"]) # 기준점은 항상 Original
        
        # Plotting
        plt.figure(figsize=(12, 10))
        
        # 색상 팔레트 자동 생성
        unique_labels = list(data_map.keys())
        colors = sns.color_palette("hls", len(unique_labels)) # 예쁜 색상 자동 배정
        
        # Original을 맨 먼저 그리기 (파란색 고정 추천)
        # 나머지는 자동 색상
        
        for i, label in enumerate(unique_labels):
            feats = data_map[label]
            proj = pca.transform(feats) # 투영
            
            # 스타일 설정
            if label == "Original":
                c = 'black' # 원본은 검정/진한 색으로 강조
                m = 'o'
                alpha = 0.3
                zorder = 0 # 맨 뒤에 깔기
                label_name = "Original (Base)"
            else:
                c = colors[i]
                m = 'x' # 생성된건 X 마크
                alpha = 0.8
                zorder = 10 # 앞으로 띄우기
                label_name = f"Gen: {label}"

            plt.scatter(proj[:, 0], proj[:, 1], 
                        c=[c], marker=m, alpha=alpha, label=label_name, s=40, zorder=zorder)

        plt.title(f"CLIP Feature Space Analysis (PCA Basis: Original Normal)", fontsize=15)
        plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2%})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 범례를 그래프 밖으로
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(args.output, dpi=300)
        print(f"\n[Success] Analysis saved to {args.output}")
        print("Check the plot to see how far the generated moods shifted from the original distribution!")
        
    else:
        print("[Error] Not enough original data to fit PCA.")

if __name__ == "__main__":
    main()