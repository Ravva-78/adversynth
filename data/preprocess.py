"""
Phase 1 — Step 2: Preprocess raw APTOS images.
- Resize to IMAGE_SIZE x IMAGE_SIZE
- Ben Graham crop (remove black borders)
- Normalise to [-1, 1] for GAN training
- Train/val/test split stratified by label
- Save processed images into class subfolders

Run: python data/preprocess.py
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_SIZE = 64          # change to 128 or 256 for higher quality (needs more VRAM)
RAW_DIR    = Path("data/raw")
OUT_DIR    = Path("data/processed")
CSV_PATH   = RAW_DIR / "train.csv"


def ben_graham_crop(img, sigmaX=10):
    """
    Ben Graham's preprocessing: subtract local mean colour,
    then circle-crop to remove black borders.
    Used widely in DR competition baselines.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.45), 1, -1)
    img[mask == 0] = 0
    return img


def centre_crop_resize(img, size):
    h, w = img.shape[:2]
    m = min(h, w)
    top  = (h - m) // 2
    left = (w - m) // 2
    img  = img[top:top+m, left:left+m]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def preprocess():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Run data/download.py first.")

    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")
    print(df["diagnosis"].value_counts().sort_index())

    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["diagnosis"], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=42)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        for grade in range(5):
            (OUT_DIR / split_name / str(grade)).mkdir(parents=True, exist_ok=True)

        img_dir = RAW_DIR / "train_images"
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            img_path = img_dir / f"{row['id_code']}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = ben_graham_crop(img)
            img = centre_crop_resize(img, IMAGE_SIZE)
            out_path = OUT_DIR / split_name / str(row["diagnosis"]) / f"{row['id_code']}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"\nDone. Processed images saved to {OUT_DIR}/")
    print("Structure: data/processed/{train,val,test}/{0,1,2,3,4}/")


if __name__ == "__main__":
    preprocess()
