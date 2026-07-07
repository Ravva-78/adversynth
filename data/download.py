"""
Phase 1 — Step 1: Download APTOS 2019 dataset from Kaggle.
Run: python data/download.py
"""
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise ValueError(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY in .env\n"
            "Get your token at: https://www.kaggle.com/settings -> API"
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    print("Downloading APTOS 2019 dataset...")
    os.system(
        f"kaggle competitions download -c aptos2019-blindness-detection -p {RAW_DIR}"
    )

    zip_path = RAW_DIR / "aptos2019-blindness-detection.zip"
    if zip_path.exists():
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        print(f"Done. Images saved to {RAW_DIR}/")
    else:
        print("Download may have failed. Check your Kaggle credentials.")

if __name__ == "__main__":
    download()
