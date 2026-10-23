"""
Compute FID score between real and generated images.
Run: python evaluation/fid_score.py --real data/processed/train --fake data/synthetic/cgan
"""
import argparse
import subprocess
from pathlib import Path
import torch


def compute_fid(real_dir, fake_dir, batch_size=50, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    Uses pytorch-fid package.
    Expects real_dir and fake_dir to contain images directly (not in subfolders).
    Creates a flat copy if needed.
    """
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)

    # pytorch-fid expects flat directories — flatten if class-structured
    flat_real = Path("outputs/fid_tmp/real")
    flat_fake = Path("outputs/fid_tmp/fake")
    flat_real.mkdir(parents=True, exist_ok=True)
    flat_fake.mkdir(parents=True, exist_ok=True)

    import shutil
    for img in real_path.rglob("*.png"):
        shutil.copy(img, flat_real / img.name)
    for img in fake_path.rglob("*.png"):
        shutil.copy(img, flat_fake / img.name)

                
    cmd = [
        "python", "-m", "pytorch_fid",
        str(flat_real), str(flat_fake),
        "--batch-size", str(batch_size),
        "--device", device
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",   default="data/processed/train")
    parser.add_argument("--fake",   default="data/synthetic/cgan")
    parser.add_argument("--batch",  default=50, type=int)
    parser.add_argument("--device", default=None, help="Device to run on ('cuda' or 'cpu')")
    args = parser.parse_args()
    compute_fid(args.real, args.fake, args.batch, args.device)
