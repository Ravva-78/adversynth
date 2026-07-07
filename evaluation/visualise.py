"""
Visualise real vs synthetic images side by side.
Run: python evaluation/visualise.py --phase cgan --n 32
"""
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def load_images_from_dir(directory, n, image_size=64):
    paths = list(Path(directory).rglob("*.png"))
    if not paths:
        return None
    paths = random.sample(paths, min(n, len(paths)))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs)


def make_grid_plot(phase="cgan", n_per_class=4, image_size=64):
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Real vs Synthetic — {phase.upper()}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(5, n_per_class * 2 + 2, figure=fig,
                           wspace=0.05, hspace=0.3)

    for grade in range(5):
        real_dir  = Path(f"data/processed/train/{grade}")
        synth_dir = Path(f"data/synthetic/{phase}/{grade}")

        real_imgs  = load_images_from_dir(real_dir,  n_per_class, image_size)
        synth_imgs = load_images_from_dir(synth_dir, n_per_class, image_size)

        # Grade label
        ax_label = fig.add_subplot(gs[grade, 0])
        ax_label.text(0.5, 0.5, f"Grade {grade}\n{GRADE_NAMES[grade]}",
                      ha="center", va="center", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#E6F1FB"))
        ax_label.axis("off")

        # Real images
        for i in range(n_per_class):
            ax = fig.add_subplot(gs[grade, i + 1])
            if real_imgs is not None and i < len(real_imgs):
                ax.imshow(real_imgs[i].permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
            if grade == 0:
                ax.set_title("Real" if i == n_per_class // 2 else "", fontsize=8)

        # Divider
        ax_div = fig.add_subplot(gs[grade, n_per_class + 1])
        ax_div.axvline(0.5, color="#cccccc", linewidth=1)
        ax_div.axis("off")

        # Synthetic images
        for i in range(n_per_class):
            ax = fig.add_subplot(gs[grade, n_per_class + 2 + i])
            if synth_imgs is not None and i < len(synth_imgs):
                ax.imshow(synth_imgs[i].permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
            if grade == 0:
                ax.set_title("Synthetic" if i == n_per_class // 2 else "", fontsize=8)

    save_path = out_dir / f"real_vs_synth_{phase}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="cgan",
                        choices=["vanilla_gan", "dcgan", "cgan", "cyclegan"])
    parser.add_argument("--n",     default=4, type=int, help="Images per class per column")
    args = parser.parse_args()
    make_grid_plot(phase=args.phase, n_per_class=args.n)
