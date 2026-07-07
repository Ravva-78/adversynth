"""
Plot AUC-ROC curves for all ablation conditions.
Run after training/train_classifier.py --mode ablation
Run: python evaluation/auc_roc.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from pathlib import Path

from models.classifier import DRClassifier
from data.dataset import APTOSDataset, get_classifier_transforms

GRADE_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
ABLATION_MODES = ["real_only", "real_plus_cgan", "real_plus_cycle"]
COLORS         = ["#378ADD", "#1D9E75", "#D85A30"]


def get_probs_and_labels(ckpt_path, device):
    model = DRClassifier(n_classes=5).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ds     = APTOSDataset(split="test", transform=get_classifier_transforms(train=False), image_size=224)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


def plot_roc(target_grade=3):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"AUC-ROC: Grade {target_grade} ({GRADE_NAMES[target_grade]}) Detection",
                 fontsize=13, fontweight="bold")

    for ax_idx, grade in enumerate([target_grade, 4]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

        for mode, color in zip(ABLATION_MODES, COLORS):
            ckpt = Path(f"checkpoints/classifier_{mode}/best.pt")
            if not ckpt.exists():
                print(f"Checkpoint not found: {ckpt} — skipping")
                continue
            probs, labels = get_probs_and_labels(ckpt, device)
            binary = (labels == grade).astype(int)
            fpr, tpr, _ = roc_curve(binary, probs[:, grade])
            roc_auc     = auc(fpr, tpr)
            label       = mode.replace("_", " ").title()
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={roc_auc:.3f})")

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"Grade {grade} — {GRADE_NAMES[grade]}", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = out_dir / f"auc_roc_grade{target_grade}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_roc(target_grade=3)
    plot_roc(target_grade=4)
