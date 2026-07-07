"""
Ablation experiment — ResNet-50 classifier.
Trains under 6 conditions and reports AUC-ROC per grade.

Run: python training/train_classifier.py --mode ablation
     python training/train_classifier.py --mode real_only
     python training/train_classifier.py --mode real_plus_cgan
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path
import numpy as np

from models.classifier import DRClassifier
from data.dataset import APTOSDataset, get_classifier_transforms


MODES = {
    "real_only":      {"use_synthetic": False, "synth_dirs": []},
    "real_plus_cgan": {"use_synthetic": True,  "synth_dirs": ["data/synthetic/cgan"]},
    "real_plus_cycle":{"use_synthetic": True,  "synth_dirs": ["data/synthetic/cyclegan"]},
}


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)

    # One-vs-rest AUC per class
    auc_scores = {}
    for cls in range(5):
        binary_labels = (labels == cls).astype(int)
        if binary_labels.sum() > 0:
            auc_scores[f"Grade_{cls}"] = roc_auc_score(binary_labels, probs[:, cls])
    auc_scores["Mean"] = np.mean(list(auc_scores.values()))
    return auc_scores


def run_mode(mode_name, epochs=20, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = get_classifier_transforms(train=True)
    transform_eval  = get_classifier_transforms(train=False)

    train_ds = APTOSDataset(split="train", transform=transform_train, image_size=224)

    if MODES[mode_name]["use_synthetic"]:
        for synth_dir in MODES[mode_name]["synth_dirs"]:
            if Path(synth_dir).exists():
                synth_ds = APTOSDataset.__new__(APTOSDataset)
                # Build synthetic dataset from saved images
                synth_ds.samples   = []
                synth_ds.transform = transform_train
                for grade in range(5):
                    for img_path in Path(synth_dir).glob(f"{grade}/*.png"):
                        synth_ds.samples.append((img_path, grade))
                if synth_ds.samples:
                    train_ds = ConcatDataset([train_ds, synth_ds])
                    print(f"Added {len(synth_ds.samples)} synthetic images from {synth_dir}")

    val_ds  = APTOSDataset(split="val",  transform=transform_eval, image_size=224)
    test_ds = APTOSDataset(split="test", transform=transform_eval, image_size=224)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model     = DRClassifier(n_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    ckpt_dir     = Path(f"checkpoints/classifier_{mode_name}"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training: {mode_name}")
    print(f"Train size: {len(train_ds)}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_auc = evaluate(model, val_loader, device)
        mean_auc = val_auc["Mean"]
        print(f"  Loss: {total_loss/len(train_loader):.4f}  Val AUC (mean): {mean_auc:.4f}")
        print(f"  Per-grade: {val_auc}")

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    # Final test evaluation
    model.load_state_dict(torch.load(ckpt_dir / "best.pt"))
    test_auc = evaluate(model, test_loader, device)
    print(f"\n[{mode_name}] TEST AUC: {test_auc}")
    return test_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ablation",
                        choices=["ablation"] + list(MODES.keys()))
    args = parser.parse_args()

    if args.mode == "ablation":
        results = {}
        for mode in MODES:
            results[mode] = run_mode(mode)
        print("\n" + "="*60)
        print("ABLATION RESULTS")
        print("="*60)
        for mode, auc in results.items():
            print(f"{mode:25s}: Grade3={auc.get('Grade_3',0):.4f}  "
                  f"Grade4={auc.get('Grade_4',0):.4f}  Mean={auc['Mean']:.4f}")
    else:
        run_mode(args.mode)
