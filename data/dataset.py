"""
PyTorch Dataset classes for all training phases.

APTOSDataset     — standard labeled dataset (all phases)
UnpairedDataset  — two unpaired domains for CycleGAN (Phase 5)
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_gan_transforms(image_size=64):
    """Normalise to [-1, 1] as required by Tanh output in Generator."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(image_size=64):
    """No augmentation for val/test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_classifier_transforms(image_size=224, train=True):
    """ImageNet stats for ResNet-50 fine-tuning."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Datasets ────────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    Loads processed images from data/processed/{split}/{grade}/*.png
    Returns (image_tensor, label) pairs.

    Usage:
        ds = APTOSDataset(split='train', grades=[0,1,2,3,4])
        img, label = ds[0]
    """
    def __init__(self, split="train", grades=None, transform=None, image_size=64):
        self.root = Path("data/processed") / split
        self.transform = transform or get_gan_transforms(image_size)
        grades = grades if grades is not None else [0, 1, 2, 3, 4]

        self.samples = []
        for grade in grades:
            grade_dir = self.root / str(grade)
            if grade_dir.exists():
                for img_path in grade_dir.glob("*.png"):
                    self.samples.append((img_path, grade))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found at {self.root}. "
                "Run data/preprocess.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)

    def class_counts(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return dict(sorted(counts.items()))


class UnpairedDataset(Dataset):
    """
    Loads two unpaired domains for CycleGAN training.
    Domain A = healthy (Grade 0), Domain B = severe DR (Grade 3 or 4).

    Usage:
        ds = UnpairedDataset(domain_a_grade=0, domain_b_grade=3)
        img_a, img_b = ds[0]
    """
    def __init__(self, domain_a_grade=0, domain_b_grade=3, split="train",
                 transform=None, image_size=64):
        self.transform = transform or get_gan_transforms(image_size)
        root = Path("data/processed") / split

        self.domain_a = list((root / str(domain_a_grade)).glob("*.png"))
        self.domain_b = list((root / str(domain_b_grade)).glob("*.png"))

        if not self.domain_a or not self.domain_b:
            raise RuntimeError("Domain images not found. Run data/preprocess.py.")

    def __len__(self):
        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, idx):
        img_a = Image.open(self.domain_a[idx % len(self.domain_a)]).convert("RGB")
        img_b = Image.open(self.domain_b[idx % len(self.domain_b)]).convert("RGB")
        return self.transform(img_a), self.transform(img_b)
