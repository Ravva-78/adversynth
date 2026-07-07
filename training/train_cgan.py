"""
Phase 4 — Train Conditional GAN.
Oversample Grade 3 & 4 to address class imbalance.
Run: python training/train_cgan.py --config configs/cgan.yaml
"""
import argparse, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import wandb

from models.cgan import Generator, Discriminator, LATENT_DIM, N_CLASSES
from data.dataset import APTOSDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_weighted_sampler(dataset):
    """Inverse-frequency sampling — rare classes sampled more often."""
    labels  = [label for _, label in dataset.samples]
    counts  = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def generate_and_save(G, n_per_class, latent_dim, device, out_dir):
    """Generate n_per_class images for each DR grade and save to disk."""
    G.eval()
    synth_dir = Path("data/synthetic/cgan")
    synth_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for grade in range(N_CLASSES):
            grade_dir = synth_dir / str(grade)
            grade_dir.mkdir(exist_ok=True)
            z      = torch.randn(n_per_class, latent_dim, 1, 1, device=device)
            labels = torch.full((n_per_class,), grade, dtype=torch.long, device=device)
            imgs   = G(z, labels).cpu()
            for i, img in enumerate(imgs):
                save_image(img, grade_dir / f"synth_{i:04d}.png",
                           normalize=True, value_range=(-1, 1))

    print(f"Synthetic images saved to data/synthetic/cgan/")


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Weights & Biases tracking
    wandb.init(project="AdverSynth", config=cfg, name="cgan_training")

    out_dir  = Path("outputs/cgan"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/cgan"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset  = APTOSDataset(split="train", image_size=cfg["image_size"])
    sampler  = get_weighted_sampler(dataset)
    loader   = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)

    print("Class distribution:", dataset.class_counts())

    G = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES).to(device)
    D = Discriminator(n_classes=N_CLASSES, image_size=cfg["image_size"]).to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    # Fixed samples — one noise per class for visual progress tracking
    fixed_z = torch.randn(N_CLASSES * 8, LATENT_DIM, 1, 1, device=device)
    fixed_labels = torch.arange(N_CLASSES, device=device).repeat(8)

    for epoch in range(cfg["epochs"]):
        G.train(); D.train()
        d_losses, g_losses = [], []

        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            B      = imgs.size(0)

            real_labels_soft = torch.full((B, 1), 0.9, device=device)
            fake_labels_zero = torch.zeros(B, 1, device=device)

            # ── Train Discriminator ──
            opt_D.zero_grad()
            z         = torch.randn(B, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z, labels).detach()
            d_loss    = (criterion(D(imgs, labels), real_labels_soft) +
                         criterion(D(fake_imgs, labels), fake_labels_zero))
            d_loss.backward()
            opt_D.step()

            # ── Train Generator ──
            opt_G.zero_grad()
            z         = torch.randn(B, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z, labels)
            g_loss    = criterion(D(fake_imgs, labels), torch.ones(B, 1, device=device))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f"  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")
        
        # Log metrics to WandB
        wandb.log({"D_loss": avg_d, "G_loss": avg_g, "epoch": epoch + 1})

        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z, fixed_labels).cpu()
            
            sample_path = out_dir / f"epoch_{epoch+1:03d}.png"
            save_image(samples, sample_path,
                       nrow=8, normalize=True, value_range=(-1, 1))
            
            # Log generated image grid to WandB
            wandb.log({"Generated Samples": wandb.Image(str(sample_path)), "epoch": epoch + 1})
            torch.save(G.state_dict(), ckpt_dir / f"G_epoch_{epoch+1:03d}.pt")
            torch.save(D.state_dict(), ckpt_dir / f"D_epoch_{epoch+1:03d}.pt")

    # After training — generate synthetic dataset
    generate_and_save(G, n_per_class=cfg.get("synth_per_class", 500),
                      latent_dim=LATENT_DIM, device=device, out_dir=out_dir)
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cgan.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
