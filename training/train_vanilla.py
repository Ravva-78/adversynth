"""
Phase 2 — Train Vanilla GAN.
Run: python training/train_vanilla.py --config configs/vanilla_gan.yaml
"""
import argparse, yaml, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path

from models.vanilla_gan import Generator, Discriminator, LATENT_DIM
from data.dataset import APTOSDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir  = Path("outputs/vanilla_gan"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/vanilla_gan"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset    = APTOSDataset(split="train", image_size=cfg["image_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=2, pin_memory=True)

    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    # Fixed noise to track generator progress across epochs
    fixed_noise = torch.randn(64, LATENT_DIM).to(device)

    for epoch in range(cfg["epochs"]):
        G.train(); D.train()
        d_losses, g_losses = [], []

        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            imgs = imgs.to(device)
            B    = imgs.size(0)

            real_labels = torch.ones(B, 1).to(device)  * 0.9  # label smoothing
            fake_labels = torch.zeros(B, 1).to(device)

            # ── Train Discriminator ──
            opt_D.zero_grad()
            z         = torch.randn(B, LATENT_DIM).to(device)
            fake_imgs = G(z).detach()
            d_real    = D(imgs)
            d_fake    = D(fake_imgs)
            d_loss    = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_loss.backward()
            opt_D.step()

            # ── Train Generator ──
            opt_G.zero_grad()
            z         = torch.randn(B, LATENT_DIM).to(device)
            fake_imgs = G(z)
            g_loss    = criterion(D(fake_imgs), torch.ones(B, 1).to(device))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f"  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

        # Save sample grid every 5 epochs
        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
            save_image(samples, out_dir / f"epoch_{epoch+1:03d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))
            torch.save(G.state_dict(), ckpt_dir / f"G_epoch_{epoch+1:03d}.pt")
            torch.save(D.state_dict(), ckpt_dir / f"D_epoch_{epoch+1:03d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vanilla_gan.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)
