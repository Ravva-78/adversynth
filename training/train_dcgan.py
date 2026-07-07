"""
Phase 3 — Train DCGAN.
Run: python training/train_dcgan.py --config configs/dcgan.yaml
"""
import argparse, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path

from models.dcgan import Generator, Discriminator, LATENT_DIM
from data.dataset import APTOSDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir  = Path("outputs/dcgan"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/dcgan"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset    = APTOSDataset(split="train", image_size=cfg["image_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    G = Generator(latent_dim=LATENT_DIM).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1).to(device)

    for epoch in range(cfg["epochs"]):
        G.train(); D.train()
        d_losses, g_losses = [], []

        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            imgs = imgs.to(device)
            B    = imgs.size(0)

            real_labels = torch.full((B, 1), 0.9, device=device)   # label smoothing
            fake_labels = torch.zeros(B, 1, device=device)

            # ── Train Discriminator ──
            opt_D.zero_grad()
            z         = torch.randn(B, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z).detach()
            d_loss    = criterion(D(imgs), real_labels) + criterion(D(fake_imgs), fake_labels)
            d_loss.backward()
            opt_D.step()

            # ── Train Generator ──
            opt_G.zero_grad()
            z         = torch.randn(B, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z)
            g_loss    = criterion(D(fake_imgs), torch.ones(B, 1, device=device))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        avg_d = sum(d_losses) / len(d_losses)
        avg_g = sum(g_losses) / len(g_losses)
        print(f"  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

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
    parser.add_argument("--config", default="configs/dcgan.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
