"""
Phase 5 — Train CycleGAN (Healthy ↔ Severe DR).
Run: python training/train_cyclegan.py --config configs/cyclegan.yaml
"""
import argparse, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
import itertools

from models.cyclegan import Generator, Discriminator, ImageBuffer
from data.dataset import UnpairedDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_c = cfg.get("lambda_cycle", 10.0)
    lambda_i = cfg.get("lambda_identity", 0.5)

    out_dir  = Path("outputs/cyclegan"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/cyclegan"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnpairedDataset(
        domain_a_grade=cfg.get("domain_a_grade", 0),
        domain_b_grade=cfg.get("domain_b_grade", 3),
        split="train",
        image_size=cfg["image_size"]
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    G_AB = Generator().to(device)   # Healthy → Severe DR
    G_BA = Generator().to(device)   # Severe DR → Healthy
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)

    buf_A = ImageBuffer()
    buf_B = ImageBuffer()

    criterion_gan   = nn.MSELoss()   # LSGAN — more stable than BCE
    criterion_cycle = nn.L1Loss()
    criterion_idt   = nn.L1Loss()

    opt_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=cfg["lr"], betas=(0.5, 0.999)
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    for epoch in range(cfg["epochs"]):
        g_losses, d_losses = [], []

        for x_a, x_b in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            B   = x_a.size(0)
            real_label = torch.ones(B,  *D_A(x_a).shape[1:], device=device)
            fake_label = torch.zeros(B, *D_A(x_a).shape[1:], device=device)

            # ── Train Generators G_AB and G_BA ──
            opt_G.zero_grad()

            fake_b = G_AB(x_a)
            fake_a = G_BA(x_b)
            rec_a  = G_BA(fake_b)
            rec_b  = G_AB(fake_a)

            # Adversarial
            loss_adv = (criterion_gan(D_B(fake_b), real_label) +
                        criterion_gan(D_A(fake_a), real_label))
            # Cycle consistency
            loss_cyc = lambda_c * (criterion_cycle(rec_a, x_a) + criterion_cycle(rec_b, x_b))
            # Identity
            loss_idt = lambda_i * lambda_c * (criterion_idt(G_BA(x_a), x_a) +
                                               criterion_idt(G_AB(x_b), x_b))

            loss_G = loss_adv + loss_cyc + loss_idt
            loss_G.backward()
            opt_G.step()

            # ── Train Discriminator D_A ──
            opt_D_A.zero_grad()
            fake_a_buf = buf_A.push_and_pop(fake_a.detach())
            loss_D_A   = 0.5 * (criterion_gan(D_A(x_a), real_label) +
                                 criterion_gan(D_A(fake_a_buf), fake_label))
            loss_D_A.backward()
            opt_D_A.step()

            # ── Train Discriminator D_B ──
            opt_D_B.zero_grad()
            fake_b_buf = buf_B.push_and_pop(fake_b.detach())
            loss_D_B   = 0.5 * (criterion_gan(D_B(x_b), real_label) +
                                 criterion_gan(D_B(fake_b_buf), fake_label))
            loss_D_B.backward()
            opt_D_B.step()

            g_losses.append(loss_G.item())
            d_losses.append((loss_D_A + loss_D_B).item())

        print(f"  G_loss: {sum(g_losses)/len(g_losses):.4f}  "
              f"D_loss: {sum(d_losses)/len(d_losses):.4f}  "
              f"Cycle: {loss_cyc.item():.4f}")

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_a  = x_a[:4]
                sample_b  = x_b[:4]
                trans_b   = G_AB(sample_a)
                trans_a   = G_BA(sample_b)
                grid = torch.cat([sample_a, trans_b, sample_b, trans_a], dim=0)
            save_image(grid.cpu(), out_dir / f"epoch_{epoch+1:03d}.png",
                       nrow=4, normalize=True, value_range=(-1, 1))
            torch.save(G_AB.state_dict(), ckpt_dir / f"G_AB_epoch_{epoch+1:03d}.pt")
            torch.save(G_BA.state_dict(), ckpt_dir / f"G_BA_epoch_{epoch+1:03d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cyclegan.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
