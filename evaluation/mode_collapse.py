"""
Detect mode collapse by tracking intra-batch feature variance
at the discriminator's penultimate layer.

A collapsing generator produces near-identical images → low variance.
Run during or after training: python evaluation/mode_collapse.py --phase cgan
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

from models.dcgan  import Generator as DCGen,  Discriminator as DCDis,  LATENT_DIM as DC_Z
from models.cgan   import Generator as CGANGen, Discriminator as CGANDis, LATENT_DIM as CG_Z, N_CLASSES
from data.dataset  import APTOSDataset


def hook_penultimate(module, input, output, store):
    store.append(output.detach().cpu())


def measure_variance(G, D, latent_dim, device, n_batches=20, batch_size=32,
                     labels=None, is_cgan=False):
    G.eval(); D.eval()
    variances = []
    store     = []

    # Hook into second-to-last conv layer of D
    target_layer = list(D.net.children())[-3]
    handle = target_layer.register_forward_hook(
        lambda m, i, o: store.append(o.detach().cpu())
    )

    with torch.no_grad():
        for _ in range(n_batches):
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            if is_cgan:
                lbl  = torch.randint(0, N_CLASSES, (batch_size,), device=device)
                fake = G(z, lbl)
                D(fake, lbl)
            else:
                fake = G(z)
                D(fake)

            if store:
                feat = store[-1].view(batch_size, -1)
                var  = feat.var(dim=0).mean().item()
                variances.append(var)
                store.clear()

    handle.remove()
    return np.array(variances)


def plot_variance(phase="cgan"):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

    ckpt_dir = Path(f"checkpoints/{phase}")
    ckpt_G   = sorted(ckpt_dir.glob("G_epoch_*.pt"))
    ckpt_D   = sorted(ckpt_dir.glob("D_epoch_*.pt"))

    if not ckpt_G:
        print(f"No checkpoints found in {ckpt_dir}"); return

    epochs    = []
    mean_vars = []

    for g_path, d_path in zip(ckpt_G, ckpt_D):
        epoch_num = int(g_path.stem.split("_")[-1])

        if phase == "cgan":
            G = CGANGen(latent_dim=CG_Z, n_classes=N_CLASSES).to(device)
            D = CGANDis(n_classes=N_CLASSES).to(device)
            G.load_state_dict(torch.load(g_path, map_location=device))
            D.load_state_dict(torch.load(d_path, map_location=device))
            vars_ = measure_variance(G, D, CG_Z, device, is_cgan=True)
        else:
            G = DCGen(latent_dim=DC_Z).to(device)
            D = DCDis().to(device)
            G.load_state_dict(torch.load(g_path, map_location=device))
            D.load_state_dict(torch.load(d_path, map_location=device))
            vars_ = measure_variance(G, D, DC_Z, device, is_cgan=False)

        epochs.append(epoch_num)
        mean_vars.append(vars_.mean())
        print(f"Epoch {epoch_num:3d} — mean feature variance: {vars_.mean():.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, mean_vars, "o-", color="#378ADD", lw=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Feature Variance", fontsize=11)
    ax.set_title(f"Mode Collapse Monitor — {phase.upper()}\n"
                 "Increasing variance = healthy diversity. Flat/zero = mode collapse.",
                 fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"mode_collapse_{phase}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="cgan",
                        choices=["vanilla_gan", "dcgan", "cgan"])
    args = parser.parse_args()
    plot_variance(args.phase)
