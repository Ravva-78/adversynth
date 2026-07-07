"""
Phase 3 — Deep Convolutional GAN (DCGAN).
Radford et al. (2015) architecture adapted for 64x64 fundus images.

Key upgrades over Vanilla GAN:
- Transposed convolutions in G (learnable upsampling)
- Strided convolutions in D (learnable downsampling)
- BatchNorm everywhere except G output and D input
- LeakyReLU in D, ReLU in G, Tanh at G output
"""
import torch
import torch.nn as nn

LATENT_DIM   = 100
FEATURES_G   = 64    # base channel multiplier for Generator
FEATURES_D   = 64    # base channel multiplier for Discriminator
IMAGE_CHANNELS = 3


class Generator(nn.Module):
    """
    z (100) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 RGB image.
    Each block: ConvTranspose2d → BatchNorm → ReLU.
    """
    def __init__(self, latent_dim=LATENT_DIM, features_g=FEATURES_G, channels=IMAGE_CHANNELS):
        super().__init__()
        fg = features_g
        self.net = nn.Sequential(
            # Input: z (latent_dim x 1 x 1)
            self._block(latent_dim, fg * 16, 4, 1, 0),   # → fg*16 x 4 x 4
            self._block(fg * 16,   fg * 8,  4, 2, 1),    # → fg*8  x 8 x 8
            self._block(fg * 8,    fg * 4,  4, 2, 1),    # → fg*4  x 16 x 16
            self._block(fg * 4,    fg * 2,  4, 2, 1),    # → fg*2  x 32 x 32
            # Final layer: no BatchNorm, Tanh activation
            nn.ConvTranspose2d(fg * 2, channels, 4, 2, 1, bias=False),  # → 3 x 64 x 64
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    64×64 image → scalar probability.
    Each block: Conv2d (stride=2) → BatchNorm → LeakyReLU.
    No BatchNorm on first layer (standard DCGAN practice).
    """
    def __init__(self, features_d=FEATURES_D, channels=IMAGE_CHANNELS):
        super().__init__()
        fd = features_d
        self.net = nn.Sequential(
            # First layer: no BatchNorm
            nn.Conv2d(channels, fd, 4, 2, 1, bias=False),  # → fd x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),

            self._block(fd,     fd * 2,  4, 2, 1),  # → fd*2  x 16 x 16
            self._block(fd * 2, fd * 4,  4, 2, 1),  # → fd*4  x 8  x 8
            self._block(fd * 4, fd * 8,  4, 2, 1),  # → fd*8  x 4  x 4

            nn.Conv2d(fd * 8, 1, 4, 1, 0, bias=False),     # → 1 x 1 x 1
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.net(x).view(-1, 1)


if __name__ == "__main__":
    G = Generator()
    D = Discriminator()
    z    = torch.randn(4, LATENT_DIM, 1, 1)
    fake = G(z)
    pred = D(fake)
    print(f"Generator output:      {fake.shape}")   # (4, 3, 64, 64)
    print(f"Discriminator output:  {pred.shape}")   # (4, 1)
    print("DCGAN model check passed.")
