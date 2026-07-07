"""
Phase 2 — Vanilla GAN (Fully Connected).
Baseline model to confirm adversarial convergence.
Images: 64x64x3 flattened to 12288-dim vectors.
"""
import torch
import torch.nn as nn

IMAGE_SIZE = 64
IMG_DIM    = IMAGE_SIZE * IMAGE_SIZE * 3   # 12288
LATENT_DIM = 100


class Generator(nn.Module):
    """
    Noise z (100-dim) → fake image (12288-dim → reshaped to 3×64×64).
    Uses BatchNorm + ReLU hidden layers, Tanh output (range [-1,1]).
    """
    def __init__(self, latent_dim=LATENT_DIM, img_dim=IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)


class Discriminator(nn.Module):
    """
    Image (12288-dim) → real/fake probability (scalar).
    Uses LeakyReLU (no BatchNorm in D — standard practice).
    """
    def __init__(self, img_dim=IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def weights_init(model):
    """Apply to any GAN model after instantiation."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    G = Generator()
    D = Discriminator()
    z = torch.randn(4, LATENT_DIM)
    fake = G(z)
    score = D(fake)
    print(f"Generator output: {fake.shape}")    # (4, 3, 64, 64)
    print(f"Discriminator output: {score.shape}") # (4, 1)
    print("Vanilla GAN model check passed.")
