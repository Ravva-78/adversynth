"""
Phase 5 — CycleGAN (Zhu et al., 2017).
Unpaired image-to-image translation: Healthy (Grade 0) ↔ Severe DR (Grade 3).

Components:
  G_AB : Healthy   → Severe DR
  G_BA : Severe DR → Healthy
  D_A  : Discriminator for domain A (Healthy)
  D_B  : Discriminator for domain B (Severe DR)

Losses:
  Adversarial : G_AB fools D_B; G_BA fools D_A
  Cycle       : G_BA(G_AB(x_A)) ≈ x_A  (and vice versa)  [lambda=10]
  Identity    : G_AB(x_B) ≈ x_B  [lambda_id=0.5]
"""
import torch
import torch.nn as nn


# ── Residual Block ──────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ── Generator ───────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    ResNet-based generator.
    Encoder → 9 Residual Blocks → Decoder.
    Uses InstanceNorm (better for style transfer than BatchNorm).
    Uses ReflectionPad to reduce border artefacts.
    """
    def __init__(self, channels=3, features=64, n_res_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, features, 7, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            # Downsample
            nn.Conv2d(features,     features * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True),
        ]
        # Residual blocks
        for _ in range(n_res_blocks):
            layers.append(ResBlock(features * 4))
        # Upsample
        layers += [
            nn.ConvTranspose2d(features * 4, features * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 2, features, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, channels, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.net(x)


# ── PatchGAN Discriminator ──────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    70×70 PatchGAN discriminator (Isola et al., 2017).
    Outputs a grid of real/fake scores rather than a single scalar.
    Uses InstanceNorm, no sigmoid (use MSE loss instead of BCE).
    """
    def __init__(self, channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(features,     features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 1, 1),

            nn.Conv2d(features * 8, 1, 4, padding=1),
            # No sigmoid — use MSELoss (LSGAN objective)
        )
        self.apply(self._init_weights)

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.net(x)


# ── Image Buffer ────────────────────────────────────────────────────────────────

class ImageBuffer:
    """
    Replay buffer — stores 50 past generated images and returns
    a random mix of new and buffered images to train discriminator.
    Reduces model oscillation.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer   = []

    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.max_size, (1,)).item()
                    result.append(self.buffer[idx].clone())
                    self.buffer[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


if __name__ == "__main__":
    G_AB = Generator()
    G_BA = Generator()
    D_A  = Discriminator()
    D_B  = Discriminator()

    x_a  = torch.randn(2, 3, 64, 64)
    x_b  = torch.randn(2, 3, 64, 64)

    fake_b = G_AB(x_a)
    fake_a = G_BA(x_b)
    rec_a  = G_BA(fake_b)   # cycle reconstruction

    print(f"G_AB output: {fake_b.shape}")   # (2, 3, 64, 64)
    print(f"G_BA output: {fake_a.shape}")   # (2, 3, 64, 64)
    print(f"Cycle rec_a: {rec_a.shape}")    # (2, 3, 64, 64)
    print(f"D_B output:  {D_B(fake_b).shape}")
    print("CycleGAN model check passed.")
