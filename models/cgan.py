"""
Phase 4 — Conditional GAN (cGAN).
Mirza & Osindero (2014) adapted to DCGAN conv architecture.

Conditioning: severity label (0-4) embedded and injected into
both Generator (concatenated to noise z) and Discriminator
(concatenated as extra channels to image input).

This forces G to synthesise images that match a given DR grade,
and D to reject images that are realistic but clinically wrong.
"""
import torch
import torch.nn as nn

LATENT_DIM     = 100
N_CLASSES      = 5       # DR grades 0-4
EMBED_DIM      = 50      # label embedding size
FEATURES_G     = 64
FEATURES_D     = 64
IMAGE_CHANNELS = 3
IMAGE_SIZE     = 64


class Generator(nn.Module):
    """
    (z, label) → fake image conditioned on DR grade.
    Label is embedded and concatenated to z before the first conv block.
    """
    def __init__(self, latent_dim=LATENT_DIM, n_classes=N_CLASSES,
                 embed_dim=EMBED_DIM, features_g=FEATURES_G, channels=IMAGE_CHANNELS):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, embed_dim)
        in_channels    = latent_dim + embed_dim
        fg = features_g

        self.net = nn.Sequential(
            self._block(in_channels, fg * 16, 4, 1, 0),
            self._block(fg * 16,    fg * 8,  4, 2, 1),
            self._block(fg * 8,     fg * 4,  4, 2, 1),
            self._block(fg * 4,     fg * 2,  4, 2, 1),
            nn.ConvTranspose2d(fg * 2, channels, 4, 2, 1, bias=False),
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

    def forward(self, z, labels):
        # z: (B, latent_dim, 1, 1)  labels: (B,)
        emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)  # (B, embed_dim, 1, 1)
        x   = torch.cat([z, emb], dim=1)                          # (B, latent+embed, 1, 1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    (image, label) → real/fake probability.
    Label is embedded to an IMAGE_SIZE×IMAGE_SIZE map and
    concatenated to the image as extra channels.
    D must learn: is this image realistic AND does it match the label?
    """
    def __init__(self, n_classes=N_CLASSES, image_size=IMAGE_SIZE,
                 features_d=FEATURES_D, channels=IMAGE_CHANNELS):
        super().__init__()
        self.image_size = image_size
        # Embed label → single spatial channel
        self.label_emb = nn.Embedding(n_classes, image_size * image_size)
        in_channels    = channels + 1   # RGB + label channel
        fd = features_d

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, fd, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(fd,     fd * 2, 4, 2, 1),
            self._block(fd * 2, fd * 4, 4, 2, 1),
            self._block(fd * 4, fd * 8, 4, 2, 1),

            nn.Conv2d(fd * 8, 1, 4, 1, 0, bias=False),
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

    def forward(self, x, labels):
        # x: (B, 3, H, W)  labels: (B,)
        emb = self.label_emb(labels).view(-1, 1, self.image_size, self.image_size)
        x   = torch.cat([x, emb], dim=1)   # (B, 4, H, W)
        return self.net(x).view(-1, 1)


if __name__ == "__main__":
    G      = Generator()
    D      = Discriminator()
    z      = torch.randn(4, LATENT_DIM, 1, 1)
    labels = torch.randint(0, N_CLASSES, (4,))
    fake   = G(z, labels)
    pred   = D(fake, labels)
    print(f"Generator output:      {fake.shape}")   # (4, 3, 64, 64)
    print(f"Discriminator output:  {pred.shape}")   # (4, 1)
    print("Conditional GAN model check passed.")
