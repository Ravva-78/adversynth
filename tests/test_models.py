import pytest
import torch
from models.vanilla_gan import Generator as VanillaG, Discriminator as VanillaD, LATENT_DIM as VANILLA_Z
from models.dcgan import Generator as DCGen, Discriminator as DCDis, LATENT_DIM as DC_Z
from models.cgan import Generator as CGANGen, Discriminator as CGANDis, LATENT_DIM as CGAN_Z, N_CLASSES
from models.cyclegan import Generator as CycleGen, Discriminator as CycleDis, ImageBuffer
from models.classifier import DRClassifier


def test_vanilla_gan_forward():
    batch_size = 4
    G = VanillaG(latent_dim=VANILLA_Z)
    D = VanillaD()

    z = torch.randn(batch_size, VANILLA_Z)
    fake_imgs = G(z)
    assert fake_imgs.shape == (batch_size, 3, 64, 64), f"Unexpected shape {fake_imgs.shape}"
    assert fake_imgs.min() >= -1.0 and fake_imgs.max() <= 1.0, "Vanilla G output not in Tanh range [-1, 1]"

    pred = D(fake_imgs)
    assert pred.shape == (batch_size, 1), f"Unexpected shape {pred.shape}"
    assert (pred >= 0.0).all() and (pred <= 1.0).all(), "D output not in Sigmoid range [0, 1]"


def test_dcgan_forward():
    batch_size = 4
    G = DCGen(latent_dim=DC_Z)
    D = DCDis()

    z = torch.randn(batch_size, DC_Z, 1, 1)
    fake_imgs = G(z)
    assert fake_imgs.shape == (batch_size, 3, 64, 64), f"Unexpected shape {fake_imgs.shape}"

    pred = D(fake_imgs)
    assert pred.shape == (batch_size, 1), f"Unexpected shape {pred.shape}"


def test_cgan_forward():
    batch_size = 5
    G = CGANGen(latent_dim=CGAN_Z, n_classes=N_CLASSES)
    D = CGANDis(n_classes=N_CLASSES, image_size=64)

    z = torch.randn(batch_size, CGAN_Z, 1, 1)
    labels = torch.arange(N_CLASSES)  # One of each grade 0-4
    fake_imgs = G(z, labels)
    assert fake_imgs.shape == (batch_size, 3, 64, 64)

    pred = D(fake_imgs, labels)
    assert pred.shape == (batch_size, 1)


def test_cgan_label_conditioning_sensitivity():
    """Verify that different labels injected into the same noise vector produce distinct images."""
    G = CGANGen(latent_dim=CGAN_Z, n_classes=N_CLASSES)
    G.eval()

    z = torch.randn(1, CGAN_Z, 1, 1).repeat(2, 1, 1, 1)
    label_healthy = torch.tensor([0])  # Grade 0
    label_severe = torch.tensor([4])   # Grade 4
    labels = torch.cat([label_healthy, label_severe])

    with torch.no_grad():
        out = G(z, labels)

    diff = torch.abs(out[0] - out[1]).sum().item()
    assert diff > 1e-4, "Generator output must depend on conditioning label!"


def test_cyclegan_forward_and_buffer():
    batch_size = 2
    G_AB = CycleGen(channels=3, features=32, n_res_blocks=3)  # lightweight for quick testing
    D_B = CycleDis(channels=3, features=32)

    x = torch.randn(batch_size, 3, 64, 64)
    translated = G_AB(x)
    assert translated.shape == (batch_size, 3, 64, 64)

    patch_out = D_B(translated)
    assert patch_out.dim() == 4
    assert patch_out.shape[0] == batch_size

    # Test ImageBuffer
    buffer = ImageBuffer(max_size=10)
    pushed = buffer.push_and_pop(translated)
    assert pushed.shape == translated.shape


def test_dr_classifier_forward():
    batch_size = 2
    # DRClassifier with ResNet-50
    model = DRClassifier(n_classes=5, freeze_backbone=True)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, 5), f"Expected (2, 5), got {logits.shape}"
