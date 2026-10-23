import pytest
import torch
import numpy as np
from evaluation.mode_collapse import measure_variance
from models.dcgan import Generator as DCGen, Discriminator as DCDis, LATENT_DIM as DC_Z
from models.cgan import Generator as CGANGen, Discriminator as CGANDis, LATENT_DIM as CGAN_Z, N_CLASSES


def test_measure_variance_dcgan():
    device = torch.device("cpu")
    G = DCGen(latent_dim=DC_Z).to(device)
    D = DCDis().to(device)

    variances = measure_variance(G, D, DC_Z, device, n_batches=3, batch_size=8, is_cgan=False)
    assert isinstance(variances, np.ndarray)
    assert len(variances) == 3
    assert (variances >= 0).all(), "Variance must be non-negative"


def test_measure_variance_cgan():
    device = torch.device("cpu")
    G = CGANGen(latent_dim=CGAN_Z, n_classes=N_CLASSES).to(device)
    D = CGANDis(n_classes=N_CLASSES, image_size=64).to(device)

    variances = measure_variance(G, D, CGAN_Z, device, n_batches=3, batch_size=8, is_cgan=True)
    assert isinstance(variances, np.ndarray)
    assert len(variances) == 3
    assert (variances >= 0).all(), "Variance must be non-negative"
