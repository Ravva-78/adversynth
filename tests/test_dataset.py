import pytest
import torch
from PIL import Image
from data.dataset import get_gan_transforms, get_eval_transforms, get_classifier_transforms


def test_gan_transforms():
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    transform = get_gan_transforms(image_size=64)
    tensor = transform(img)

    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    # Check normalized bounds [-1, 1]
    assert tensor.min() >= -1.01 and tensor.max() <= 1.01


def test_eval_transforms():
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    transform = get_eval_transforms(image_size=128)
    tensor = transform(img)

    assert tensor.shape == (3, 128, 128)
    assert tensor.min() >= -1.01 and tensor.max() <= 1.01


def test_classifier_transforms():
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    train_tf = get_classifier_transforms(image_size=224, train=True)
    eval_tf = get_classifier_transforms(image_size=224, train=False)

    tensor_train = train_tf(img)
    tensor_eval = eval_tf(img)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_eval.shape == (3, 224, 224)
