import pytest
import yaml
from pathlib import Path


CONFIG_DIR = Path("configs")
CONFIG_FILES = [
    "vanilla_gan.yaml",
    "dcgan.yaml",
    "cgan.yaml",
    "cyclegan.yaml",
]


@pytest.mark.parametrize("config_filename", CONFIG_FILES)
def test_config_file_validity(config_filename):
    config_path = CONFIG_DIR / config_filename
    assert config_path.exists(), f"Missing config: {config_path}"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict), f"Config {config_filename} is not a valid dict"
    assert "image_size" in cfg, f"'image_size' missing in {config_filename}"
    assert "batch_size" in cfg, f"'batch_size' missing in {config_filename}"
    assert "lr" in cfg, f"'lr' missing in {config_filename}"
    assert "epochs" in cfg, f"'epochs' missing in {config_filename}"
    assert cfg["lr"] > 0, "Learning rate must be positive"
    assert cfg["batch_size"] > 0, "Batch size must be positive"
