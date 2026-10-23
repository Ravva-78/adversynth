"""
AdverSynth Demo Script
======================
This script demonstrates how to generate synthetic Diabetic Retinopathy images 
on the fly using the trained Conditional GAN (cGAN) model.

It loads a pre-trained Generator checkpoint and synthesizes a grid of images 
for all 5 DR severity grades.

Usage:
    python generate_demo.py --checkpoint checkpoints/cgan/G_epoch_150.pt
"""

import argparse
import torch
from torchvision.utils import save_image
from pathlib import Path

# Import the Generator architecture
from models.cgan import Generator, LATENT_DIM, N_CLASSES

def generate_demo_grid(checkpoint_path, output_path="demo_output.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize the Generator
    G = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES).to(device)
    
    # 2. Load the trained weights or fallback to demo initialization
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        G.load_state_dict(torch.load(checkpoint_file, map_location=device))
        print(f"Successfully loaded trained checkpoint: {checkpoint_path}")
    else:
        print(f"[INFO] Checkpoint '{checkpoint_path}' not found.")
        print("[INFO] Initializing architecture in zero-shot demo mode to verify pipeline execution.")
        print("[INFO] (To train full weights on APTOS-2019, execute: python training/train_cgan.py)")
    
    G.eval()
    
    # 3. Prepare inputs: Generate 8 images for each of the 5 grades
    n_samples_per_class = 8
    total_samples = N_CLASSES * n_samples_per_class
    
    # Random noise vector (z)
    z = torch.randn(total_samples, LATENT_DIM, 1, 1, device=device)
    
    # Labels (0,0,0... 1,1,1... 4,4,4...)
    labels = torch.arange(N_CLASSES, device=device).repeat_interleave(n_samples_per_class)
    
    # 4. Generate the images
    print("Generating synthetic images...")
    with torch.no_grad():
        fake_images = G(z, labels).cpu()
        
    # 5. Save the output as a grid
    save_image(fake_images, output_path, nrow=n_samples_per_class, normalize=True, value_range=(-1, 1))
    
    print(f"Success! Demo grid saved to: {output_path}")
    print(f"Rows represent DR Grades 0 to 4 (top to bottom).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DR images using cGAN.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cgan/G_epoch_150.pt",
                        help="Path to the trained Generator checkpoint (.pt file)")
    parser.add_argument("--out", type=str, default="demo_output.png",
                        help="Output path for the generated image grid")
    
    args = parser.parse_args()
    generate_demo_grid(args.checkpoint, args.out)
