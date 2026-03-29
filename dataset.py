"""
Dataset for training the denoiser.

Uses any folder of clean images. Noise is synthesized on-the-fly using
the physics-based noise model, so the network sees different noise
realizations every epoch — natural augmentation.

Supports:
  - Random cropping to patch_size x patch_size
  - Random horizontal/vertical flips
  - Random 90-degree rotations
  - On-the-fly noise application with random parameters
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from noise_model import apply_noise


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class DenoisingDataset(Dataset):
    """
    Loads clean images from a directory, applies random crops and
    physics-based noise on the fly.

    Each __getitem__ returns (noisy_patch, clean_patch) as tensors in [0, 1].
    """

    def __init__(self, image_dir, patch_size=128, augment=True):
        """
        Args:
            image_dir: Path to folder containing clean images.
            patch_size: Size of random crops for training.
            augment: Whether to apply random flips/rotations.
        """
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.augment = augment

        self.image_paths = sorted([
            p for p in self.image_dir.rglob('*')
            if p.suffix.lower() in SUPPORTED_EXTS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Supported formats: {SUPPORTED_EXTS}"
            )

        print(f"Dataset: {len(self.image_paths)} images from {image_dir}")

        self.to_tensor = transforms.ToTensor()  # Converts to [0, 1] float

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Ensure image is large enough for cropping
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            # Resize smallest dimension up
            scale = max(self.patch_size / w, self.patch_size / h) * 1.1
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            w, h = img.size

        # Random crop
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        img = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Augmentation: random flips and rotations
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                img = img.rotate(90)

        clean = self.to_tensor(img)  # (3, patch_size, patch_size) in [0, 1]

        # Apply physics-based noise (random params each time)
        noisy, _ = apply_noise(clean)

        return noisy, clean


class ImageFolderInference(Dataset):
    """Simple dataset for inference — loads full images, no cropping."""

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for p in self.image_dir.rglob('*')
            if p.suffix.lower() in SUPPORTED_EXTS
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        tensor = self.to_tensor(img)
        return tensor, str(path)


if __name__ == "__main__":
    print("Dataset module loaded successfully.")
    print("Usage: DenoisingDataset('path/to/clean/images', patch_size=128)")
