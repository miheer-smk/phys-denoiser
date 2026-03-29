"""
Training script for PhysDenoiser.

Features:
  - Mixed loss: L1 + SSIM for perceptually better results
  - On-the-fly physics noise augmentation (no pre-generated noisy data)
  - Cosine annealing LR schedule
  - PSNR/SSIM tracking per epoch
  - Checkpoint saving (best model + periodic)
  - Works on GPU or CPU automatically

Usage:
    python train.py --data_dir path/to/clean/images --epochs 50 --batch_size 16

Training data:
    Any folder of clean images works. Recommendations:
      - BSD500 (500 images, good variety) — download from the repo README
      - DIV2K (800 high-res images, standard benchmark)
      - Or your own collection of clean photos
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import PhysDenoiser, PhysDenoiserSmall, count_params
from dataset import DenoisingDataset


# ── Metrics ──

def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio in dB."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def gaussian_kernel(size=11, sigma=1.5, channels=3, device='cpu'):
    """Create 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(1) * g.unsqueeze(0)  # outer product
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
    return kernel


def compute_ssim(pred, target, kernel_size=11, sigma=1.5):
    """Structural Similarity Index (simplified, differentiable)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = pred.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels, pred.device)

    mu_pred = torch.nn.functional.conv2d(pred, kernel, padding=kernel_size // 2, groups=channels)
    mu_target = torch.nn.functional.conv2d(target, kernel, padding=kernel_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.conv2d(pred ** 2, kernel, padding=kernel_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.conv2d(target ** 2, kernel, padding=kernel_size // 2, groups=channels) - mu_target_sq
    sigma_cross = torch.nn.functional.conv2d(pred * target, kernel, padding=kernel_size // 2, groups=channels) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


class MixedLoss(nn.Module):
    """L1 + (1 - SSIM) loss for sharper, perceptually better results."""

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)

        # Simplified SSIM loss (differentiable)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = pred.mean(dim=[2, 3], keepdim=True)
        mu_t = target.mean(dim=[2, 3], keepdim=True)
        sigma_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_t = ((target - mu_t) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=[2, 3], keepdim=True)

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        ssim_loss = 1 - ssim.mean()

        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


# ── Training Loop ──

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    count = 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += compute_psnr(output.detach(), clean)
        count += 1

    return total_loss / count, total_psnr / count


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = criterion(output, clean)

        total_loss += loss.item()
        total_psnr += compute_psnr(output, clean)
        total_ssim += compute_ssim(output, clean)
        count += 1

    return total_loss / count, total_psnr / count, total_ssim / count


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    full_dataset = DenoisingDataset(args.data_dir, patch_size=args.patch_size)

    # Train/val split (90/10)
    n_val = max(1, int(len(full_dataset) * 0.1))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    print(f"Train: {n_train} images | Val: {n_val} images")

    # Model
    if args.small:
        model = PhysDenoiserSmall(in_channels=3).to(device)
    else:
        model = PhysDenoiser(in_channels=3, num_features=64, num_layers=12).to(device)

    print(f"Model: {type(model).__name__} ({count_params(model):,} params)")

    # Training setup
    criterion = MixedLoss(alpha=0.8)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    history = []

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_psnr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} PSNR: {train_psnr:.2f} dB | "
            f"Val Loss: {val_loss:.4f} PSNR: {val_psnr:.2f} dB SSIM: {val_ssim:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_psnr': train_psnr,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'lr': lr,
        })

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }, save_dir / 'best_model.pth')
            print(f"  -> New best model saved (PSNR: {val_psnr:.2f} dB)")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')

    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Outputs saved to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PhysDenoiser')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to clean images folder')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Where to save models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--small', action='store_true', help='Use smaller model variant')
    args = parser.parse_args()
    main(args)
