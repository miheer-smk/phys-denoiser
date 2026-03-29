"""
PhysDenoiser — A lightweight residual CNN for image denoising.

Architecture: DnCNN-style with batch normalization and residual learning.
The network learns to predict the noise component, which is then subtracted
from the noisy input to recover the clean image.

This is a from-scratch implementation — no pretrained weights, no API calls.
"""

import torch
import torch.nn as nn


class PhysDenoiser(nn.Module):
    """
    Residual learning denoiser inspired by DnCNN (Zhang et al., 2017).

    Instead of learning the clean image directly, the network predicts
    the noise residual. This makes training more stable and converges faster
    because noise has simpler structure than natural images.

    Architecture:
        - Conv + ReLU (input layer)
        - N x (Conv + BatchNorm + ReLU) (hidden layers)
        - Conv (output layer, predicts noise residual)
    """

    def __init__(self, in_channels=3, num_features=64, num_layers=12):
        super().__init__()

        layers = []

        # First layer: Conv + ReLU (no batch norm)
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: Conv (predicts noise residual)
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for stable training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass: predict noise and subtract from input.

        Args:
            x: Noisy image tensor, shape (B, C, H, W), values in [0, 1]

        Returns:
            Denoised image tensor, same shape, clamped to [0, 1]
        """
        noise_residual = self.network(x)
        clean = x - noise_residual
        return torch.clamp(clean, 0.0, 1.0)


class PhysDenoiserSmall(nn.Module):
    """
    Compact variant for fast inference on CPU / low-VRAM GPUs.
    6 layers, 32 features. Good enough for moderate noise levels.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.clamp(x - self.net(x), 0.0, 1.0)


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = PhysDenoiser(in_channels=3, num_features=64, num_layers=12)
    print(f"PhysDenoiser    — params: {count_params(model):,}")

    small = PhysDenoiserSmall(in_channels=3)
    print(f"PhysDenoiserSmall — params: {count_params(small):,}")

    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print("Sanity check passed.")
