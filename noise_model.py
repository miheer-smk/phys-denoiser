"""
Physics-based noise synthesis for realistic image degradation.

Real camera noise is NOT purely Gaussian. It is a combination of:
  1. Photon shot noise   — Poisson distributed, signal-dependent
  2. Read noise          — Gaussian, from sensor electronics
  3. Quantization noise  — from ADC digitization
  4. Fixed-pattern noise — pixel-to-pixel sensor variation

This module synthesizes realistic noisy images by modeling these physical
processes, giving the denoiser training data that matches real-world conditions
instead of the naive "just add Gaussian" approach.

Reference physics:
  - Shot noise follows Poisson statistics: variance = signal intensity
  - Read noise is approximately N(0, sigma_read) per pixel
  - At low light, read noise dominates; at high light, shot noise dominates
"""

import torch
import numpy as np


def add_poisson_gaussian_noise(clean_img, peak=50.0, sigma_read=0.02):
    """
    Apply physically-motivated Poisson-Gaussian noise to a clean image.

    This models the dominant noise sources in a real CMOS sensor:
      - Poisson (shot) noise: photon counting statistics
      - Gaussian (read) noise: electronics thermal noise

    Args:
        clean_img: Tensor of shape (C, H, W) or (B, C, H, W), values in [0, 1].
        peak: Simulated photon count at max intensity. Lower = noisier (low light).
              Typical range: 10 (very dark) to 200 (well-lit).
        sigma_read: Std dev of additive read noise. Typical: 0.01 to 0.05.

    Returns:
        Noisy image tensor, same shape, clamped to [0, 1].
    """
    # Scale to photon counts
    scaled = clean_img * peak

    # Poisson shot noise (signal-dependent)
    noisy = torch.poisson(scaled) / peak

    # Additive Gaussian read noise (signal-independent)
    read_noise = torch.randn_like(noisy) * sigma_read
    noisy = noisy + read_noise

    return torch.clamp(noisy, 0.0, 1.0)


def add_gaussian_noise(clean_img, sigma=0.05):
    """
    Simple additive white Gaussian noise (AWGN).
    Baseline comparison — not physically accurate but widely used.

    Args:
        clean_img: Tensor, values in [0, 1].
        sigma: Noise standard deviation.

    Returns:
        Noisy image tensor, clamped to [0, 1].
    """
    noise = torch.randn_like(clean_img) * sigma
    return torch.clamp(clean_img + noise, 0.0, 1.0)


def add_heteroscedastic_gaussian(clean_img, sigma_slope=0.1, sigma_bias=0.01):
    """
    Signal-dependent Gaussian noise — a practical approximation of
    the Poisson-Gaussian model used in many denoising papers.

    Noise variance = sigma_slope * pixel_intensity + sigma_bias

    This is faster than Poisson sampling and differentiable, making it
    suitable for noise-aware training augmentation.

    Args:
        clean_img: Tensor, values in [0, 1].
        sigma_slope: Scales noise with signal intensity.
        sigma_bias: Constant noise floor (read noise approximation).

    Returns:
        Noisy image, clamped to [0, 1].
    """
    sigma_map = torch.sqrt(sigma_slope * clean_img + sigma_bias)
    noise = torch.randn_like(clean_img) * sigma_map
    return torch.clamp(clean_img + noise, 0.0, 1.0)


def sample_noise_params():
    """
    Randomly sample noise parameters to simulate varying capture conditions
    during training (different ISO, lighting, sensors).

    Returns:
        dict with noise type and parameters.
    """
    noise_type = np.random.choice(
        ['poisson_gaussian', 'gaussian', 'heteroscedastic'],
        p=[0.5, 0.25, 0.25]
    )

    if noise_type == 'poisson_gaussian':
        # Simulate ISO 100 (bright) to ISO 6400 (dark)
        peak = np.random.uniform(15, 200)
        sigma_read = np.random.uniform(0.005, 0.04)
        return {'type': noise_type, 'peak': peak, 'sigma_read': sigma_read}

    elif noise_type == 'gaussian':
        sigma = np.random.uniform(0.01, 0.08)
        return {'type': noise_type, 'sigma': sigma}

    else:
        sigma_slope = np.random.uniform(0.02, 0.15)
        sigma_bias = np.random.uniform(0.005, 0.02)
        return {'type': noise_type, 'sigma_slope': sigma_slope, 'sigma_bias': sigma_bias}


def apply_noise(clean_img, params=None):
    """
    Apply noise with given or randomly sampled parameters.

    Args:
        clean_img: Tensor, values in [0, 1].
        params: dict from sample_noise_params(), or None to auto-sample.

    Returns:
        (noisy_img, params) tuple.
    """
    if params is None:
        params = sample_noise_params()

    if params['type'] == 'poisson_gaussian':
        noisy = add_poisson_gaussian_noise(clean_img, params['peak'], params['sigma_read'])
    elif params['type'] == 'gaussian':
        noisy = add_gaussian_noise(clean_img, params['sigma'])
    else:
        noisy = add_heteroscedastic_gaussian(clean_img, params['sigma_slope'], params['sigma_bias'])

    return noisy, params


if __name__ == "__main__":
    # Demo: generate noisy versions of a synthetic gradient image
    print("Noise model sanity check:")
    test_img = torch.linspace(0, 1, 256).unsqueeze(0).unsqueeze(0).expand(1, 3, 256, 256)

    for _ in range(5):
        noisy, p = apply_noise(test_img)
        psnr = 10 * torch.log10(1.0 / torch.mean((test_img - noisy) ** 2))
        print(f"  {p['type']:25s} -> PSNR: {psnr.item():.1f} dB")
    print("Done.")
