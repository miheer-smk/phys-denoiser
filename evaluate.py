"""
Evaluation script — benchmark PhysDenoiser against classical methods.

Adds noise to clean test images, denoises with all methods, and compares
PSNR/SSIM. Generates a results summary table.

Usage:
    python evaluate.py --test_dir path/to/clean/test/images --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

from model import PhysDenoiser, PhysDenoiserSmall, count_params
from noise_model import add_poisson_gaussian_noise, add_gaussian_noise
from inference import load_model, denoise_image, tensor_to_numpy


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def psnr(pred, target):
    """PSNR between two numpy uint8 images."""
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def ssim_channel(a, b, C1=6.5025, C2=58.5225):
    """SSIM on a single channel (numpy float64)."""
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a2 = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a ** 2
    sigma_b2 = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b ** 2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return np.mean(num / den)


def ssim(pred, target):
    """SSIM between two numpy uint8 images (averaged over channels)."""
    pred_f = pred.astype(np.float64)
    target_f = target.astype(np.float64)
    channels = []
    for c in range(3):
        channels.append(ssim_channel(pred_f[:, :, c], target_f[:, :, c]))
    return np.mean(channels)


def evaluate_all(test_dir, model, device, noise_configs):
    """Run evaluation across noise levels and methods."""
    to_tensor = transforms.ToTensor()
    test_dir = Path(test_dir)
    image_paths = sorted([p for p in test_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS])

    if not image_paths:
        print(f"No images found in {test_dir}")
        return

    print(f"Evaluating on {len(image_paths)} images\n")

    all_results = []

    for noise_cfg in noise_configs:
        print(f"--- Noise: {noise_cfg['name']} ---")
        method_psnrs = {'Noisy': [], 'PhysDenoiser': [], 'NLM': [], 'Bilateral': []}
        method_ssims = {'Noisy': [], 'PhysDenoiser': [], 'NLM': [], 'Bilateral': []}

        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            # Resize for consistency
            img = img.resize((256, 256), Image.BICUBIC)
            clean_tensor = to_tensor(img).unsqueeze(0)
            clean_np = np.array(img)

            # Add noise
            if noise_cfg['type'] == 'poisson_gaussian':
                noisy_tensor = add_poisson_gaussian_noise(
                    clean_tensor.squeeze(0),
                    peak=noise_cfg['peak'],
                    sigma_read=noise_cfg['sigma_read']
                ).unsqueeze(0)
            else:
                noisy_tensor = clean_tensor + torch.randn_like(clean_tensor) * noise_cfg['sigma']
                noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

            noisy_np = tensor_to_numpy(noisy_tensor)

            # Measure noisy baseline
            method_psnrs['Noisy'].append(psnr(noisy_np, clean_np))
            method_ssims['Noisy'].append(ssim(noisy_np, clean_np))

            # Neural denoising
            denoised_tensor = denoise_image(model, noisy_tensor, device, tile_size=0)
            denoised_np = tensor_to_numpy(denoised_tensor)
            method_psnrs['PhysDenoiser'].append(psnr(denoised_np, clean_np))
            method_ssims['PhysDenoiser'].append(ssim(denoised_np, clean_np))

            # Classical: NLM
            nlm = cv2.fastNlMeansDenoisingColored(noisy_np, None, h=10, hForColorComponents=10)
            method_psnrs['NLM'].append(psnr(nlm, clean_np))
            method_ssims['NLM'].append(ssim(nlm, clean_np))

            # Classical: Bilateral
            bilateral = cv2.bilateralFilter(noisy_np, d=9, sigmaColor=75, sigmaSpace=75)
            method_psnrs['Bilateral'].append(psnr(bilateral, clean_np))
            method_ssims['Bilateral'].append(ssim(bilateral, clean_np))

        # Print results table
        print(f"\n{'Method':<18} {'PSNR (dB)':<14} {'SSIM':<10}")
        print("-" * 42)
        for method in method_psnrs:
            avg_psnr = np.mean(method_psnrs[method])
            avg_ssim = np.mean(method_ssims[method])
            print(f"{method:<18} {avg_psnr:<14.2f} {avg_ssim:<10.4f}")
            all_results.append({
                'noise': noise_cfg['name'],
                'method': method,
                'psnr': float(avg_psnr),
                'ssim': float(avg_ssim),
            })
        print()

    return all_results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = load_model(args.checkpoint, small=args.small, device=device)

    noise_configs = [
        {'name': 'Low Light (ISO 3200)', 'type': 'poisson_gaussian', 'peak': 25, 'sigma_read': 0.03},
        {'name': 'Moderate (ISO 800)', 'type': 'poisson_gaussian', 'peak': 80, 'sigma_read': 0.015},
        {'name': 'Gaussian sigma=0.05', 'type': 'gaussian', 'sigma': 0.05},
    ]

    results = evaluate_all(args.test_dir, model, device, noise_configs)

    # Save results
    if results:
        out_path = Path(args.output) if args.output else Path('eval_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PhysDenoiser')
    parser.add_argument('--test_dir', type=str, required=True, help='Clean test images folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='eval_results.json', help='Output JSON path')
    parser.add_argument('--small', action='store_true', help='Use small model')
    args = parser.parse_args()
    main(args)
