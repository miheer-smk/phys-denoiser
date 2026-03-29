"""
Inference script — denoise images using a trained PhysDenoiser.

Usage:
    # Single image
    python inference.py --input noisy_photo.jpg --checkpoint checkpoints/best_model.pth

    # Entire folder
    python inference.py --input noisy_images/ --checkpoint checkpoints/best_model.pth --output_dir results/

    # Use small model variant
    python inference.py --input noisy_photo.jpg --checkpoint checkpoints/best_model.pth --small

    # Compare with classical methods
    python inference.py --input noisy_photo.jpg --checkpoint checkpoints/best_model.pth --compare
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

from model import PhysDenoiser, PhysDenoiserSmall


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_model(checkpoint_path, small=False, device='cpu'):
    """Load trained model from checkpoint."""
    if small:
        model = PhysDenoiserSmall(in_channels=3)
    else:
        model = PhysDenoiser(in_channels=3, num_features=64, num_layers=12)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'val_psnr' in ckpt:
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, val PSNR: {ckpt['val_psnr']:.2f} dB)")
    else:
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    return model


def denoise_image(model, img_tensor, device, tile_size=512, overlap=64):
    """
    Denoise an image with optional tiling for large images.

    Tiling avoids GPU OOM on high-res images by processing overlapping
    patches and blending them together.

    Args:
        model: Trained PhysDenoiser.
        img_tensor: (1, 3, H, W) tensor in [0, 1].
        device: torch device.
        tile_size: Tile size for processing. Set to 0 for full image.
        overlap: Overlap between tiles for seamless blending.

    Returns:
        Denoised tensor, same shape.
    """
    _, _, H, W = img_tensor.shape

    if tile_size == 0 or (H <= tile_size and W <= tile_size):
        # Process full image at once
        with torch.no_grad():
            return model(img_tensor.to(device)).cpu()

    # Tiled processing for large images
    step = tile_size - overlap
    output = torch.zeros_like(img_tensor)
    weight_map = torch.zeros_like(img_tensor)

    # Simple linear blending weights
    blend = torch.ones(1, 3, tile_size, tile_size)
    # Feather edges
    for i in range(overlap):
        factor = i / overlap
        blend[:, :, i, :] *= factor
        blend[:, :, -(i + 1), :] *= factor
        blend[:, :, :, i] *= factor
        blend[:, :, :, -(i + 1)] *= factor

    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            # Clamp to valid range
            y_start = max(0, y_start)
            x_start = max(0, x_start)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]

            # Pad if needed
            pad_h = tile_size - tile.shape[2]
            pad_w = tile_size - tile.shape[3]
            if pad_h > 0 or pad_w > 0:
                tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                denoised_tile = model(tile.to(device)).cpu()

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                denoised_tile = denoised_tile[:, :, :tile_size - pad_h, :tile_size - pad_w]

            actual_h = y_end - y_start
            actual_w = x_end - x_start
            b = blend[:, :, :actual_h, :actual_w]

            output[:, :, y_start:y_end, x_start:x_end] += denoised_tile * b
            weight_map[:, :, y_start:y_end, x_start:x_end] += b

    output = output / (weight_map + 1e-8)
    return torch.clamp(output, 0, 1)


def classical_denoise(img_np):
    """
    Apply classical (non-ML) denoising methods for comparison.

    Returns dict of method_name -> denoised_image_np.
    """
    results = {}

    # Non-Local Means (best classical method for most cases)
    nlm = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hForColorComponents=10)
    results['NLM'] = nlm

    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    results['Bilateral'] = bilateral

    # Gaussian blur (naive baseline)
    gaussian = cv2.GaussianBlur(img_np, (5, 5), 0)
    results['Gaussian'] = gaussian

    return results


def tensor_to_numpy(tensor):
    """Convert (1, 3, H, W) tensor to (H, W, 3) uint8 numpy array."""
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def numpy_to_tensor(img_np):
    """Convert (H, W, 3) uint8 numpy array to (1, 3, H, W) tensor."""
    img = torch.from_numpy(img_np).float() / 255.0
    return img.permute(2, 0, 1).unsqueeze(0)


def process_single(input_path, model, device, output_dir, compare=False, tile_size=512):
    """Process a single image."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img = Image.open(input_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)  # (1, 3, H, W)

    print(f"\nProcessing: {input_path.name} ({img.size[0]}x{img.size[1]})")

    # Neural denoising
    t0 = time.time()
    denoised = denoise_image(model, img_tensor, device, tile_size=tile_size)
    neural_time = time.time() - t0
    print(f"  Neural denoise: {neural_time:.2f}s")

    # Save result
    out_name = f"{input_path.stem}_denoised{input_path.suffix}"
    denoised_np = tensor_to_numpy(denoised)
    Image.fromarray(denoised_np).save(output_dir / out_name, quality=95)
    print(f"  Saved: {output_dir / out_name}")

    # Classical comparison
    if compare:
        img_np = np.array(img)
        classical = classical_denoise(img_np)
        for name, result in classical.items():
            comp_name = f"{input_path.stem}_{name.lower()}{input_path.suffix}"
            Image.fromarray(result).save(output_dir / comp_name, quality=95)
            print(f"  Saved ({name}): {output_dir / comp_name}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, small=args.small, device=device)

    input_path = Path(args.input)

    if input_path.is_file():
        process_single(input_path, model, device, args.output_dir, args.compare, args.tile_size)
    elif input_path.is_dir():
        images = [p for p in input_path.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
        print(f"Found {len(images)} images in {input_path}")
        for img_path in images:
            process_single(img_path, model, device, args.output_dir, args.compare, args.tile_size)
    else:
        print(f"Error: {input_path} not found")
        return

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise images with PhysDenoiser')
    parser.add_argument('--input', type=str, required=True, help='Image file or folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='results', help='Output folder')
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size (0 = full image)')
    parser.add_argument('--small', action='store_true', help='Use small model variant')
    parser.add_argument('--compare', action='store_true', help='Also run classical methods')
    args = parser.parse_args()
    main(args)
