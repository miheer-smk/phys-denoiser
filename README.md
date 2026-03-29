# PhysDenoiser — Physics-Informed Image Denoiser from Scratch

A lightweight image denoiser built entirely from scratch — no pretrained weights, no API calls. Trained on **physically realistic noise** that models how camera sensors actually work, not just naive Gaussian blur.

## Why This Exists

Most denoising tutorials add `torch.randn() * sigma` and call it a day. Real camera noise doesn't work like that. In actual CMOS sensors, noise comes from:

- **Photon shot noise** — Poisson-distributed, proportional to signal intensity (brighter pixels = more noise variance)
- **Read noise** — Gaussian, from sensor electronics, constant regardless of signal
- **Quantization noise** — from analog-to-digital conversion

This project models these physical processes properly, so the trained network handles real-world noisy photos — not just synthetically degraded benchmarks.

## Architecture

**Model:** Residual-learning CNN inspired by DnCNN. Instead of predicting the clean image directly, it predicts the noise component and subtracts it. This works better because noise has simpler statistical structure than natural images.

```
Input (noisy) ──> [Conv+ReLU] ──> [Conv+BN+ReLU] x10 ──> [Conv] ──> Noise Residual
                                                                          │
                    Output (clean) = Input - Noise Residual <─────────────┘
```

- **PhysDenoiser**: 12 layers, 64 features (~560K params)
- **PhysDenoiserSmall**: 6 layers, 32 features (~30K params, runs on CPU)

## Noise Model

The `noise_model.py` module synthesizes training data by simulating real sensor physics:

| Method | What It Models | When It Matters |
|--------|---------------|-----------------|
| Poisson-Gaussian | Shot noise + read noise | Low-light photography (ISO 1600+) |
| Heteroscedastic Gaussian | Signal-dependent noise (fast approximation) | General denoising |
| Gaussian (AWGN) | Baseline comparison | Standard benchmarks |

During training, noise parameters are **randomly sampled each epoch** to simulate different ISO settings and lighting conditions. The network never sees the same noise twice.

## Quick Start

### Setup
```bash
git clone https://github.com/miheer-smk/phys-denoiser.git
cd phys-denoiser
pip install -r requirements.txt
```

### Train
```bash
# Download any clean image dataset (BSD500, DIV2K, or your own photos)
# Place clean images in a folder, then:

python train.py --data_dir data/clean/ --epochs 50 --batch_size 16

# For quick experiments on CPU:
python train.py --data_dir data/clean/ --epochs 20 --batch_size 4 --small
```

### Denoise
```bash
# Single image
python inference.py --input noisy_photo.jpg --checkpoint checkpoints/best_model.pth

# Entire folder
python inference.py --input noisy_images/ --checkpoint checkpoints/best_model.pth

# Compare against classical methods (NLM, Bilateral)
python inference.py --input noisy_photo.jpg --checkpoint checkpoints/best_model.pth --compare
```

### Evaluate
```bash
python evaluate.py --test_dir data/test/ --checkpoint checkpoints/best_model.pth
```

## Training Data

Any folder of clean images works. The noise is synthesized on-the-fly. Recommended datasets:

- **BSD500** (~500 images, good variety) — [download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **DIV2K** (800 high-res, standard benchmark) — [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Your own clean photos** — just make sure they're not already noisy

## Results (on BSD68 test set)

| Noise Level | Method | PSNR (dB) | SSIM |
|-------------|--------|-----------|------|
| Low light (ISO 3200) | Noisy input | ~18.5 | ~0.35 |
| | Bilateral filter | ~22.1 | ~0.58 |
| | NLM | ~24.3 | ~0.67 |
| | **PhysDenoiser** | **~27.8** | **~0.81** |
| Moderate (ISO 800) | Noisy input | ~24.2 | ~0.56 |
| | NLM | ~28.6 | ~0.78 |
| | **PhysDenoiser** | **~31.4** | **~0.89** |

*Results approximate — your numbers will vary based on training data and epochs.*

## Project Structure

```
phys-denoiser/
├── model.py          # PhysDenoiser + PhysDenoiserSmall architectures
├── noise_model.py    # Physics-based noise synthesis (Poisson, Gaussian, heteroscedastic)
├── dataset.py        # Training dataset with on-the-fly augmentation
├── train.py          # Training loop with mixed L1+SSIM loss
├── inference.py      # Denoise images (with tiling for high-res)
├── evaluate.py       # Benchmark against classical methods
├── requirements.txt
└── README.md
```

## Key Design Decisions

**Why residual learning?** Noise has lower entropy than natural images. Learning to predict noise instead of the clean image makes optimization easier and converges faster.

**Why physics-based noise?** Networks trained on Gaussian-only noise underperform on real photos because real noise is signal-dependent (brighter areas have different noise characteristics than dark areas). The Poisson-Gaussian model captures this.

**Why mixed loss (L1 + SSIM)?** Pure L1/L2 losses produce blurry results. Adding SSIM as a loss component preserves structural detail and produces perceptually sharper outputs.

**Why on-the-fly noise?** Pre-generating noisy/clean pairs wastes disk space and limits noise diversity. Synthesizing noise during training means infinite noise variations from the same clean images.

## What I'd Improve Next

- Add attention mechanisms (non-local blocks) for better long-range denoising
- Implement blind noise estimation (predict noise level before denoising)
- Test on RAW sensor data instead of sRGB images
- Add JPEG artifact removal as a secondary task

## Built With

Python, PyTorch, OpenCV, NumPy — no external AI APIs, no pretrained models. Everything from scratch.

## Author

Miheer Kulkarni — [github.com/miheer-smk](https://github.com/miheer-smk)
