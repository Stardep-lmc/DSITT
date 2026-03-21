#!/usr/bin/env python3
"""
Smoke test: verify the DSITT model can be instantiated and run forward pass.
Usage: python -m tools.test_model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from models.dsitt import build_dsitt


def test_forward_pass():
    """Test model construction and forward pass with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model with default config
    print("\n[1/4] Building model...")
    model = build_dsitt()
    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {num_trainable / 1e6:.1f}M")

    # Create dummy video clip (2 frames)
    print("\n[2/4] Creating dummy data (2-frame clip, 320x320)...")
    B, C, H, W = 1, 3, 320, 320
    num_frames = 2

    frames = [torch.randn(B, C, H, W, device=device) for _ in range(num_frames)]

    # Create dummy targets
    targets = []
    for t in range(num_frames):
        num_targets = 5
        targets.append({
            'labels': torch.randint(0, 7, (num_targets,), device=device),
            'boxes': torch.rand(num_targets, 4, device=device) * 0.5 + 0.25,
            'track_ids': torch.arange(num_targets, device=device) + (t * 2),
        })

    # Forward pass (training mode)
    print("\n[3/4] Running forward pass (training mode)...")
    try:
        loss_dict = model(frames, targets)
        print(f"  ✓ Forward pass successful!")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.item():.4f}")
            else:
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Forward pass (inference mode)
    print("\n[4/4] Running forward pass (inference mode)...")
    model.eval()
    try:
        with torch.no_grad():
            result = model(frames)
        preds = result['predictions']
        print(f"  ✓ Inference successful! Got {len(preds)} frame predictions.")
        for i, pred in enumerate(preds):
            high_score = (pred['scores'] > 0.3).sum().item()
            print(f"    Frame {i}: {high_score} detections with score > 0.3")
    except Exception as e:
        print(f"  ✗ Inference FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All smoke tests passed!")
    print("=" * 50)
    return True


if __name__ == '__main__':
    success = test_forward_pass()
    sys.exit(0 if success else 1)