#!/usr/bin/env python3
"""
Smoke test: verify the DSITTv2 model can be instantiated and run forward pass.
Tests dual-modality input, MTUQ queries, CMC loss, gate weights, and inference.

Usage: python -m tools.test_model_v2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from models.dsitt_v2 import build_dsitt_v2


def test_forward_pass():
    """Test v2 model construction and forward pass with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model with default config
    print("\n[1/5] Building DSITTv2 model...")
    model = build_dsitt_v2()
    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {num_trainable / 1e6:.1f}M")

    # Create dummy dual-modality video clip (2 frames)
    print("\n[2/5] Creating dummy dual-modality data (2-frame clip, 320x320)...")
    B, C, H, W = 1, 3, 320, 320
    num_frames = 2

    frames_rgb = [torch.randn(B, C, H, W, device=device) for _ in range(num_frames)]
    frames_ir = [torch.randn(B, C, H, W, device=device) for _ in range(num_frames)]

    targets = []
    for t in range(num_frames):
        num_targets = 5
        targets.append({
            'labels': torch.randint(0, 7, (num_targets,), device=device),
            'boxes': torch.rand(num_targets, 4, device=device) * 0.5 + 0.25,
            'track_ids': torch.arange(num_targets, device=device),
        })

    # Forward pass (training mode)
    print("\n[3/5] Running forward pass (training mode)...")
    try:
        loss_dict = model(frames_rgb, frames_ir, targets)
        print(f"  ✓ Forward pass successful!")
        for k, v in sorted(loss_dict.items()):
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {v.item():.4f}")
            else:
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify expected loss keys
    print("\n[4/5] Verifying loss dict keys...")
    expected_keys = ['loss', 'loss_cls', 'loss_l1']
    optional_keys = ['loss_nwd', 'loss_giou', 'loss_cmc', 'gate_rgb', 'gate_ir', 'gate_motion']
    missing = [k for k in expected_keys if k not in loss_dict]
    if missing:
        print(f"  ✗ Missing required keys: {missing}")
        return False
    print(f"  ✓ All required keys present: {expected_keys}")
    found_optional = [k for k in optional_keys if k in loss_dict]
    print(f"  ✓ Optional keys found: {found_optional}")

    # Forward pass (inference mode)
    print("\n[5/5] Running forward pass (inference mode)...")
    model.eval()
    try:
        with torch.no_grad():
            result = model(frames_rgb, frames_ir)
        preds = result['predictions']
        print(f"  ✓ Inference successful! Got {len(preds)} frame predictions.")
        for i, pred in enumerate(preds):
            n_det = (pred['scores'] > 0.3).sum().item()
            print(f"    Frame {i}: {pred['scores'].shape[0]} queries, "
                  f"{n_det} detections with score > 0.3")
            # Check prediction keys
            pred_keys = list(pred.keys())
            print(f"    Keys: {pred_keys}")
    except Exception as e:
        print(f"  ✗ Inference FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All DSITTv2 smoke tests passed!")
    print("=" * 50)
    return True


if __name__ == '__main__':
    success = test_forward_pass()
    sys.exit(0 if success else 1)