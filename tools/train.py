#!/usr/bin/env python3
"""
DSITT Training Script.

Usage:
    python tools/train.py --config configs/dsitt_base.yaml
    python tools/train.py --config configs/dsitt_base.yaml --data_root /path/to/rgbt_tiny

For development without dataset:
    python tools/train.py --dummy --epochs 5 --print_freq 1
"""

import sys
import os
import argparse
import time
import yaml
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.dsitt import build_dsitt
from datasets.rgbt_tiny import build_rgbt_tiny_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='DSITT Training')
    parser.add_argument('--config', type=str, default='configs/dsitt_base.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='data/rgbt_tiny',
                        help='Path to RGBT-Tiny dataset root')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy data for development testing')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency (iterations)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"[WARNING] Config file not found: {config_path}, using defaults")
        return {}


def get_clip_length_for_epoch(epoch: int, schedule: list) -> int:
    """Get clip length based on training schedule."""
    clip_length = 2
    for item in schedule:
        if epoch >= item['epoch']:
            clip_length = item['clip_length']
    return clip_length


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.1,
    print_freq: int = 50,
    writer: SummaryWriter = None,
    global_step: int = 0,
) -> int:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (frames, targets) in enumerate(dataloader):
        # Move to device
        frames = [f.to(device) for f in frames]
        targets_device = []
        for t in targets:
            targets_device.append({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            })

        # Forward
        loss_dict = model(frames, targets_device)

        loss = loss_dict['loss']

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        total_cls += loss_dict['loss_cls'].item() if isinstance(loss_dict['loss_cls'], torch.Tensor) else loss_dict['loss_cls']
        total_l1 += loss_dict['loss_l1'].item() if isinstance(loss_dict['loss_l1'], torch.Tensor) else loss_dict['loss_l1']
        # Support both GIoU and NWD loss keys
        box_loss_key = 'loss_nwd' if 'loss_nwd' in loss_dict else 'loss_giou'
        box_loss_val = loss_dict[box_loss_key]
        total_giou += box_loss_val.item() if isinstance(box_loss_val, torch.Tensor) else box_loss_val
        num_batches += 1
        global_step += 1

        # Logging
        if (batch_idx + 1) % print_freq == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_batches
            box_loss_name = 'nwd' if 'loss_nwd' in loss_dict else 'giou'
            print(f"  Epoch [{epoch}] Iter [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                  f"cls: {loss_dict['loss_cls']:.4f} "
                  f"l1: {loss_dict['loss_l1']:.4f} "
                  f"{box_loss_name}: {box_loss_val:.4f} "
                  f"Time: {elapsed:.1f}s")

            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/loss_cls', total_cls / num_batches, global_step)
                writer.add_scalar('train/loss_l1', total_l1 / num_batches, global_step)
                writer.add_scalar(f'train/loss_{box_loss_name}', total_giou / num_batches, global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    avg_loss = total_loss / max(num_batches, 1)
    print(f"  Epoch [{epoch}] Complete. Avg Loss: {avg_loss:.4f}")

    return global_step


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    train_cfg = config.get('train', {})
    data_cfg = config.get('data', {})
    clip_schedule = config.get('clip_schedule', [
        {'epoch': 1, 'clip_length': 2},
        {'epoch': 50, 'clip_length': 3},
        {'epoch': 90, 'clip_length': 4},
        {'epoch': 150, 'clip_length': 5},
    ])

    # Override with command line args
    epochs = args.epochs or train_cfg.get('epochs', 200)
    lr = args.lr or train_cfg.get('base_lr', 2e-4)
    max_norm = train_cfg.get('clip_max_norm', 0.1)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # Build model
    print("\n=== Building Model ===")
    model = build_dsitt(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {num_params:.1f}M (trainable: {num_trainable:.1f}M)")

    # Build dataset
    print("\n=== Building Dataset ===")
    data_root = args.data_root if not args.dummy else 'data/nonexistent'
    modality = data_cfg.get('modality', 'ir')

    dataset, dataloader = build_rgbt_tiny_dataset(
        data_root=data_root,
        split='train',
        modality=modality,
        clip_length=2,
        batch_size=1,
        num_workers=0 if args.dummy else 4,
    )

    # Optimizer
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * train_cfg.get('backbone_lr_factor', 0.1)},
        {'params': other_params, 'lr': lr},
    ], weight_decay=train_cfg.get('weight_decay', 1e-4))

    # LR Scheduler
    lr_drop = train_cfg.get('lr_drop_epoch', 100)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop, gamma=0.1)

    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    print(f"\n=== Training for {epochs} epochs ===")
    print(f"LR: {lr}, LR drop at epoch {lr_drop}")
    print(f"Clip schedule: {clip_schedule}")
    print()

    for epoch in range(start_epoch, epochs + 1):
        # Update clip length based on schedule
        clip_length = get_clip_length_for_epoch(epoch, clip_schedule)
        dataset.set_clip_length(clip_length)

        print(f"Epoch {epoch}/{epochs} (clip_length={clip_length}, "
              f"lr={optimizer.param_groups[1]['lr']:.2e})")

        # Train one epoch
        global_step = train_one_epoch(
            model, dataloader, optimizer, device, epoch,
            max_norm=max_norm, print_freq=args.print_freq,
            writer=writer, global_step=global_step
        )

        # Step LR scheduler
        lr_scheduler.step()

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == epochs:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'config': config,
            }
            save_path = os.path.join(
                args.output_dir, 'checkpoints', f'checkpoint_{epoch:04d}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f"  Saved checkpoint: {save_path}")

    writer.close()
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()