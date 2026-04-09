#!/usr/bin/env python3
"""
DSITT Training Script.

Usage:
    python tools/train.py --config configs/dsitt_full.yaml --data_root data/rgbt_tiny --epochs 200
    python tools/train.py --config configs/dsitt_full.yaml --data_root data/rgbt_tiny --epochs 200 --amp

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

# Force unbuffered output for nohup/redirect logging
if not sys.stdout.isatty():
    import functools
    print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.dsitt import build_dsitt
from models.dsitt_v2 import build_dsitt_v2
from datasets.rgbt_tiny import build_rgbt_tiny_dataset
from tools.eval import evaluate


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
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision (fp16)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Dataloader num_workers')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
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
    use_amp: bool = False,
    scaler: GradScaler = None,
    lr_scheduler=None,
    warmup_iters: int = 0,
    accum_steps: int = 1,
) -> int:
    """Train for one epoch with optional gradient accumulation."""
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (frames, targets) in enumerate(dataloader):
        # Move to device — handle both single and dual modality
        if isinstance(frames[0], (list, tuple)):
            frames_rgb = [f[0].to(device) for f in frames]
            frames_ir = [f[1].to(device) for f in frames]
        else:
            frames_moved = [f.to(device) for f in frames]
            frames_rgb = frames_moved
            frames_ir = frames_moved

        targets_device = []
        for t in targets:
            targets_device.append({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            })

        # Forward + backward (with gradient accumulation)
        is_accum_step = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 < len(dataloader))

        try:
            if use_amp and scaler is not None:
                with autocast('cuda'):
                    if hasattr(model, 'dual_backbone'):
                        loss_dict = model(frames_rgb, frames_ir, targets_device)
                    else:
                        loss_dict = model(frames_rgb, targets_device)
                    loss = loss_dict['loss'] / accum_steps

                # Check for NaN/Inf — let scaler handle it properly
                if not torch.isfinite(loss):
                    print(f"  [WARNING] NaN/Inf loss at iter {batch_idx + 1}, skipping batch")
                    optimizer.zero_grad()
                    # Reset model tracking state to avoid corrupted queries
                    if hasattr(model, 'track_manager'):
                        model.track_manager.reset()
                    if hasattr(model, 'mtuq_manager'):
                        model.mtuq_manager.reset()
                    continue

                scaler.scale(loss).backward()
                if not is_accum_step:
                    if max_norm > 0:
                        scaler.unscale_(optimizer)
                        # Check for inf gradients (scaler will skip step if found)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        if not torch.isfinite(grad_norm):
                            print(f"  [WARNING] Inf grad norm at iter {batch_idx + 1}, scaler will skip")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if hasattr(model, 'dual_backbone'):
                    loss_dict = model(frames_rgb, frames_ir, targets_device)
                else:
                    loss_dict = model(frames_rgb, targets_device)
                loss = loss_dict['loss'] / accum_steps

                if not torch.isfinite(loss):
                    print(f"  [WARNING] NaN/Inf loss at iter {batch_idx + 1}, skipping batch")
                    optimizer.zero_grad()
                    if hasattr(model, 'track_manager'):
                        model.track_manager.reset()
                    if hasattr(model, 'mtuq_manager'):
                        model.mtuq_manager.reset()
                    continue

                loss.backward()
                if not is_accum_step:
                    if max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
        except (ValueError, RuntimeError) as e:
            print(f"  [WARNING] Error at iter {batch_idx + 1}: {e}, skipping batch")
            optimizer.zero_grad()
            if hasattr(model, 'track_manager'):
                model.track_manager.reset()
            if hasattr(model, 'mtuq_manager'):
                model.mtuq_manager.reset()
            continue

        # Step LR scheduler on each optimizer step (not each micro-batch)
        if not is_accum_step and lr_scheduler is not None:
            lr_scheduler.step()

        # Accumulate stats (use unscaled loss for logging)
        total_loss += loss.item() * accum_steps
        total_cls += loss_dict['loss_cls'].item() if isinstance(loss_dict['loss_cls'], torch.Tensor) else loss_dict['loss_cls']
        total_l1 += loss_dict['loss_l1'].item() if isinstance(loss_dict['loss_l1'], torch.Tensor) else loss_dict['loss_l1']
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
    ])

    # Override with command line args
    epochs = args.epochs or train_cfg.get('epochs', 200)
    lr = args.lr or train_cfg.get('base_lr', 2e-4)
    max_norm = train_cfg.get('clip_max_norm', 0.1)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"AMP: {'enabled' if args.amp else 'disabled'}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # Build model
    print("\n=== Building Model ===")
    model_version = config.get('model', {}).get('version', 'v1')
    if model_version == 'v2':
        print("Using DSITTv2 (MTUQ + MAD + SAS + Motion)")
        model = build_dsitt_v2(config)
    else:
        print("Using DSITTv1 (baseline)")
        model = build_dsitt(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {num_params:.1f}M (trainable: {num_trainable:.1f}M)")

    # Build dataset
    print("\n=== Building Dataset ===")
    data_root = args.data_root if not args.dummy else 'data/nonexistent'
    modality = data_cfg.get('modality', 'ir')
    num_workers = 0 if args.dummy else args.num_workers

    dataset, dataloader = build_rgbt_tiny_dataset(
        data_root=data_root,
        split='train',
        modality=modality,
        clip_length=2,
        batch_size=1,
        num_workers=num_workers,
    )

    # Build validation dataloader for best model tracking
    val_dataset, val_dataloader = build_rgbt_tiny_dataset(
        data_root=data_root,
        split='test',
        modality=modality,
        clip_length=2,
        batch_size=1,
        num_workers=0,
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

    # LR Scheduler with warmup
    from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR

    warmup_iters = train_cfg.get('warmup_iters', 1000)
    lr_drop = train_cfg.get('lr_drop_epoch', 100)
    iters_per_epoch = len(dataloader)

    def build_lr_scheduler(optimizer, warmup_iters, lr_drop, iters_per_epoch):
        """Build LR scheduler with correct step_size for current dataset."""
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters
        )
        main_scheduler = StepLR(
            optimizer, step_size=lr_drop * iters_per_epoch, gamma=0.1
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters],
        )

    lr_scheduler = build_lr_scheduler(optimizer, warmup_iters, lr_drop, iters_per_epoch)

    # AMP scaler
    scaler = GradScaler('cuda') if args.amp else None

    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Check if iters_per_epoch changed; if so, rebuild scheduler with correct step_size
        saved_iters = checkpoint.get('iters_per_epoch', None)
        if saved_iters is not None and saved_iters != iters_per_epoch:
            print(f"  [WARNING] iters_per_epoch changed: {saved_iters} -> {iters_per_epoch}")
            print(f"  Rebuilding LR scheduler with current iters_per_epoch")
            lr_scheduler = build_lr_scheduler(optimizer, warmup_iters, lr_drop, iters_per_epoch)
            # Fast-forward scheduler to match global_step
            saved_step = checkpoint.get('global_step', 0)
            for _ in range(saved_step):
                lr_scheduler.step()
        else:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resumed at epoch {start_epoch}")

    # Best model tracking
    best_mota = -float('inf')
    best_epoch = 0

    # Training loop
    print(f"\n=== Training for {epochs} epochs (from epoch {start_epoch}) ===")
    accum_steps = args.accum_steps
    print(f"LR: {lr}, Warmup: {warmup_iters} iters, LR drop at epoch {lr_drop}")
    print(f"Gradient accumulation: {accum_steps} steps (effective batch = {accum_steps})")
    print(f"Clip schedule: {clip_schedule}")
    print()

    optimizer.zero_grad()  # Initialize gradients for accumulation

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
            writer=writer, global_step=global_step,
            use_amp=args.amp, scaler=scaler,
            lr_scheduler=lr_scheduler, warmup_iters=warmup_iters,
            accum_steps=accum_steps,
        )

        # Save checkpoint + validate at save_freq epochs
        if epoch % args.save_freq == 0 or epoch == epochs:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'iters_per_epoch': iters_per_epoch,
                'config': config,
            }
            if scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
            save_path = os.path.join(
                args.output_dir, 'checkpoints', f'checkpoint_{epoch:04d}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f"  Saved checkpoint: {save_path}")

            # Validation for best model tracking
            print(f"  Running validation...")
            val_results = evaluate(model, val_dataloader, device, score_threshold=0.3)
            val_mota = val_results['MOTA']
            val_hota = val_results['HOTA']
            print(f"  Val MOTA: {val_mota:.4f}, HOTA: {val_hota:.4f}, "
                  f"IDF1: {val_results['IDF1']:.4f}")

            if writer is not None:
                writer.add_scalar('val/MOTA', val_mota, epoch)
                writer.add_scalar('val/HOTA', val_hota, epoch)
                writer.add_scalar('val/IDF1', val_results['IDF1'], epoch)

            # Save best model
            if val_mota > best_mota:
                best_mota = val_mota
                best_epoch = epoch
                best_path = os.path.join(
                    args.output_dir, 'checkpoints', 'checkpoint_best.pth'
                )
                torch.save(checkpoint, best_path)
                print(f"  ★ New best model! MOTA={best_mota:.4f} (epoch {epoch})")

    writer.close()
    print(f"\n=== Training Complete ===")
    print(f"Best MOTA: {best_mota:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()