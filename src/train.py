# ═══════════════════════════════════════════════
# train.py — Training loop
# ═══════════════════════════════════════════════

import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    GRAD_CLIP, MODEL_SAVE_PATH, DEVICE)
from src.losses import CombinedLoss
from src.evaluate import evaluate_model, print_metrics


def train_one_epoch(model, loader, optimizer,
                    criterion, edge_index, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels, dist_ids) in enumerate(loader):
        images   = images.to(device)
        labels   = labels.float().to(device)
        dist_ids = dist_ids.to(device)

        optimizer.zero_grad()
        preds = model(images, dist_ids, edge_index)
        loss  = criterion(preds.squeeze(), labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def train(model, train_loader, val_loader,
          edge_index, device=None):
    """
    Full training loop with:
    - CombinedLoss (Focal + Dice)
    - AdamW optimizer
    - Cosine annealing LR schedule
    - Best model checkpoint saving
    """
    if device is None:
        device = torch.device(
            DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model     = model.to(device)
    edge_index = edge_index.to(device)
    criterion  = CombinedLoss(weight_focal=0.6, weight_dice=0.4)
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_f1   = 0.0
    history   = []
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} "
              f"| LR: {scheduler.get_last_lr()[0]:.6f}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, edge_index, device)

        val_metrics, _, _ = evaluate_model(
            model, val_loader, edge_index, device)

        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f}")

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            **val_metrics
        })

        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'f1':          best_f1,
                'metrics':     val_metrics
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved "
                  f"(F1={best_f1:.4f})")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    return history