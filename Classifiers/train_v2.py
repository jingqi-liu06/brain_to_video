"""
Training script v2 with modular data pipeline.

This script uses the new EEGPreprocessor and EEGDataset classes for consistent
data processing between training and inference.

Usage:
------
    CUDA_VISIBLE_DEVICES=0 python Classifiers/train_v2.py \
        --raw_dir data/Preprocessing/Segmented_1000ms_sw \
        --subj_name sub3 \
        --category label_cluster \
        --model glmnet \
        --epochs 500

The key differences from train_classifier_mono.py:
1. Uses EEGPreprocessor for all data processing
2. Saves preprocessor state alongside model checkpoints
3. Cleaner separation of concerns
"""

import os
import sys
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Classifiers.data.dataset import EEGPreprocessor, EEGDataset, create_datasets
from Classifiers.modules.models import glmnet, eegnet, deepnet
from Classifiers.modules.utils import block_split

# Occipital electrode indices (for glmnet)
OCCIPITAL_IDX = list(range(50, 62))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    p = argparse.ArgumentParser(description="Train EEG classifier with modular data pipeline")
    
    # Data arguments
    p.add_argument("--raw_dir", default="data/Preprocessing/Segmented_1000ms_sw",
                   help="Directory containing .npy files")
    p.add_argument("--label_dir", default="data/Video/meta-info",
                   help="Directory containing label files")
    p.add_argument("--subj_name", default="sub3",
                   help="Subject name to process")
    
    # Task arguments
    p.add_argument("--category", default="label_cluster",
                   choices=["color", "color_binary", "face_apperance", "human_apperance",
                            "label_cluster", "label", "obj_number", "optical_flow_score"],
                   help="Classification category")
    p.add_argument("--cluster", type=int, default=None,
                   help="Cluster index (only for --category label)")
    
    # Model arguments
    p.add_argument("--model", choices=["glmnet", "eegnet", "deepnet"], default="glmnet",
                   help="Model architecture")
    
    # Training arguments
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--bs", type=int, default=100, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--scheduler", choices=["steplr", "reducelronplateau", "cosine"],
                   default="reducelronplateau")
    
    # Experiment arguments
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true",
                   help="Use random shuffle split instead of block split")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--use_wandb", action="store_true")
    
    return p.parse_args()


def format_labels(labels: np.ndarray, category: str) -> np.ndarray:
    """Format labels based on category type."""
    match category:
        case "color":
            return labels.astype(np.int64)
        case "face_appearance" | "human_appearance" | "label_cluster" | "face_apperance" | "human_apperance":
            return labels.astype(np.int64)
        case "color_binary":
            return (labels != 0).astype(np.int64)
        case "label" | "obj_number":
            labels = labels - 1
            return labels.astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(f"Unknown category: {category}")


def load_and_prepare_data(args):
    """Load raw EEG data and labels, prepare for training."""
    
    # Load raw EEG data
    raw_path = os.path.join(args.raw_dir, f"{args.subj_name}.npy")
    raw = np.load(raw_path)
    n_blocks, n_concepts, n_rep, n_win, C, T = raw.shape
    print(f"Loaded raw EEG: {raw.shape}")
    
    # Parse window duration from directory name
    duration_match = re.search(r"_(\d+)ms_", os.path.basename(args.raw_dir))
    if duration_match:
        win_sec = int(duration_match.group(1)) / 1000
    else:
        win_sec = T / 200  # Fallback: calculate from data
    print(f"Window duration: {win_sec}s")
    
    # Reshape raw data: (blocks, concepts*rep, windows, channels, time)
    raw = raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)
    
    # Load labels
    label_path = os.path.join(args.label_dir, f"All_video_{args.category}.npy")
    if args.category == "color_binary" and not os.path.exists(label_path):
        label_path = os.path.join(args.label_dir, "All_video_color.npy")
    labels_raw = np.load(label_path)
    
    # Repeat labels for repetitions
    if labels_raw.shape[1] == n_concepts:
        labels_raw = np.repeat(labels_raw[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
    
    # Create mask for valid samples
    mask_2d = (labels_raw != 0) if args.category == "color" else np.ones_like(labels_raw, bool)
    
    # Apply cluster filter if specified
    if args.cluster is not None:
        clusters = np.load(os.path.join(args.label_dir, "All_video_label_cluster.npy"))
        if clusters.shape[1] == n_concepts:
            clusters = np.repeat(clusters[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
        mask_2d &= (clusters == args.cluster)
    
    # Flatten and apply mask
    block_ids = np.repeat(np.arange(n_blocks), n_concepts * n_rep)
    mask_flat = mask_2d.reshape(-1)
    
    block_ids = block_ids[mask_flat]
    raw = raw.reshape(-1, n_win, C, T)[mask_flat]
    labels_flat = labels_raw.reshape(-1)[mask_flat] - (1 if args.category == "color" else 0)
    
    # Expand labels for windows
    labels = format_labels(
        np.repeat(labels_flat[:, None], n_win, axis=1),
        args.category
    )
    
    # Remap labels if using cluster filter
    if args.cluster is not None and args.category == "label":
        uniq = np.sort(np.unique(labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        labels = np.vectorize(mapping.get)(labels)
        print(f"Cluster {args.cluster}: remapped {uniq.tolist()} -> {list(mapping.values())}")
    
    # Flatten to sample level
    X_all = raw.reshape(-1, C, T)
    y_all = labels.reshape(-1)
    block_ids_win = np.repeat(block_ids, n_win)
    
    # Print label distribution
    unique_labels, counts = np.unique(y_all, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    
    return X_all, y_all, block_ids_win, n_blocks, C, T, win_sec


def create_data_splits(args, X_all, y_all, block_ids_win, n_blocks):
    """Create train/val/test splits."""
    
    if args.shuffle:
        # Random shuffle split
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(y_all))
        train_end = int(0.8 * len(idx))
        val_end = int(0.9 * len(idx))
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
    else:
        # Block-based split
        mode = "shuffle" if args.shuffle else "ordered"
        ckpt_seed_dir = os.path.join(
            args.save_dir, "mono", args.subj_name, mode, f"seed{args.seed}"
        )
        val_block, test_block = block_split(args.seed, n_blocks, ckpt_seed_dir)
        print(f"Validation block: {val_block}, Test block: {test_block}")
        
        train_mask = (block_ids_win != val_block) & (block_ids_win != test_block)
        val_mask = block_ids_win == val_block
        test_mask = block_ids_win == test_block
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
    
    return train_idx, val_idx, test_idx


def build_model(args, C, T, feat_dim, num_classes, device):
    """Build the model based on architecture choice."""
    
    if args.model == "glmnet":
        model = glmnet(OCCIPITAL_IDX, C=C, T=T, feat_dim=feat_dim, out_dim=num_classes)
    elif args.model == "eegnet":
        model = eegnet(out_dim=num_classes, C=C, T=T)
    elif args.model == "deepnet":
        model = deepnet(out_dim=num_classes, C=C, T=T)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model.to(device)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_samples += len(yb)
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            loss = criterion(logits, yb)
            
            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_samples += len(yb)
            
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb.cpu())
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    return total_loss / total_samples, total_correct / total_samples, preds, labels


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup checkpoint directory
    ckpt_name = args.category + (f"_cluster{args.cluster}" if args.cluster is not None else "")
    mode = "shuffle" if args.shuffle else "ordered"
    ckpt_dir = os.path.join(
        args.save_dir, "mono", args.subj_name, mode, f"seed{args.seed}", args.model, ckpt_name
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {ckpt_dir}")
    
    # Load and prepare data
    X_all, y_all, block_ids_win, n_blocks, C, T, win_sec = load_and_prepare_data(args)
    
    # Create train/val/test splits
    train_idx, val_idx, test_idx = create_data_splits(args, X_all, y_all, block_ids_win, n_blocks)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create datasets with preprocessing
    train_dataset, val_dataset, test_dataset, preprocessor = create_datasets(
        raw_data=X_all,
        labels=y_all,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_type=args.model,
        fs=200,
        win_sec=win_sec,
    )
    
    # Save preprocessor state
    preprocessor.save(ckpt_dir)
    print(f"Preprocessor saved to {ckpt_dir}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs)
    test_loader = DataLoader(test_dataset, batch_size=args.bs)
    
    # Determine feature dimension and number of classes
    sample_x, _ = train_dataset[0]
    feat_dim = sample_x.shape[-1] - T if args.model == "glmnet" else 0
    num_classes = len(np.unique(y_all))
    print(f"Feature dimension: {feat_dim}, Number of classes: {num_classes}")
    
    # Build model
    model = build_model(args, C, T, feat_dim, num_classes, device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=10, min_lr=args.min_lr)
    elif args.scheduler == "steplr":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs // 2, eta_min=args.min_lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb if requested
    if args.use_wandb and HAS_WANDB:
        run_name = f"{args.subj_name}_{ckpt_name}_{mode}_v2"
        wandb.init(project="eeg2video-classifiersv4-mono", name=run_name, config=vars(args))
        wandb.watch(model, log="all")
    
    # Training loop
    best_val_acc = 0.0
    model_path = os.path.join(ckpt_dir, f"{args.model}_best.pt")
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        if args.scheduler == "reducelronplateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Enforce minimum learning rate
        for pg in optimizer.param_groups:
            if pg["lr"] < args.min_lr:
                pg["lr"] = args.min_lr
        
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            tqdm.write(f"Epoch {epoch:05d}: LR reduced to {new_lr:.4e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            if args.model == "glmnet":
                torch.save(model.raw_global.state_dict(), os.path.join(ckpt_dir, "shallownet.pt"))
                torch.save(model.freq_local.state_dict(), os.path.join(ckpt_dir, "mlpnet.pt"))
            tqdm.write(f"Epoch {epoch:05d}: New best val_acc={val_acc:.4f}")
        
        # Log to wandb
        if args.use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": new_lr,
            })
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("\nNormalized Confusion Matrix (%):")
    print(confusion_matrix(labels, preds, normalize="true") * 100)
    
    if args.use_wandb and HAS_WANDB:
        class_names = [str(c) for c in np.unique(labels)]
        cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=class_names)
        wandb.log({"test/acc": test_acc, "test/confusion_matrix": cm_plot})
        wandb.finish()
    
    print(f"\nTraining complete. Best model saved to: {model_path}")


if __name__ == "__main__":
    main()
