import os, argparse, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import pickle
from sklearn.metrics import confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Classifiers.modules.utils import standard_scale_features, compute_raw_stats, normalize_raw, subject_split
from Classifiers.modules.models import mlpnet, glmnet, deepnet, eegnet


# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-classifiersv4"  # <‑‑ change if you need another project

# ------------------------------ constants ---------------------------------
OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels


# ------------------------------ utils -------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="./data/Preprocessing/Segmented_1000ms_sw", help="directory with .npy files")
    p.add_argument("--label_dir", default="./data/meta_info", help="Label file")
    p.add_argument(
        "--category",
        default="label_cluster",
        choices=[
            "color",
            "color_binary",
            "face_appearance",
            "human_appearance",
            "label_cluster",
            "label",
            "obj_number",
            "optical_flow_score",
        ],
        help="Label file",
    )
    p.add_argument("--save_dir", default="./Classifiers/checkpoints")
    p.add_argument(
        "--cluster",
        type=int,
        help="Cluster index to filter labels (only valid when --category label)",
    )
    p.add_argument("--model", choices=["glmnet","eegnet","deepnet"], default="glmnet")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the scheduler")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=["steplr", "reducelronplateau", "cosine"],
        default="reducelronplateau",
        help="Type of learning rate scheduler",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--n_subj", type=int, default=15, help="Number of subjects to sample for train/val when not provided")
    p.add_argument("--seed", type=int, default=0, help="Random seed for subject sampling")
    p.add_argument(
        "--cache_dir",
        default="./Classifiers/cache",
        help="Directory where per-subject features are stored",
    )
    return p.parse_args()


def format_labels(labels: np.ndarray, category: str) -> np.ndarray:
    match category:
        case "color":
            return labels.astype(np.int64)
        case "face_appearance" | "human_appearance" | "label_cluster":
            return labels.astype(np.int64)
        case "color_binary":
            # Collapse all non-zero colors into the dominant color class
            return (labels != 0).astype(np.int64)
        case "label" | "obj_number":
            labels = labels - 1
            return labels.astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(
                f"Unknown category: {category}. Must be one of: color, color_binary, face_appearance, human_appearance, label_cluster, label, obj_number, optical_flow_score."
            )


# ------------------------------ main -------------------------------------
def main():
    args = parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ 1
    # Subject discovery and split
    # ---------------------------------------------------------------------
    all_subj = sorted(f[:-4] for f in os.listdir(args.raw_dir)
                      if f.startswith("sub") and f.endswith(".npy"))
    if args.n_subj > len(all_subj):
        raise ValueError("Not enough subject files in raw_dir")

    ckpt_seed_dir = os.path.join(args.save_dir, "multi", str(args.seed))
    train_subj, val_subj, test_subj = subject_split(
        args.seed, all_subj, ckpt_seed_dir=ckpt_seed_dir, n_select=args.n_subj
    )
    print("Training subjects:", train_subj)
    print("Validation subjects:", val_subj)
    print("Test subjects:",      test_subj)

    # ------------------------------------------------------------------ 2
    # Persistent paths & global shapes
    # ---------------------------------------------------------------------
    ckpt_name = args.category + (f"_cluster{args.cluster}" if args.cluster is not None else "")
    ckpt_dir  = os.path.join(args.save_dir, "multi", str(args.seed), args.model, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"{args.model}_best.pt")
    stats_path = os.path.join(ckpt_dir, "raw_stats.npz")
    if args.model == "glmnet":
        shallownet_path = os.path.join(ckpt_dir, "shallownet.pt")
        mlpnet_path = os.path.join(ckpt_dir, "mlpnet.pt")
        scaler_path = os.path.join(ckpt_dir, "scaler.pkl")

    # infer basic dims from one file
    sample_raw = np.load(os.path.join(args.raw_dir, f"{train_subj[0]}.npy"))
    n_blocks, n_concepts, n_rep, n_win, C, T = sample_raw.shape

    duration_ms = int(re.search(r"_(\d+)ms_", os.path.basename(args.raw_dir)).group(1)) / 1000 \
                  if args.model == "glmnet" else None

    # ---------------------------------------------------------------------
    # 3) Labels & masks
    # ---------------------------------------------------------------------
    label_path = os.path.join(args.label_dir, f"All_video_{args.category}.npy")
    if args.category == "color_binary" and not os.path.exists(label_path):
        label_path = os.path.join(args.label_dir, "All_video_color.npy")

    labels_raw = np.load(label_path)
    if labels_raw.shape[1] == n_concepts:
        labels_raw = np.repeat(labels_raw[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)

    mask_2d = (labels_raw != 0) if args.category == "color" else np.ones_like(labels_raw, bool)

    if args.cluster is not None:
        clusters = np.load(os.path.join(args.label_dir, "All_video_label_cluster.npy"))
        if clusters.shape[1] == n_concepts:
            clusters = np.repeat(clusters[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
        mask_2d &= clusters == args.cluster

    mask_flat = mask_2d.reshape(-1)
    labels_flat = labels_raw.reshape(-1)[mask_flat] - (1 if args.category == "color" else 0)

    def expand_labels_flat(labels_1d: np.ndarray, n_win: int) -> np.ndarray:
        return np.repeat(labels_1d[:, None], n_win, axis=1)

    base_labels = format_labels(expand_labels_flat(labels_flat, n_win), args.category)

    if args.cluster is not None and args.category == "label":
        uniq = np.sort(np.unique(base_labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        base_labels = np.vectorize(mapping.get)(base_labels)
        print(f"Cluster {args.cluster}: mapping original labels {uniq.tolist()} -> {list(mapping.values())}")

    unique_labels, counts_labels = np.unique(base_labels, return_counts=True)
    num_unique_labels = len(unique_labels)
    label_final_distribution = {int(u): int(c) for u, c in zip(unique_labels, counts_labels)}
    print("Label distribution after formating:", label_final_distribution)
    
    # ---------------------------------------------------------------------
    # 4) Helpers: load one subject   ➜  returns flattened windows + labels + ids
    # ---------------------------------------------------------------------
    def _load_or_cache_raw(name: str) -> np.ndarray:
        """
        Load the subject’s flattened raw windows from cache if available;
        otherwise build them from the source file, save to cache, and return.

        Returns
        -------
        np.ndarray
            Shape (samples·W, C, T)  — raw EEG windows, still un-normalised.
        """
        raw_cache = os.path.join(args.cache_dir, f"{name}_raw.npy")

        if os.path.exists(raw_cache):
            # Use memory mapping for lazy loading; avoids copying everything into RAM
            return np.load(raw_cache, mmap_mode="r")

        # --- Build from original file ---
        raw = np.load(os.path.join(args.raw_dir, f"{name}.npy"))
        raw = raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)
        raw = raw.reshape(-1, n_win, C, T)[mask_flat]          # (samples, W, C, T)
        X_flat = raw.reshape(-1, C, T).astype(np.float32)       # (samples·W, C, T)

        # Save to cache for future runs
        np.save(raw_cache, X_flat)
        return X_flat


    def load_subject(name: str):
        """
        Returns (X_flat, y_flat, ids) for a given subject.

        Parameters
        ----------
        name : str
            Subject identifier, e.g. "sub1".

        Returns
        -------
        X_flat : np.ndarray
            Raw windows, shape (samples·W, C, T), still un-normalised.
        y_flat : np.ndarray
            Corresponding labels, shape (samples·W,).
        ids : np.ndarray
            Subject ID repeated for each sample, shape (samples·W,).
        """
        X_flat = _load_or_cache_raw(name)

        # Labels are identical across subjects and already masked by `mask_flat`
        y_flat = base_labels.reshape(-1)                        # (samples·W,)
        return X_flat, y_flat

    def concat_subjects(names: list[str]):
        X_list, y_list, slice_map = [], [], {}
        offset = 0
        for n in names:
            xf, yf = load_subject(n)
            nsamp       = len(yf)
            slice_map[n] = slice(offset, offset + nsamp)    # keep track for caching
            offset += nsamp

            X_list.append(xf)
            y_list.append(yf)
        if not X_list:                                       # empty split
            return np.empty((0, C, T)), np.empty((0,), np.int64), {}
        return (np.concatenate(X_list),
                np.concatenate(y_list),
                slice_map)

    print("Loading & concatenating subjects…")
    X_train, y_train, slice_train = concat_subjects(train_subj)
    X_val,   y_val,   slice_val   = concat_subjects(val_subj)
    X_test,  y_test,  slice_test  = concat_subjects(test_subj)
    
    X_train = X_train.reshape(-1, C, T)      # (N_train, C, T)
    X_val   = X_val.reshape(-1, C, T)
    X_test  = X_test.reshape(-1, C, T)

    # ------------------------------------------------------------------ 5
    # Feature extraction with per-subject cache
    # ---------------------------------------------------------------------
    if args.model == "glmnet":
        def extract_or_load(X_all, sl_map):
            """Return feature matrix aligned to X_all, caching per subject."""
            parts, feat_dim = [], None
            for sid in sl_map:  # guaranteed order == concat order
                sl   = sl_map[sid]
                fpth = os.path.join(
                    args.cache_dir, f"{sid}_{duration_ms*1000:.1f}_feat.npy"
                )
                if os.path.exists(fpth):
                    feat = np.load(fpth)
                else:
                    feat = mlpnet.compute_features(X_all[sl], win_sec=duration_ms)
                    np.save(fpth, feat)
                if feat_dim is None:
                    feat_dim = feat.shape[2]
                parts.append(feat)
            return np.concatenate(parts), feat_dim

        print("→ Features (train)…")
        F_train, feat_dim = extract_or_load(X_train, slice_train)
        print("→ Features (val)…")
        F_val, _          = extract_or_load(X_val,   slice_val)
        print("→ Features (test)…")
        F_test, _         = extract_or_load(X_test,  slice_test)
    else:
        F_train = F_val = F_test = None
        feat_dim = 0   # placeholder

    # ------------------------------------------------------------------ 6
    # Normalisation & scaling
    # ---------------------------------------------------------------------
    raw_mean, raw_std = compute_raw_stats(X_train)
    X_train = normalize_raw(X_train, raw_mean, raw_std)
    X_val   = normalize_raw(X_val,   raw_mean, raw_std)
    X_test  = normalize_raw(X_test,  raw_mean, raw_std)

    if args.model == "glmnet":
        F_train_scaled, scaler = standard_scale_features(F_train, return_scaler=True)
        F_val_scaled           = standard_scale_features(F_val,  scaler=scaler)
        F_test_scaled          = standard_scale_features(F_test, scaler=scaler)

        X_train = np.concatenate([X_train, F_train_scaled], axis=2)
        X_val   = np.concatenate([X_val,   F_val_scaled],   axis=2)
        X_test  = np.concatenate([X_test,  F_test_scaled],  axis=2)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
    np.savez(stats_path, mean=raw_mean, std=raw_std)

    # ---------------------------------------------------------------------
    # 7) TensorDatasets & DataLoaders
    # ---------------------------------------------------------------------
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),   # (N,1,C,feat)
        torch.tensor(y_train),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val),
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test),
    )

    dl_train = DataLoader(ds_train, args.bs, shuffle=True)
    dl_val   = DataLoader(ds_val,   args.bs)
    dl_test  = DataLoader(ds_test,  args.bs)
    
    if args.model == "glmnet":
        model = glmnet(OCCIPITAL_IDX,C=C,T=T,feat_dim=feat_dim,out_dim=num_unique_labels).to(device)
    elif args.model == "eegnet" or args.model == "deepnet":
        model_cls = deepnet if args.model == "deepnet" else eegnet
        model = model_cls(out_dim=num_unique_labels, C=C, T=T).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}. Must be one of: glmnet, eegnet, deepnet.")
    
    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.8, patience=10, verbose=False, min_lr=args.min_lr)
    elif args.scheduler == "steplr":
        scheduler = StepLR(opt, step_size=10, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs // 2, eta_min=args.min_lr)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.init(project=PROJECT_NAME, name=f"seed_{args.seed}_{ckpt_name}_{args.model}", config=vars(args))
        wandb.watch(model, log="all")

    best_val = 0.0
    for ep in tqdm(range(1, args.epochs + 1)):
        model.train()
        tl = ta = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            ta += (pred.argmax(1) == yb).sum().item()
        train_acc = ta / len(ds_train)

        model.eval()
        vl = va = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = criterion(pred, yb)
                vl += vloss.item() * len(yb)
                va += (pred.argmax(1) == yb).sum().item()
        val_acc = va / len(ds_val)
        val_loss = vl / len(ds_val)
        if scheduler is not None:
            old_lr = opt.param_groups[0]["lr"]
            if args.scheduler == "reducelronplateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
            for pg in opt.param_groups:
                if pg["lr"] < args.min_lr:
                    pg["lr"] = args.min_lr
            new_lr = opt.param_groups[0]["lr"]
            if new_lr < old_lr:
                tqdm.write(f"Epoch {ep:05d}: reducing learning rate of group 0 to {new_lr:.4e}.")
        current_lr = opt.param_groups[0]["lr"]

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            if args.model == "glmnet":
                torch.save(model.raw_global.state_dict(), shallownet_path)
                torch.save(model.freq_local.state_dict(), mlpnet_path)
            tqdm.write(f"New best model saved at epoch {ep} with val_acc={val_acc:.3f}")

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": ep,
                    "train/acc": train_acc,
                    "val/acc": val_acc,
                    "train/loss": tl / len(ds_train),
                    "val/loss": val_loss,
                    "lr": current_lr,
                }
            )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    test_acc = 0
    preds, labels_test = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred_labels = out.argmax(1)
            test_acc += (pred_labels == yb).sum().item()
            preds.append(pred_labels.cpu())
            labels_test.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    labels_test = torch.cat(labels_test).numpy()
    cm = confusion_matrix(labels_test, preds)
    cm_percent = confusion_matrix(labels_test, preds, normalize="true") * 100
    test_acc /= len(ds_test)
    print(f"Test accuracy = {test_acc:.3f}")
    print("Confusion matrix:\n", cm)
    print("Weighted confusion matrix (%)\n", cm_percent)
    if args.use_wandb:
        class_names = [str(c) for c in np.unique(labels_test)]
        cm_plot = wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels_test,
            preds=preds,
            class_names=class_names,
        )
        table_columns = ["true"] + class_names
        cm_table = wandb.Table(columns=table_columns)
        for idx, row in enumerate(cm_percent):
            cm_table.add_data(class_names[idx], *row.tolist())
        wandb.log(
            {
                "test/acc": test_acc,
                "test/confusion_matrix": cm_plot,
                "test/confusion_matrix_percent": cm_table,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
