"""Generate embeddings or logits from EEG windows using pre-trained models."""

import os
import sys
import torch
import numpy as np
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from Classifiers.modules.utils import (
    standard_scale_features,
    normalize_raw,
    load_scaler,
    load_raw_stats,
)
from Classifiers.modules.models import mlpnet, glmnet, eegnet, deepnet


OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels


def inf_model(model, model_type, raw_sw, stats, scaler=None, device="cuda"):
    """Infer embeddings or logits for all windows."""

    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    raw_flat = normalize_raw(raw_flat, stats[0], stats[1])

    if model_type == "glmnet":
        feat_flat = mlpnet.compute_features(raw_flat)
        feat_scaled = standard_scale_features(feat_flat, scaler=scaler)
        input_flat = [np.concatenate([r, f], axis=-1) for r, f in zip(raw_flat, feat_scaled)]
    else:
        input_flat = raw_flat

    outputs = []
    with torch.no_grad():
        for x in input_flat:
            # Add batch and channel dimensions for convolutional models
            t = (
                torch.tensor(x, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(1)
                .to(device)
            )
            if model_type == "glmnet":
                out = model(t, return_features=True)
            else:
                out = model(t)
            outputs.append(out.squeeze(0).cpu().numpy())

    return np.stack(outputs)

# --- Main generation loop ---
def generate_all_embeddings(
    raw_dir,
    ckpt_path,
    output_dir,
    subject_prefix,
    model_type="glmnet",
    device="cuda",
):
    """Run inference for all subjects matching the prefix."""

    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(ckpt_path, "raw_stats.npz")
    scaler_path = os.path.join(ckpt_path, "scaler.pkl")
    model_path = os.path.join(ckpt_path, f"{model_type}_best.pt")

    stats = load_raw_stats(stats_path)
    scaler = load_scaler(scaler_path) if model_type == "glmnet" else None

    for fname in os.listdir(raw_dir):
        if not (fname.endswith('.npy') and fname.startswith(subject_prefix)):
            continue
        print(f"Processing {fname}...")
        subj = os.path.splitext(fname)[0]

        # load pre-segmented windows
        RAW_SW = np.load(os.path.join(raw_dir, fname))
        # expect shape: (7, 40, 5, 7, 62, T)
        time_len = RAW_SW.shape[-1]
        num_channels = RAW_SW.shape[-2]
        state = torch.load(model_path, map_location=device)
        if model_type == "glmnet":
            ckpt_time_len = glmnet.infer_time_len(state)
            if ckpt_time_len != time_len:
                raise ValueError(
                    f"EEG window length {time_len} does not match checkpoint (expected {ckpt_time_len})."
                )
            out_dim = glmnet.infer_out_dim(state)
            feat_dim = glmnet.infer_feat_dim(state, len(OCCIPITAL_IDX))
            model = glmnet(
                OCCIPITAL_IDX,
                C=num_channels,
                T=time_len,
                feat_dim=feat_dim,
                out_dim=out_dim,
            )
        else:
            out_dim = state["out.weight"].shape[0]
            model_cls = eegnet if model_type == "eegnet" else deepnet
            model = model_cls(out_dim=out_dim, C=num_channels, T=time_len)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        embeddings = inf_model(model, model_type, RAW_SW, stats, scaler, device)
        
        out_path = os.path.join(output_dir, f"{subj}.npy")
        np.save(out_path, embeddings)
        print(f"Saved embeddings for {subj}, shape {embeddings.shape}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', default="./data/Preprocessing/Segmented_500ms_sw", help='directory of pre-windowed raw EEG .npy files')
    parser.add_argument('--subject_prefix', default='sub3', help='prefix of subject files to process')
    parser.add_argument('--checkpoint_path', help='path to model checkpoint')
    parser.add_argument('--train_mode', choices=['ordered', 'shuffle'], default='ordered', help='training mode for mono model')
    parser.add_argument('--seed', type=int, default=0, help='Training seed')
    parser.add_argument('--model', choices=['glmnet', 'eegnet', 'deepnet'], default='glmnet', help='Model type')
    parser.add_argument('--output_dir', default="./data/eeg_segments", help='where to save projected embeddings')
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(
            "./Classifiers/checkpoints",
            "mono",
            args.subject_prefix,
            args.train_mode,
            str(args.seed),
            args.model,
            "label_cluster",
        )

    generate_all_embeddings(
        args.raw_dir,
        args.checkpoint_path,
        args.output_dir,
        args.subject_prefix,
        args.model,
    )
