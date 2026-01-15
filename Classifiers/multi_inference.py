# -*- coding: utf-8 -*-
"""Run multiple EEG classification models on a single EEG example.

This script loads several checkpoints and converts their
predictions into text using ``label_mappings.json``.  For each model we
evaluate all seven windows of the EEG sample and keep the label that
appears most often.  The textual outputs are concatenated to form a
descriptive phrase and a confidence score is reported for each label.
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

from modules.utils import (
    load_scaler,
    load_raw_stats,
    normalize_raw,
    standard_scale_features,
)
from modules.models import mlpnet, glmnet, eegnet, deepnet


OCCIPITAL_IDX = list(range(50, 62))

# Mapping from label cluster index to the range of original label IDs (inclusive)
# used during cluster-specific training.
CLUSTER_RANGES = [
    (1, 7),   # Land Animal
    (8, 11),  # Water Animal
    (12, 14), # Plant
    (15, 18), # Exercise
    (19, 21), # Human
    (22, 27), # Natural Scene
    (28, 32), # Food
    (33, 35), # Musical
    (36, 40), # Transportation
]

def load_label_mappings(path: str) -> Dict[str, Dict[int, str]]:
    """Load textual descriptions for every label category."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings: Dict[str, Dict[int, str]] = {}
    for cat, mapping in data.items():
        mappings[cat] = {int(k): str(v) for k, v in mapping.items()}
    return mappings


def prepare_input(eeg: np.ndarray, stats, scaler, model_type: str, fs: int = 200) -> torch.Tensor:
    """Normalize raw EEG and optionally compute features."""
    # Keep a copy of un-normalized raw data for feature extraction
    raw_orig = eeg[np.newaxis, ...].astype(np.float32)
    
    # Normalize raw for the spatial/temporal branch (Shallownet)
    # print(f"Raw Input Stats: Mean={raw.mean():.4f}, Std={raw.std():.4f}")
    # print(f"Stats loaded: Mean={stats[0].mean():.4f}, Std={stats[1].mean():.4f}")
    raw_norm = normalize_raw(raw_orig, stats[0], stats[1])
    # print(f"Normalized Stats: Mean={raw.mean():.4f}, Std={raw.std():.4f}")

    if model_type == "glmnet":
        # Calculate window duration in seconds based on input length
        # raw shape: (1, C, T)
        time_len = raw_orig.shape[-1]
        win_sec = time_len / fs
        
        # EXTRACT FEATURES FROM ORIGINAL (UN-NORMALIZED) DATA
        # This matches training logic in train_classifier_mono.py
        feat = mlpnet.compute_features(raw_orig, fs=fs, win_sec=win_sec)
        
        # print(f"Feature Stats (Before Scale): Mean={feat.mean():.4f}, Std={feat.std():.4f}")
        # if scaler is not None:
        #      print(f"Scaler Mean: {scaler.mean_.mean():.4f}, Scaler Var: {scaler.var_.mean():.4f}")
             
        feat = standard_scale_features(feat, scaler=scaler)
        # print(f"Feature Stats (After Scale): Mean={feat.mean():.4f}, Std={feat.std():.4f}")
        
        x = np.concatenate([raw_norm, feat], axis=-1)
    else:
        x = raw_norm
    # Return tensor with batch and channel dimensions
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)


def load_model(
    ckpt_dir: str, channels: int, time_len: int, device: str, model_type: str
) -> tuple[glmnet, any, tuple[np.ndarray, np.ndarray]]:
    """Load model with optional scaler and raw statistics."""

    stats = load_raw_stats(os.path.join(ckpt_dir, "raw_stats.npz"))
    if model_type == "glmnet":
        scaler = load_scaler(os.path.join(ckpt_dir, "scaler.pkl"))
        model_path = os.path.join(ckpt_dir, "glmnet_best.pt")
        state = torch.load(model_path, map_location=device)
        ckpt_time_len = glmnet.infer_time_len(state)
        if ckpt_time_len != time_len:
            raise ValueError(
                f"EEG window length {time_len} does not match checkpoint (expected {ckpt_time_len})."
            )
        out_dim = glmnet.infer_out_dim(state)
        feat_dim = glmnet.infer_feat_dim(state, len(OCCIPITAL_IDX))
        model = glmnet(
            OCCIPITAL_IDX,
            C=channels,
            T=time_len,
            feat_dim=feat_dim,
            out_dim=out_dim,
        )
    else:
        scaler = None
        model_path = os.path.join(ckpt_dir, f"{model_type}_best.pt")
        state = torch.load(model_path, map_location=device)
        out_dim = state["out.weight"].shape[0]
        model_cls = eegnet if model_type == "eegnet" else deepnet
        model = model_cls(out_dim=out_dim, C=channels, T=time_len)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, scaler, stats

def index_to_text(
    category: str,
    idx: int,
    label_map: Dict[str, Dict[int, str]],
    cluster_idx: int | None = None,
) -> str:
    """Convert predicted class index to a text label."""
    if category == "label" and cluster_idx is not None:
        start, _ = CLUSTER_RANGES[cluster_idx]
        idx = start + idx
    elif category in {"label", "obj_number"}:
        idx += 1
    return label_map.get(category, {}).get(idx, str(idx))


def majority_vote(
    eeg: np.ndarray,
    model: torch.nn.Module,
    scaler,
    stats,
    device: str,
    model_type: str,
) -> Tuple[int, float]:
    """Return the most common prediction index and its confidence."""

    preds = []
    for win in eeg:
        x = prepare_input(win, stats, scaler, model_type, fs=200).to(device)
        with torch.no_grad():
            logits = model(x)
            # print(f"Logits for model {model_type}: {logits.cpu().numpy()}")
            preds.append(int(logits.argmax(dim=-1).item()))
    values, counts = np.unique(preds, return_counts=True)
    best_idx = counts.argmax()
    majority = int(values[best_idx])
    conf = counts[best_idx] / len(preds)
    return majority, conf


def infer_description(
    eeg: np.ndarray,
    models: Dict[str, torch.nn.Module],
    scalers: Dict[str, any],
    stats: Dict[str, tuple],
    device: str,
    label_map: Dict[str, Dict[int, str]],
    model_type: str,
) -> Tuple[str, List[float]]:
    """Generate a descriptive phrase and confidences for one EEG sample."""

    confidences: List[float] = []
    # Descriptive pieces collected from each category
    color_desc = None
    face_desc = None
    human_desc = None
    obj_desc = None
    flow_desc = None

    if "color_binary" in models:
        idx, conf = majority_vote(
            eeg,
            models["color_binary"],
            scalers["color_binary"],
            stats["color_binary"],
            device,
            model_type,
        )
        confidences.append(conf)
        if idx == 1:
            if "color" not in models:
                raise FileNotFoundError(
                    "Color checkpoint required when dominant color is predicted"
                )
            idx_col, conf_col = majority_vote(
                eeg,
                models["color"],
                scalers["color"],
                stats["color"],
                device,
                model_type,
            )
            color_desc = f"with dominant color {index_to_text('color', idx_col, label_map)}"
            confidences.append(conf_col)
        else:
            color_desc = index_to_text("color_binary", idx, label_map)

    if "face_apperance" in models:
        idx, conf = majority_vote(
            eeg,
            models["face_apperance"],
            scalers["face_apperance"],
            stats["face_apperance"],
            device,
            model_type,
        )
        face_desc = index_to_text("face_apperance", idx, label_map)
        confidences.append(conf)

    if "human_apperance" in models:
        idx, conf = majority_vote(
            eeg,
            models["human_apperance"],
            scalers["human_apperance"],
            stats["human_apperance"],
            device,
            model_type,
        )
        human_desc = index_to_text("human_apperance", idx, label_map)
        confidences.append(conf)

    idx_cluster, conf_cluster = majority_vote(
        eeg,
        models["label_cluster"],
        scalers["label_cluster"],
        stats["label_cluster"],
        device,
        model_type,
    )
    cluster_text = index_to_text("label_cluster", idx_cluster, label_map)
    confidences.append(conf_cluster)

    label_cat = f"label_cluster{idx_cluster}"
    idx_label, conf_label = majority_vote(
        eeg,
        models[label_cat],
        scalers[label_cat],
        stats[label_cat],
        device,
        model_type,
    )
    label_text = index_to_text("label", idx_label, label_map, cluster_idx=idx_cluster)
    confidences.append(conf_label)

    if "obj_number" in models:
        idx, conf = majority_vote(
            eeg,
            models["obj_number"],
            scalers["obj_number"],
            stats["obj_number"],
            device,
            model_type,
        )
        obj_desc = index_to_text("obj_number", idx, label_map)
        confidences.append(conf)

    if "optical_flow_score" in models:
        idx, conf = majority_vote(
            eeg,
            models["optical_flow_score"],
            scalers["optical_flow_score"],
            stats["optical_flow_score"],
            device,
            model_type,
        )
        flow_desc = index_to_text("optical_flow_score", idx, label_map)
        confidences.append(conf)

    phrase = f"A {cluster_text} which is more specifically {label_text}"
    if human_desc:
        phrase += f", {human_desc}"
    if face_desc:
        phrase += f", {face_desc}"
    if obj_desc:
        phrase += f", {obj_desc}"
    if color_desc:
        phrase += f", {color_desc}"
    if flow_desc:
        phrase += f", {flow_desc}"

    return phrase, confidences


def main() -> None:
    p = argparse.ArgumentParser(description="Run multiple EEG models on EEG windows")
    p.add_argument("--eeg", required=True, help="Path to EEG numpy file (concept, repetition, window, C, T)")
    p.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[i for i in range(7)],
        help="List of blocks to load",
    )
    p.add_argument(
        "--concepts",
        type=int,
        nargs="+",
        default=[i for i in range(40)],
        help="List of concepts to load",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        nargs="+",
        default=[i for i in range(5)],
        help="List of repetitions to load",
    )
    p.add_argument(
        "--checkpoint_root",
        required=True,
        help="Directory containing all model checkpoints"
    )
    p.add_argument(
        "--model",
        choices=["glmnet", "eegnet", "deepnet"],
        default="glmnet",
        help="Type of model used"
    )
    p.add_argument(
        "--mapping_path",
        default=os.path.join(os.path.dirname(__file__), "label_mappings.json"),
        help="Path to label_mappings.json"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--prompts_path",
        default="Outputs/prompts.txt",
        help="File to save generated prompts",
    )
    p.add_argument(
        "--confidences_path",
        default="Outputs/confidences.txt",
        help="File to save confidence scores",
    )
    args = p.parse_args()
    model_type = args.model

    eeg_all = np.load(args.eeg)
    print(f"Loaded EEG data with shape {eeg_all.shape}")

    # Determine the sample shape using the first provided indices
    b0, c0, r0 = args.blocks[0], args.concepts[0], args.repetitions[0]
    if eeg_all.ndim == 6:
        sample = eeg_all[b0, c0, r0]
    else:
        sample = eeg_all[c0, r0]
    channels, time_len = sample.shape[-2], sample.shape[-1]

    ckpt_dirs = {
        os.path.basename(os.path.normpath(d)): os.path.join(args.checkpoint_root, d)
        for d in os.listdir(args.checkpoint_root)
        if os.path.isdir(os.path.join(args.checkpoint_root, d))
    }

    required = ["label_cluster"] + [f"label_cluster{i}" for i in range(9)]
    for cat in required:
        if cat not in ckpt_dirs:
            raise FileNotFoundError(
                f"Required checkpoint '{cat}' not found in {args.checkpoint_root}"
            )

    optional = [
        "color_binary",
        "color",
        "face_apperance",
        "human_apperance",
        "obj_number",
        "optical_flow_score",
    ]
    for cat in optional:
        if cat not in ckpt_dirs and cat != "color":
            warnings.warn(f"Checkpoint for {cat} not found - skipping")

    label_map = load_label_mappings(args.mapping_path)

    def load(cat: str):
        path = ckpt_dirs[cat]
        return load_model(path, channels, time_len, args.device, args.model)

    models = {}
    scalers = {}
    stats = {}
    for cat in required + [c for c in optional if c in ckpt_dirs]:
        mdl, sc, st = load(cat)
        models[cat] = mdl
        scalers[cat] = sc
        stats[cat] = st

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.prompts_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.confidences_path)), exist_ok=True)

    # Write generated prompts and confidence scores to text files
    with open(args.prompts_path, "w", encoding="utf-8") as f_prompts, open(
        args.confidences_path, "w", encoding="utf-8"
    ) as f_conf:
        for blk in args.blocks:
            for con in args.concepts:
                for rep in args.repetitions:
                    if eeg_all.ndim == 6:
                        eeg = eeg_all[blk, con, rep]
                    else:
                        eeg = eeg_all[con, rep]

                    phrase, confidences = infer_description(
                        eeg, models, scalers, stats, args.device, label_map, model_type
                    )

                    print(f"Block {blk} Concept {con} Repetition {rep}:")
                    print(phrase)
                    print("Confidences:", [f"{c:.2f}" for c in confidences])

                    f_prompts.write(phrase + "\n")
                    f_conf.write(" ".join(f"{c:.4f}" for c in confidences) + "\n")


if __name__ == "__main__":
    main()
