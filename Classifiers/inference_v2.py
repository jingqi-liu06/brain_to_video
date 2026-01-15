"""
Inference script v2 with modular data pipeline.

This script uses the EEGPreprocessor class to ensure consistent data processing
between training and inference.

Usage:
------
    CUDA_VISIBLE_DEVICES=0 python Classifiers/inference_v2.py \
        --eeg data/Preprocessing/Segmented_1000ms_sw/sub3.npy \
        --checkpoint_root Classifiers/checkpoints/mono/sub3/shuffle/seed0/glmnet \
        --model glmnet \
        --prompts_path Outputs/prompts_sub3.txt \
        --confidences_path Outputs/confidences_sub3.txt

Key features:
1. Uses EEGPreprocessor.load() to restore exact training preprocessing
2. Consistent feature extraction and normalization
3. Multi-model ensemble for attribute prediction
"""

import os
import sys
import json
import argparse
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Classifiers.data.dataset import EEGPreprocessor
from Classifiers.modules.models import glmnet, eegnet, deepnet

# Occipital electrode indices
OCCIPITAL_IDX = list(range(50, 62))


def parse_args():
    p = argparse.ArgumentParser(description="EEG Multi-attribute Inference v2")
    
    p.add_argument("--eeg", required=True, help="Path to EEG .npy file")
    p.add_argument("--checkpoint_root", required=True, help="Root directory containing model checkpoints")
    p.add_argument("--model", choices=["glmnet", "eegnet", "deepnet"], default="glmnet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Data selection (optional)
    p.add_argument("--blocks", type=int, nargs="+", default=None, help="Block indices to process")
    p.add_argument("--concepts", type=int, nargs="+", default=None, help="Concept indices to process")
    p.add_argument("--repetitions", type=int, nargs="+", default=None, help="Repetition indices to process")
    
    # Output
    p.add_argument("--prompts_path", default="Outputs/prompts.txt")
    p.add_argument("--confidences_path", default="Outputs/confidences.txt")
    p.add_argument("--label_map_path", default="Classifiers/label_map.json")
    
    return p.parse_args()


def load_label_map(path: str) -> dict:
    """Load label mapping from JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {cat: {int(k): str(v) for k, v in mapping.items()} for cat, mapping in data.items()}


def load_model(
    ckpt_dir: str,
    model_type: str,
    device: str,
    C: int = 62,
    T: int = 200,
) -> tuple:
    """Load model, preprocessor, and determine output dimension.
    
    Returns:
        Tuple of (model, preprocessor, num_classes)
    """
    # Load preprocessor
    preprocessor = EEGPreprocessor(model_type=model_type, fs=200)
    preprocessor.load(ckpt_dir)
    
    # Determine feature dimension from preprocessor state
    if model_type == "glmnet" and preprocessor.scaler is not None:
        # Infer feature dimension from scaler
        feat_dim = preprocessor.scaler.n_features_in_ // C
    else:
        feat_dim = 0
    
    # Load model checkpoint to get number of classes
    model_path = os.path.join(ckpt_dir, f"{model_type}_best.pt")
    state = torch.load(model_path, map_location=device)
    
    # Infer output dimension from state dict
    # Look for the last linear layer's weight
    out_dim = None
    for key in reversed(list(state.keys())):
        if 'weight' in key and len(state[key].shape) == 2:
            out_dim = state[key].shape[0]
            break
    
    if out_dim is None:
        raise ValueError(f"Could not infer output dimension from checkpoint: {model_path}")
    
    # Build model
    if model_type == "glmnet":
        model = glmnet(OCCIPITAL_IDX, C=C, T=T, feat_dim=feat_dim, out_dim=out_dim)
    elif model_type == "eegnet":
        model = eegnet(out_dim=out_dim, C=C, T=T)
    elif model_type == "deepnet":
        model = deepnet(out_dim=out_dim, C=C, T=T)
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model, preprocessor, out_dim


def majority_vote(
    eeg_windows: np.ndarray,
    model: torch.nn.Module,
    preprocessor: EEGPreprocessor,
    device: str,
) -> tuple:
    """Perform majority vote over sliding windows.
    
    Args:
        eeg_windows: EEG data of shape (n_windows, C, T)
        model: Loaded model
        preprocessor: Fitted preprocessor
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    preds = []
    
    for win in eeg_windows:
        # Process single window
        win_batch = win[np.newaxis, ...]  # (1, C, T)
        x = preprocessor.transform(win_batch)  # (1, C, T+F)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)  # (1, 1, C, T+F)
        
        with torch.no_grad():
            logits = model(x_tensor)
            pred = int(logits.argmax(dim=-1).item())
            preds.append(pred)
    
    # Majority vote
    counter = Counter(preds)
    most_common = counter.most_common(1)[0]
    pred_label = most_common[0]
    confidence = most_common[1] / len(preds)
    
    return pred_label, confidence


def index_to_text(category: str, idx: int, label_map: dict) -> str:
    """Convert predicted index to text description."""
    if category in label_map and idx in label_map[category]:
        return label_map[category][idx]
    return str(idx)


def load_all_models(checkpoint_root: str, model_type: str, device: str) -> dict:
    """Load all category models from checkpoint directory."""
    
    models = {}
    preprocessors = {}
    
    # Define expected categories
    required = ["label_cluster"]
    optional = [
        "color_binary", "color", "face_apperance", "human_apperance",
        "obj_number", "optical_flow_score",
    ]
    cluster_categories = [f"label_cluster{i}" for i in range(9)]
    
    all_categories = required + optional + cluster_categories
    
    for cat in all_categories:
        ckpt_dir = os.path.join(checkpoint_root, cat)
        if not os.path.isdir(ckpt_dir):
            if cat in required:
                raise FileNotFoundError(f"Required checkpoint not found: {ckpt_dir}")
            if cat in optional:
                warnings.warn(f"Checkpoint for {cat} not found - skipping")
            continue
        
        try:
            model, preprocessor, _ = load_model(ckpt_dir, model_type, device)
            models[cat] = model
            preprocessors[cat] = preprocessor
            print(f"Loaded: {cat}")
        except Exception as e:
            warnings.warn(f"Failed to load {cat}: {e}")
    
    return models, preprocessors


def generate_prompt(
    eeg_windows: np.ndarray,
    models: dict,
    preprocessors: dict,
    device: str,
    label_map: dict,
) -> tuple:
    """Generate text prompt from EEG data using all attribute models.
    
    Args:
        eeg_windows: EEG data of shape (n_windows, C, T)
        models: Dictionary of loaded models
        preprocessors: Dictionary of preprocessors
        device: Device for inference
        label_map: Label to text mapping
        
    Returns:
        Tuple of (prompt_string, confidence_list)
    """
    confidences = []
    
    # 1. Predict main cluster (required)
    cluster_idx, cluster_conf = majority_vote(
        eeg_windows, models["label_cluster"], preprocessors["label_cluster"], device
    )
    cluster_desc = index_to_text("label_cluster", cluster_idx, label_map)
    confidences.append(cluster_conf)
    
    # 2. Predict fine-grained label within cluster
    cluster_cat = f"label_cluster{cluster_idx}"
    if cluster_cat in models:
        label_idx, label_conf = majority_vote(
            eeg_windows, models[cluster_cat], preprocessors[cluster_cat], device
        )
        label_desc = index_to_text(cluster_cat, label_idx, label_map)
        confidences.append(label_conf)
    else:
        label_desc = "unknown"
        confidences.append(0.0)
    
    # 3. Predict optional attributes
    face_desc = "0"
    human_desc = "0"
    
    if "face_apperance" in models:
        idx, conf = majority_vote(
            eeg_windows, models["face_apperance"], preprocessors["face_apperance"], device
        )
        face_desc = index_to_text("face_apperance", idx, label_map)
        confidences.append(conf)
    
    if "human_apperance" in models:
        idx, conf = majority_vote(
            eeg_windows, models["human_apperance"], preprocessors["human_apperance"], device
        )
        human_desc = index_to_text("human_apperance", idx, label_map)
        confidences.append(conf)
    
    # Object count
    obj_num_desc = "unknown"
    if "obj_number" in models:
        idx, conf = majority_vote(
            eeg_windows, models["obj_number"], preprocessors["obj_number"], device
        )
        obj_num_map = {0: "one object", 1: "two objects", 2: "three objects"}
        obj_num_desc = obj_num_map.get(idx, f"{idx+1} objects")
        confidences.append(conf)
    
    # Color
    color_desc = "unknown"
    if "color" in models:
        idx, conf = majority_vote(
            eeg_windows, models["color"], preprocessors["color"], device
        )
        color_desc = index_to_text("color", idx, label_map)
        confidences.append(conf)
    elif "color_binary" in models:
        idx, conf = majority_vote(
            eeg_windows, models["color_binary"], preprocessors["color_binary"], device
        )
        color_desc = "many colors" if idx == 1 else "single color"
        confidences.append(conf)
    
    # Motion
    motion_desc = "unknown"
    if "optical_flow_score" in models:
        idx, conf = majority_vote(
            eeg_windows, models["optical_flow_score"], preprocessors["optical_flow_score"], device
        )
        motion_desc = "high motion" if idx == 1 else "low motion"
        confidences.append(conf)
    
    # Compose prompt
    prompt = f"A {cluster_desc} which is more specifically {label_desc}, {face_desc}, {human_desc}, {obj_num_desc}, {color_desc}, {motion_desc}"
    
    return prompt, confidences


def main():
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")
    
    # Load EEG data
    eeg_data = np.load(args.eeg)
    print(f"Loaded EEG data with shape {eeg_data.shape}")
    
    # Expected shape: (n_blocks, n_concepts, n_rep, n_windows, C, T)
    n_blocks, n_concepts, n_rep, n_win, C, T = eeg_data.shape
    
    # Load label map
    label_map = load_label_map(args.label_map_path)
    
    # Load all models
    print("Loading models...")
    models, preprocessors = load_all_models(args.checkpoint_root, args.model, device)
    print(f"Loaded {len(models)} models")
    
    # Determine which samples to process
    blocks = args.blocks if args.blocks is not None else range(n_blocks)
    concepts = args.concepts if args.concepts is not None else range(n_concepts)
    repetitions = args.repetitions if args.repetitions is not None else range(n_rep)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.prompts_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.confidences_path)), exist_ok=True)
    
    # Run inference
    all_prompts = []
    all_confidences = []
    
    for b in tqdm(blocks, desc="Blocks"):
        for c in concepts:
            for r in repetitions:
                # Get EEG windows for this sample
                eeg_windows = eeg_data[b, c, r]  # (n_windows, C, T)
                
                # Generate prompt
                prompt, confidences = generate_prompt(
                    eeg_windows, models, preprocessors, device, label_map
                )
                
                all_prompts.append(prompt)
                all_confidences.append(confidences)
                
                print(f"Block {b} Concept {c} Repetition {r}:")
                print(prompt)
                print(f"Confidences: {[f'{c:.2f}' for c in confidences]}")
    
    # Save results
    with open(args.prompts_path, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(prompt + "\n")
    
    with open(args.confidences_path, "w", encoding="utf-8") as f:
        for conf in all_confidences:
            f.write(", ".join([f"{c:.2f}" for c in conf]) + "\n")
    
    print(f"\nSaved {len(all_prompts)} prompts to {args.prompts_path}")
    print(f"Saved confidences to {args.confidences_path}")


if __name__ == "__main__":
    main()
