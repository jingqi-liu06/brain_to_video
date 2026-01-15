#!/bin/bash
#
# Training script v2 with modular data pipeline
# 
# This script trains all category classifiers for a single subject using
# the new EEGPreprocessor-based data pipeline.
#
# Usage:
#   bash train_v2.sh
#
# Key differences from train_mono.sh:
# 1. Uses Classifiers/train_v2.py which employs EEGPreprocessor
# 2. Preprocessor state (stats, scaler) is saved alongside model checkpoints
# 3. Ensures consistency between training and inference
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Python interpreter
PYTHON="python"

# Subject to train
SUBJECT="sub3"

# Model architecture
MODEL="glmnet"

# Random seed
SEED=0

# Training parameters
EPOCHS=500
BATCH_SIZE=100
LEARNING_RATE=1e-4
SCHEDULER="reducelronplateau"

# Data directories
RAW_DIR="data/Preprocessing/Segmented_1000ms_sw"
LABEL_DIR="data/Video/meta-info"
SAVE_DIR="checkpoints"

# Training script
TRAIN_SCRIPT="Classifiers/train_v2.py"

# GPU device
GPU_ID=0

# Categories to train (basic attributes)
CATEGORIES=(
    "color"
    "color_binary"
    "face_apperance"
    "human_apperance"
    "label_cluster"
    "obj_number"
    "optical_flow_score"
)

# Label clusters (fine-grained categories)
LABEL_CLUSTERS=(0 1 2 3 4 5 6 7 8)

# ============================================================================
# Training Loop
# ============================================================================

echo "=========================================="
echo "Training v2 with modular data pipeline"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Model: ${MODEL}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "=========================================="

# Train basic attribute classifiers
for category in "${CATEGORIES[@]}"; do
    echo ""
    echo ">>> Training: ${category}"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${TRAIN_SCRIPT} \
        --raw_dir "${RAW_DIR}" \
        --label_dir "${LABEL_DIR}" \
        --subj_name "${SUBJECT}" \
        --category "${category}" \
        --model "${MODEL}" \
        --epochs ${EPOCHS} \
        --bs ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --scheduler "${SCHEDULER}" \
        --seed ${SEED} \
        --shuffle \
        --save_dir "${SAVE_DIR}"
done

# Train fine-grained label classifiers for each cluster
for cluster in "${LABEL_CLUSTERS[@]}"; do
    echo ""
    echo ">>> Training: label (cluster ${cluster})"
    echo ""
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${TRAIN_SCRIPT} \
        --raw_dir "${RAW_DIR}" \
        --label_dir "${LABEL_DIR}" \
        --subj_name "${SUBJECT}" \
        --category "label" \
        --cluster ${cluster} \
        --model "${MODEL}" \
        --epochs ${EPOCHS} \
        --bs ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --scheduler "${SCHEDULER}" \
        --seed ${SEED} \
        --shuffle \
        --save_dir "${SAVE_DIR}"
done

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Checkpoints saved to: ${SAVE_DIR}/mono/${SUBJECT}/shuffle/seed${SEED}/${MODEL}/"
