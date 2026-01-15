#!/bin/bash
#
# Inference script v2 with modular data pipeline
#
# This script runs inference using the new EEGPreprocessor-based pipeline,
# ensuring consistent data processing with training.
#
# Usage:
#   bash inference_v2.sh
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Python interpreter
PYTHON="python"

# Subject to process
SUBJECT="sub3"

# Model architecture (must match training)
MODEL="glmnet"

# Random seed (must match training)
SEED=0

# Data paths
EEG_FILE="data/Preprocessing/Segmented_1000ms_sw/${SUBJECT}.npy"
CHECKPOINT_ROOT="Classifiers/checkpoints/mono/${SUBJECT}/shuffle/seed${SEED}/${MODEL}"

# Output paths
OUTPUT_DIR="Outputs"
PROMPTS_PATH="${OUTPUT_DIR}/prompts_${SUBJECT}_v2.txt"
CONFIDENCES_PATH="${OUTPUT_DIR}/confidences_${SUBJECT}_v2.txt"

# Inference script
INFERENCE_SCRIPT="Classifiers/inference_v2.py"

# GPU device
GPU_ID=0

# ============================================================================
# Run Inference
# ============================================================================

echo "=========================================="
echo "Inference v2 with modular data pipeline"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Model: ${MODEL}"
echo "EEG file: ${EEG_FILE}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "=========================================="

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"

# Run inference
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${INFERENCE_SCRIPT} \
    --eeg "${EEG_FILE}" \
    --checkpoint_root "${CHECKPOINT_ROOT}" \
    --model "${MODEL}" \
    --device cuda \
    --prompts_path "${PROMPTS_PATH}" \
    --confidences_path "${CONFIDENCES_PATH}"

echo ""
echo "=========================================="
echo "Inference complete!"
echo "=========================================="
echo "Prompts saved to: ${PROMPTS_PATH}"
echo "Confidences saved to: ${CONFIDENCES_PATH}"
