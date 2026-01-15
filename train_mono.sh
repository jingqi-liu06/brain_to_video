#!/bin/bash
set -e

# Configuration
SUBJECT=${1:-"sub3"}
SEED=0
MODEL="glmnet"
SHUFFLE=true  # Set to true to enable shuffle mode
USE_WANDB=true # Set to true to enable wandb logging

# Paths
TRAIN_SCRIPT="Classifiers/train_classifier_mono.py"
CKPT_ROOT="Classifiers/checkpoints"
PYTHON="python"

# Categories to train (excluding 'label' which is handled separately per cluster)
CATEGORIES=("color" "color_binary" "face_apperance" "human_apperance" "label_cluster" "obj_number" "optical_flow_score")

# Clusters for the 'label' category
LABEL_CLUSTERS=(0 1 2 3 4 5 6 7 8)

# Determine mode string for path construction
if [ "$SHUFFLE" = true ]; then
    MODE="shuffle"
    SHUFFLE_ARG="--shuffle"
else
    MODE="ordered"
    SHUFFLE_ARG=""
fi

# Determine wandb argument
if [ "$USE_WANDB" = true ]; then
    WANDB_ARG="--use_wandb"
else
    WANDB_ARG=""
fi

echo "Starting training for Subject: $SUBJECT, Model: $MODEL, Seed: $SEED, Mode: $MODE"

# Loop 1: Train general categories
for category in "${CATEGORIES[@]}"; do
    # Construct checkpoint path to check if it already exists
    # Path format matches Makefile: mono/SUBJECT/MODE/seedSEED/MODEL/CATEGORY/MODEL_best.pt
    ckpt_dir="$CKPT_ROOT/mono/$SUBJECT/$MODE/seed$SEED/$MODEL/$category"
    ckpt_file="$ckpt_dir/${MODEL}_best.pt"

    if [ ! -f "$ckpt_file" ]; then
        echo "----------------------------------------------------------------"
        echo "Training category: $category"
        echo "----------------------------------------------------------------"
        $PYTHON $TRAIN_SCRIPT \
            --category "$category" \
            --model "$MODEL" \
            --seed "$SEED" \
            --subj_name "$SUBJECT" \
            --save_dir "$CKPT_ROOT" \
            $SHUFFLE_ARG \
            $WANDB_ARG
    else
        echo "[Skip] $category: checkpoint already exists at $ckpt_file"
    fi
done

# Loop 2: Train label clusters
for cluster in "${LABEL_CLUSTERS[@]}"; do
    # Construct checkpoint path
    # Path format matches Makefile: mono/SUBJECT/MODE/seedSEED/MODEL/label_clusterCLUSTER/MODEL_best.pt
    ckpt_dir="$CKPT_ROOT/mono/$SUBJECT/$MODE/seed$SEED/$MODEL/label_cluster$cluster"
    ckpt_file="$ckpt_dir/${MODEL}_best.pt"

    if [ ! -f "$ckpt_file" ]; then
        echo "----------------------------------------------------------------"
        echo "Training category: label (Cluster $cluster)"
        echo "----------------------------------------------------------------"
        $PYTHON $TRAIN_SCRIPT \
            --category "label" \
            --cluster "$cluster" \
            --model "$MODEL" \
            --seed "$SEED" \
            --subj_name "$SUBJECT" \
            --save_dir "$CKPT_ROOT" \
            $SHUFFLE_ARG \
            $WANDB_ARG
    else
        echo "[Skip] label cluster $cluster: checkpoint already exists at $ckpt_file"
    fi
done

echo "All training tasks completed."
