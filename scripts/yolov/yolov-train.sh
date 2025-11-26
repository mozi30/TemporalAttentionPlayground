#!/bin/bash
# YOLOX-Swin-Base Training Script for 8x GPU
# Optimized for VisDrone dataset

set -e

# Ensure TAP_REPO and TAP_HOME are available; source the setup helper if not.
# Use the "${VAR:-}" form so the test is safe even under "set -u".
if [ -z "${TAP_REPO:-}" ] || [ -z "${TAP_HOME:-}" ]; then
  # relative to this script's location
  source ./../../setup/setup-env.sh
fi

cd "${TAP_REPO}/YOLOV"
# Use TAP_HOME (defined by setup-env.sh) instead of undefined HOME_DIR
source "${TAP_HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate yolox

echo "Starting training..."
echo ""
python3 tools/vid_train.py \
    -n yolov_swinbase_window_2 \
    -f /home/mozi/TemporalAttentionPlayground/YOLOV/exps/customed_example/yolov_swinbase.py \
    --batch-size 2 \
    --fp16 \
    -c $HOME_DIR/models/yolox-swinbase/best_ckpt.pth

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
