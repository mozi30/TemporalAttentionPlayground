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

echo "Starting eval..."
echo ""
python tools/eval.py \
  -f exps/customed_example/yolox_swinbase.py\
  -c /home/mozi/TemporalAttentionPlayground/YOLOV/YOLOX_outputs/yolox_swinbase/last_epoch_ckpt.pth\
  -b 2 \
  -d 1 \
  --conf 0.001 \
  --nms 0.7 \
  --fp16 \
  --fuse





echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: YOLOX_outputs/yolox_swinbase/"
echo "=========================================="
