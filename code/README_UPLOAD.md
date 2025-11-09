# Upload Weights to MEGA - Usage Guide

This guide explains how to use the `upload_weights_to_mega.py` script to upload your trained YOLOX weights to MEGA cloud storage.

## Prerequisites

Install the MEGA Python library:
```bash
conda activate yolox
pip install mega.py
```

## Basic Usage

### Upload Best Checkpoint (Auto-detect)

```bash
python code/upload_weights_to_mega.py \
  --email your_email@example.com \
  --password your_password \
  --output-dir YOLOX_outputs/yoloxl-visdrone
```

This will:
- Automatically find the best checkpoint (e.g., `best_ckpt.pth`)
- Upload it to MEGA's root folder
- Generate a shareable link

### Upload Specific File

```bash
python code/upload_weights_to_mega.py \
  --email your_email@example.com \
  --password your_password \
  --file YOLOX_outputs/yoloxl-visdrone/best_ckpt.pth \
  --remote-folder "yolox-weights"
```

### Upload All Checkpoints

```bash
python code/upload_weights_to_mega.py \
  --email your_email@example.com \
  --password your_password \
  --output-dir YOLOX_outputs/yoloxl-visdrone \
  --all \
  --remote-folder "yolox-visdrone-all"
```

This will upload:
- `best_ckpt.pth`
- `latest_ckpt.pth`
- `last_epoch_ckpt.pth`
- Any `epoch_X_ckpt.pth` files

## Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--email` | Yes | MEGA account email | - |
| `--password` | Yes | MEGA account password | - |
| `--output-dir` | No* | YOLOX output directory | - |
| `--file` | No* | Specific checkpoint file to upload | - |
| `--all` | No | Upload all checkpoints | False |
| `--remote-folder` | No | MEGA folder name | "yolox-weights" |

*Either `--output-dir` or `--file` must be provided

## Examples

### Example 1: After YOLOX-L Training Completes

```bash
# Wait for training to finish (75 epochs)
# Best checkpoint will be saved as: YOLOX_outputs/yoloxl-visdrone/best_ckpt.pth

python code/upload_weights_to_mega.py \
  --email myemail@gmail.com \
  --password mypassword123 \
  --output-dir YOLOX_outputs/yoloxl-visdrone \
  --remote-folder "visdrone-yoloxl-75epochs"
```

### Example 2: Upload Swin-Base Weights (8-GPU Training)

```bash
# After 8-GPU Swin training completes
python code/upload_weights_to_mega.py \
  --email myemail@gmail.com \
  --password mypassword123 \
  --output-dir YOLOX_outputs/yolox-swin-visdrone-8gpu \
  --remote-folder "visdrone-swin-base-80epochs"
```

### Example 3: Upload Specific Epoch Checkpoint

```bash
python code/upload_weights_to_mega.py \
  --email myemail@gmail.com \
  --password mypassword123 \
  --file YOLOX_outputs/yoloxl-visdrone/epoch_40_ckpt.pth \
  --remote-folder "visdrone-checkpoints"
```

## Output

The script will:
1. Find/verify the checkpoint file(s)
2. Upload to MEGA
3. Display file size and upload progress
4. Generate shareable link(s)
5. Print link(s) to console

Example output:
```
Found checkpoint: YOLOX_outputs/yoloxl-visdrone/best_ckpt.pth (210.5 MB)
Logging into MEGA...
Login successful!
Creating/accessing folder: yolox-weights
Uploading: best_ckpt.pth
Upload successful!

Shareable link: https://mega.nz/file/XXXXX#YYYYY

All uploads completed successfully!
```

## Troubleshooting

### Error: "No checkpoint file found"
- Check that training has completed and checkpoint exists
- Verify the output directory path is correct
- Use `--file` to specify exact path

### Error: "MEGA login failed"
- Verify email and password are correct
- Check internet connection
- MEGA might temporarily block automated logins (wait and retry)

### Error: "Upload failed"
- Check file size (MEGA free accounts have limits)
- Verify internet connection is stable
- Try uploading smaller files individually

## Current Training Status

**YOLOX-L Training:**
- Status: Resumed at epoch 36/75
- Best mAP so far: 17.93%
- Output: `YOLOX_outputs/yoloxl-visdrone/`
- Best checkpoint: `best_ckpt.pth` (~210 MB)

**Next Steps:**
1. Wait for training to complete (75 epochs, ~39 more epochs remaining)
2. Final model will have improved mAP (target: ~20-25%)
3. Upload final `best_ckpt.pth` to MEGA

## MEGA Storage Limits

- **Free Account**: 20 GB storage (enough for ~90 checkpoints)
- **Each YOLOX-L checkpoint**: ~210 MB
- **Each Swin-Base checkpoint**: ~450 MB

## Security Note

⚠️ **Important:** Never commit MEGA credentials to git!
- Use environment variables for automation:
  ```bash
  export MEGA_EMAIL="your@email.com"
  export MEGA_PASSWORD="yourpassword"
  python code/upload_weights_to_mega.py --output-dir YOLOX_outputs/yoloxl-visdrone
  ```
