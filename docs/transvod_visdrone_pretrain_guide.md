# TransVOD Deformable DETR Pretraining on VisDrone Dataset

This guide explains how to pretrain Deformable DETR on your VisDrone dataset following the TransVOD++ methodology.

## Overview

TransVOD++ uses a **two-stage training approach**:

1. **Stage 1: Single-Frame Baseline** - Train Deformable DETR on individual frames (image detection)
2. **Stage 2: Multi-Frame TransVOD++** - Train the temporal model using the single-frame weights as initialization

## Key Files to Understand

### Training Scripts
- `train_single_frame_swinb.sh` - Launches single-frame training
- `configs/swinb_train_single.sh` - Configuration for single-frame training
- `train_transvod++_swinb.sh` - Launches multi-frame training
- `configs/swinb_train_multi.sh` - Configuration for multi-frame training

### Dataset Files
- `datasets/vid_single.py` - Single-frame dataset loader
- `datasets/vid_multi.py` - Multi-frame dataset loader
- `datasets/__init__.py` - Dataset factory

### Main Training
- `main.py` - Main training entry point
- `engine_single.py` - Training engine for single-frame
- `engine_multi.py` - Training engine for multi-frame

## Adapting for VisDrone

### Step 1: Prepare VisDrone Annotations in COCO Format

Your VisDrone dataset needs to be converted to COCO format. Based on the TransVOD structure, you need:

```
data/
└── visdrone/
    ├── Data/
    │   ├── VID/
    │   │   └── train/
    │   │   └── val/
    │   └── DET/ (optional, if you have detection-only data)
    └── annotations/
        ├── visdrone_vid_train.json
        ├── visdrone_vid_train_joint_30.json  (if combining train+det)
        └── visdrone_vid_val.json
```

**Key annotation format requirements:**
- COCO JSON format with `images`, `annotations`, `categories`
- Each video frame has a `file_name` field pointing to image path
- Bounding boxes in `[x, y, width, height]` format
- Category IDs starting from 1 (not 0)

### Step 2: Modify Dataset Loader for VisDrone

Create a new dataset file or modify `datasets/vid_single.py`:

**File: `datasets/visdrone_single.py`** (create new):

```python
from pathlib import Path
from .vid_single import CocoDetection, make_coco_transforms
from torch.utils.data.dataset import ConcatDataset
from util.misc import get_local_rank, get_local_size

def build(image_set, args):
    root = Path(args.vid_path)  # Point to your VisDrone root
    assert root.exists(), f'provided VisDrone path {root} does not exist'
    
    PATHS = {
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'visdrone_vid_train.json')],
        "train_joint": [(root / "Data", root / "annotations" / 'visdrone_vid_train_joint.json')],
        "val": [(root / "Data" / "VID", root / "annotations" / 'visdrone_vid_val.json')],
    }
    
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(
            img_folder, 
            ann_file, 
            transforms=make_coco_transforms(image_set), 
            return_masks=args.masks, 
            cache_mode=args.cache_mode, 
            local_rank=get_local_rank(), 
            local_size=get_local_size()
        )
        datasets.append(dataset)
    
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
```

### Step 3: Register VisDrone Dataset

Modify `datasets/__init__.py`:

```python
from .visdrone_single import build as build_visdrone_single

def build_dataset(image_set, args):
    # ... existing code ...
    
    if args.dataset_file == 'visdrone_single':
        return build_visdrone_single(image_set, args)
    
    # ... rest of code ...
```

### Step 4: Create Training Configuration

**File: `configs/swinb_train_visdrone_single.sh`**:

```bash
#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/visdrone_singlebaseline_swin_384/swin_e12_bs2_numquery_100
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}

python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 12 \
    --lr_drop_epochs 8 10 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --resume ./exps/our_models/COCO_pretrained_model/swinb_checkpoint0048.pth \
    --coco_pretrain \
    --dataset_file 'visdrone_single' \
    --vid_path /root/datasets/visdrone \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
```

**Important parameters:**
- `--resume`: COCO pretrained weights (download from TransVOD++ links)
- `--coco_pretrain`: Flag indicating starting from COCO weights
- `--dataset_file`: Set to `'visdrone_single'`
- `--vid_path`: Path to your VisDrone dataset root
- `--epochs`: Adjust based on dataset size (7-15 epochs typical)
- `--lr_drop_epochs`: Learning rate drop schedule
- `--num_queries`: Number of detection queries (adjust for VisDrone object density)
- `--batch_size`: Adjust based on GPU memory

### Step 5: Download COCO Pretrained Weights

From TransVOD++ README, download the COCO pretrained Swin-B checkpoint:
- [Google Drive Link](https://drive.google.com/drive/folders/1qXq6jz2-uvnUfa-IO2CoxsvEHnk93wee?usp=share_link)
- Place in: `exps/our_models/COCO_pretrained_model/swinb_checkpoint0048.pth`

### Step 6: Run Single-Frame Training

```bash
# Single GPU
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 0 visdrone_swinb 1 configs/swinb_train_visdrone_single.sh

# Multi-GPU (8 GPUs)
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 0 visdrone_swinb 8 configs/swinb_train_visdrone_single.sh
```

### Step 7: Train Multi-Frame TransVOD++ (Optional)

After single-frame pretraining completes, create multi-frame config:

**File: `configs/swinb_train_visdrone_multi.sh`**:

```bash
#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/visdrone_transvod_swin/swin_e8_bs2
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}

python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 8 \
    --lr_drop_epochs 6 7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --num_ref_frames 3 \
    --resume ./exps/visdrone_singlebaseline_swin_384/swin_e12_bs2_numquery_100/checkpoint.pth \
    --dataset_file 'visdrone_multi' \
    --vid_path /root/datasets/visdrone \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
```

Note: You'll also need to create `datasets/visdrone_multi.py` similar to `vid_multi.py` for temporal modeling.

## Key Configuration Parameters

### Model Architecture
- `--backbone`: `swin_b_p4w7` (Swin-Base), `swin_s_p4w7` (Swin-Small), `swin_t_p4w7` (Swin-Tiny)
- `--num_feature_levels`: 1 or 4 (multi-scale features)
- `--hidden_dim`: Transformer hidden dimension (256 typical)
- `--num_queries`: Detection query slots (100-300)
- `--with_box_refine`: Iterative bounding box refinement

### Training
- `--epochs`: Total training epochs
- `--lr`: Base learning rate (2e-4)
- `--lr_backbone`: Backbone learning rate (2e-5)
- `--lr_drop_epochs`: Epochs to drop learning rate
- `--batch_size`: Per-GPU batch size

### Dataset
- `--dataset_file`: Dataset type (`visdrone_single`, `visdrone_multi`)
- `--vid_path`: Root path to dataset
- `--num_workers`: Data loading workers

### Loss Weights (default values work well)
- `--cls_loss_coef`: 2.0
- `--bbox_loss_coef`: 5.0
- `--giou_loss_coef`: 2.0

## Training Tips

1. **Start from COCO weights**: Always use `--coco_pretrain` and `--resume` with COCO checkpoint
2. **Adjust num_queries**: VisDrone has many small objects - may need more queries (200-300)
3. **Batch size**: Start with 2 per GPU, increase if memory allows
4. **Learning rate**: Keep backbone LR lower (2e-5) than head LR (2e-4)
5. **Training time**: Single-frame training on ImageNet VID takes ~7 epochs, adjust for VisDrone size
6. **Evaluation**: Use `--eval` flag to only evaluate a checkpoint

## Verification Steps

1. **Check annotations loaded correctly**:
   ```python
   from pycocotools.coco import COCO
   coco = COCO('/root/datasets/visdrone/annotations/visdrone_vid_train.json')
   print(f"Images: {len(coco.imgs)}")
   print(f"Annotations: {len(coco.anns)}")
   print(f"Categories: {coco.cats}")
   ```

2. **Test dataset loading**:
   ```bash
   python main.py --dataset_file visdrone_single --vid_path /root/datasets/visdrone --eval
   ```

3. **Monitor training**: Check `exps/<output_dir>/log.train.*` for losses and mAP

## Expected Results

- **Single-frame baseline**: After ~7-12 epochs, should achieve reasonable AP@0.5
- **Multi-frame TransVOD++**: Additional 2-5% AP improvement with temporal modeling

## Troubleshooting

**"Dataset not found"**: Check `--vid_path` points to correct root directory

**"CUDA out of memory"**: Reduce `--batch_size` or use gradient accumulation

**"Category ID error"**: Ensure category IDs start from 1, not 0

**"Low mAP"**: Check anchor scales match VisDrone object sizes, may need to adjust

## References

- TransVOD++ paper: https://arxiv.org/pdf/2201.05047.pdf
- Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR
- Original TransVOD: https://github.com/SJTU-LuHe/TransVOD
