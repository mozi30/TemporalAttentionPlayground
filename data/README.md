# 📁 Dataset Documentation

This directory contains documentation for the datasets used in the project:

- **VisDrone** – Drone-based object detection dataset
- **XS-VID** – Small object dataset with temporal sequence data (used in thesis work)

⚠️ **Important:**
Raw datasets and full processed data **are not included in this repository** due to size and licensing restrictions. Dataset locations must be configured manually (see instructions below).

---

## 🌍 Dataset Sources & Licensing

| Dataset | Description | License | Download |
|--------|-------------|---------|----------|
| **VisDrone 2019-DET** | Drone-based object detection dataset with frames from various scenes | CC BY-NC 4.0 (non-commercial use only) | https://github.com/VisDrone/VisDrone-Dataset |
| **XS-VID** | XS-VID: An Extra Small Object Video Detection Dataset| MIT licence | https://github.com/gjhhust/XS-VID |

---

## 🗂 Folder Structure (after setup)

```
data/
└── datasets/
    ├── visdrone/
    │   ├── images/
    │   │   ├──sequences/
    │   │   └──sequences/
    │   ├── annotations/
    │   │   ├── test_categories.json
    │   │   └── val_categories.json
    │   └── ...
    ├── xs_vid/
    │   │   ├──sequences/
    │   │   └──sequences/
    │   ├── annotations/
    │   │   ├── test_categories.json
    │   │   └── val_categories.json
```

---

## 🔠 Annotation Format

Annotations are generated in **COCO-style format**, including:

**For image models (YOLOX, MSDA)**
```json
{
  "images": [...],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [x, y, width, height],
      "category_id": 3,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"}
  ]
}
```

**For video models (YOLOV, TRANSVOD)**
```json
{
  "videos": [...],
  "annotations": [
    {
      "video_id": 2,
      "track_id": 1,
      "category_id": 3,
      "iscrowd": 0,
      "frames": [...],
      "bboxes": [[x, y, w, h], ...]
    }
  ],
  "categories": [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"}
  ]
}
```

---

## ⚙️ Dataset Path Configuration

Set the dataset path in:

```
setup/config.env
```

Example:

```bash
DATASET_PATH=/your/storage/location/datasets
```

> The setup script will use this location to copy datasets, structure folders, and generate annotations.

---

## 🔒 Usage & Reuse Conditions

| Component | Reuse Allowed? | Conditions |
|----------|----------------|------------|
| **VisDrone** | ✔ Yes | Cite authors, *non-commercial only* |
| **XS-VID** | ⚠ Yes | Include licence of original |
| **Generated Data** | CC BY-NC-SA 3.0 (derived from Visdrone) | Will be stored in RDM repository |
| **Model Outputs** | CC BY-NC-SA 3.0 (derived from Visdrone) | DOI pending (will be linked) |

---

## 🔍 Citation

For VisDrone dataset:

>@article{zhu2021detection, title={Detection and tracking meet drones challenge}, author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, volume={44}, number={11}, pages={7380--7399}, year={2021}, publisher={IEEE}}

For XS-VID:
>@article{guo2024XSVID, title={XS-VID: An Extremely Small Video Object Detection Dataset},author={Jiahao Guo, Ziyang Xu, Lianjun Wu, Fei Gao, Wenyu Liu, Xinggang Wang},journal={arXiv preprint arXiv:2407.18137},year={2024}}



---

## 📦 Summary

✔ Dataset path → set in `setup/config.env`
✔ Setup script → downloads datasets, generates structure & annotations
✔ Licensing → follow dataset rules
✔ Processed outputs → FAIR compliant & stored in RDM repository

---

## 📬 Contact

**Author:** <Moritz Zideck>
**Email:** <e12217036@student.tuwien.ac.at>
**ORCID:** <your ORCID>
