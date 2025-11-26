# TemporalAttentionPlayground

### Evaluating Temporal Context for Robustness to Perturbations in Video Object Detection Models

This repository contains code and experiments for the bachelor thesis "Evaluating Temporal Context for Robustness to Perturbations in Video Object Detection Models". It is exploring temporal attention mechanisms for object detection in video sequences using UAV (Unmanned Aerial Vehicle) datasets **VisDrone** and **XS-VID**. The project evaluates the performance of temporal attention models **TRANSVOD** and **YOLOV**  as well as there baselines **MSDA** and **YOLOX**.

The repository supports:
- 🔧 Enviroment setup
- 📦 Dataset structure generation & preprocessing
- 🔥 Model setup and training
- 📊 Evaluation and result visualization
- ♻️ FAIR and reproducible experiment design

---

## 🚀 Project Overview

| Component | Description |
|-----------|-------------|
| 🎯 **Goal** | Investigate influence of temporal attention on complex datasets and study the influence of temporal attention on model robustness |
| 🧪 **Datasets** |VisDrone (drone-based object detection)  XS-VID (small object dataset)|
| 🧠 **Framework** | CV Deep learning (PyTorch-based, CUDA)|
| 💡 **Output** | Thesis, Evaluation metrics (mAP, AP), trained models, prediction results, examples |
| ♻️ **FAIR compliance** | Code stored in Git, data in RDM repository (DOI forthcoming), fully documented |

---

## 📂 Repository Structure

```text
.
├─ code/                 # Python code for setup, dataset building
├─ setup/                # Setup sript and environment information
├─ data/                 # Data folder with dataset information and example
├─ results/              # Summary metrics & figures (full outputs in RDM repository)
├─ README.md             # This file
├─ scripts/              # Easy to use scripts for training, evaluating and result generation
└─ .gitignore
```

## 🔧 Setup

### 📌 Preconditions

Before running the setup, ensure that you have:

- NVIDIA GPU supporting **CUDA 11.3**
- At least **50 GB** of free storage space
- A **Linux-based system** (recommended) with `sudo` permissions
- **Conda** or **Python 3.8+** installed

---

### 🚀 Quick Setup Instructions
1. Clone repository including submodules
    ```bash
    git clone --recurse-submodules https://github.com/mozi30/TemporalAttentionPlayground.git
    ```

2. Navigate to the setup directory:

   ```bash
   cd setup
   ```
3. Configure required paths by editing the config.env file:
    ```bash
   nano setup.env
   ```
   Adjust the values according to your system, for example:
    1. Base environment path
    2. Dataset storage location.
    3. Output directory for annotations
    4. Directory for model weights
3. Run the setup script:
    ```
    sudo bash setup-env.sh
    ````
    This script will:
        Install required environments and dependencies (e.g. msda, YOLOX)
    - Download datasets from the specified storage location
    - Generate annotations in the correct format for training and evaluation
    - Download base model weights
    - Finalize the environment for model training and evaluation

You are ready to start working 🚀🚀🚀

For more information on training and evaluation check out `scripts/README.md`

## 📚 Reproducibility & FAIR Principles

This project is designed according to the FAIR principles (Findable, Accessible, Interoperable, Reusable). The experiment follows the data lifecycle described in the Data Management Plan (DMP).

### 🔍 Findable
- Code is version-controlled in this Git repository.
- Full experimental results, dataset annotations, and trained model weights will be published in the TU Wien Research Data Repository (**DOI: _to be added_**).
- Each dataset used (VisDrone, XS-VID) is referenced with its official source and citation.
Further information in `data/README.md`

### 🔓 Accessible
- Code and lightweight experiment results are openly available in this repository.
- Full datasets and large result files (e.g., video sequences, large model checkpoints) are available via RDM repository access, according to their respective licenses.
- Repository includes clear instructions on how to obtain and prepare input data.

### 🔁 Interoperable
- Standard formats are used whenever possible (JSON, YAML, COCO-style annotations, PNG/JPEG images).
- Data processing follows structured Python pipelines.
- Configuration files (e.g. `.env`, YAML) allow reproducibility of the setup.

### 🔂 Reusable
- Code will be provided under the **MIT License** (see `LICENSE` file).
- Produced experimental data will be shared under **CC-BY 4.0** unless restricted by dataset terms.
- Detailed metadata is provided in:
  - `data/README.md` – dataset structure and label mapping
  - `results/README.md` – explanation of metrics and result files
  - `scripts/` – runnable experiment scripts
  - DMP (deposited to Zenodo, DOI pending)

---

## 📎 Data Management Plan (DMP)

A detailed DMP following **Science Europe Guidelines** has been created for this project. It includes:

- Data sources and licensing
- Data processing workflow and reproducibility
- Storage, backup, and access strategy
- Metadata and documentation standards
- FAIR self-assessment and steps taken

The DMP will be deposited on **Zenodo** as part of the _Intro RDM – DMPs 2025_ collection (embargoed until deadline).

🔹 **DMP Title**: `DMP: Evaluating Temporal Context for Robustness to Perturbations in Video Object Detection Models`
🔹 **DOI:** _to be added_ (once uploaded to Zenodo)

---

## 📜 Licensing

| Component     | License            |
|---------------|-------------------|
| **Code**      | MIT License        |
| **Produced Data & Results** | CC BY-NC-SA 3.0 for results including Visdrone, MIT for XS-VID |
| **Input Data** | Per dataset terms (VisDrone: CC BY-NC-SA 3.0, XS-VID: MIT) |

#### Check dataset license before redistribution.

The appropriate license files will be added to the repository and confirmed in the DMP.

---
