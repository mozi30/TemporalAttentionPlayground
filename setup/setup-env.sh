#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MSDA + CUDA + OpenMMLab + YOLOX setup (idempotent & nounset-safe)
# - Adds per-step "already done" checks and markers so reruns skip completed work
# - Use FORCE_RUN=1 to override markers and redo a step
# -----------------------------------------------------------------------------

set -Eeuo pipefail

# ===== Config toggles (edit as needed) =======================================
EXPORT_ENV_VARS=${EXPORT_ENV_VARS:-1}  # 1 to export TAP_* vars to /etc/profile.d/tap_env.sh
INSTALL_APT=${INSTALL_APT:-1}                # 1 to install OS deps with apt-get
INSTALL_CUDA=${INSTALL_CUDA:-1}              # 1 to install CUDA 11.3 toolkit runfile
SETUP_MINICONDA=${SETUP_MINICONDA:-1}        # 1 to install/initialize Miniconda
SETUP_ENV_MSDA=${SETUP_ENV_MSDA:-1}          # 1 to create & prepare conda env "msda"
SETUP_ENV_YOLOX=${SETUP_ENV_YOLOX:-1}        # 1 to create & prepare conda env "yolox"
BUILD_TRANSVOD_OPS=${BUILD_TRANSVOD_OPS:-1}  # 1 to build TransVOD_plusplus ops
DOWNLOAD_VISDRONE=${DOWNLOAD_VISDRONE:-1}    # 1 to run VisDrone dataset scripts
DOWNLOAD_FTP_DATASETS=${DOWNLOAD_FTP_DATASETS:-1} # 1 to pull datasets via FTP
INSTALL_MEGACMD=${INSTALL_MEGACMD:-1}           # 1 = install MEGAcmd if missing
MEGA_LOGIN_ON_START=${MEGA_LOGIN_ON_START:-1}   # 1 = login before downloads
DOWNLOAD_VISDRONE_MEGA=${DOWNLOAD_VISDRONE_MEGA:-1}  # 1 to download VisDrone via MEGA (python helper)
CONVERT_VISDRONE_TRANSVOD=${CONVERT_VISDRONE_TRANSVOD:-1}  # 1 to convert VisDrone to TransVOD format
CONVERT_VISDRONE_YOLOV=${CONVERT_VISDRONE_YOLOV:-1}  # 1 to convert VisDrone to YOLOV format
CREATE_YOLOX_ANNOTS=${CREATE_YOLOX_ANNOTS:-1}  # 1 to create YOLOX annotations from VisDrone
DOWNLOAD_YOLOV_WEIGHTS=${DOWNLOAD_YOLOV_WEIGHTS:-1}  # 1 to download pretrained YOLOV weights
CREATE_VISDRONE_COCO_ANNOTS=${CREATE_VISDRONE_COCO_ANNOTS:-1}  # 1 to create COCO annots for VisDrone
DOWNLOAD_SWIN_WEIGHTS=${DOWNLOAD_SWIN_WEIGHTS:-1}    # 1 to download Swin Transformer pretrained weights
DOWNLOAD_YOLOX_COCO_WEIGHTS=${DOWNLOAD_YOLOX_COCO_WEIGHTS:-1}  # 1 to download YOLOX COCO pretrained weights
FORCE_RUN=${FORCE_RUN:-0}                    # 1 to ignore step markers and redo

# ===== Paths & versions ======================================================
source ./config.env
CONDA_HOME="${TAP_HOME}/miniconda3"
CUDA_VERSION_DIR="/usr/local/cuda-11.3"
CUDA_RUNFILE="cuda_11.3.0_465.19.01_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/${CUDA_RUNFILE}"
HELPER_SCRIPT_MSDA="${TAP_HOME}/activate_msda.sh"
HELPER_SCRIPT_YOLOX="${TAP_HOME}/activate_yolox.sh"
REPO_DIR="${TAP_REPO}"
SETUP_CODE_DIR="${REPO_DIR}/code/setup"
OPS_DIR="$REPO_DIR/TransVOD_plusplus/models/ops"
DATASETS_ROOT="${TAP_DATASETS}"
VISDRONE_ROOT="$DATASETS_ROOT/visdrone"
VISDRONE_YOLOV="$VISDRONE_ROOT/yolov"
VISDRONE_TRANSVOD="$VISDRONE_ROOT/transvod"


# Provide your Pro account credentials via env vars (or export before running)
MEGA_EMAIL="e12217036@student.tuwien.ac.at"                  # e.g., export MEGA_EMAIL="you@example.com"
MEGA_PASS="Spam_pass30"                      # e.g., export MEGA_PASS="your_password"

# ===== State / idempotency ===================================================
STATE_DIR="${TAP_HOME}/.tap_setup_state"
mkdir -p "${STATE_DIR}"

marker() { printf "%s/%s.done" "$STATE_DIR" "$1"; }
already_done() {
  local step="$1"
  if [ "${FORCE_RUN}" -eq 1 ]; then return 1; fi
  [ -f "$(marker "$step")" ]
}
mark_done() { : > "$(marker "$1")"; }

# ===== Pretty logging helpers ===============================================
section() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()      { printf "\033[1;32m✅ %s\033[0m\n" "$*"; }
warn()    { printf "\033[1;33m⚠️  %s\033[0m\n" "$*"; }
err()     { printf "\033[1;31m❌ %s\033[0m\n" "$*"; }
trap 'err "Error on line $LINENO: $BASH_COMMAND"' ERR
#====== Export TAP VARIABLES =================================================
set_tap_env() {
    [ "${EXPORT_ENV_VARS}" -ge 0 ] >/dev/null 2>&1 || true  # silence shellcheck re: unused toggles
    local step="export_tap_env"
    if already_done "$step"; then warn "Repo step already done; skipping"; return 0; fi
    local TARGET_FILE="/etc/profile.d/tap_env.sh"
    local TEMP_FILE="/tmp/tap_env.sh"

  # New desired content (dynamic: references TAP_HOME at source time, with a reasonable default)
  # We intentionally write escaped-dollar forms (e.g. \${TAP_HOME}) so the resulting
  # /etc/profile.d/tap_env.sh evaluates TAP_HOME and related vars when sourced, not
  # when this script runs. This avoids hard-coded "/home/mozi" while still providing
  # a default if TAP_HOME is not set in the target shell.
  cat > "$TEMP_FILE" << EOF
#!/bin/sh
export TAP_HOME="\${TAP_HOME:-/home/mozi}"
export TAP_DATASETS="\${TAP_HOME}/datasets"
export TAP_MODELS="\${TAP_HOME}/models"
export TAP_WEIGHTS="\${TAP_HOME}/weights"
export TAP_OUTPUTS="\${TAP_HOME}/outputs"
export TAP_REPO="\${TAP_HOME}/TemporalAttentionPlayground"
EOF

    # Check if the file already exists
    if [ -f "$TARGET_FILE" ]; then
        echo "⚠️  $TARGET_FILE already exists."

        # Compare with existing content
        if sudo cmp -s "$TEMP_FILE" "$TARGET_FILE"; then
            echo "✔️  No changes needed — environment variables are already up to date."
            rm "$TEMP_FILE"
            return 0
        else
            echo "🔄 File exists but content differs. Updating..."
        fi
    else
        echo "📄 No existing file found — creating new environment config."
    fi

    # Move file and set permissions
    sudo mv "$TEMP_FILE" "$TARGET_FILE"
    sudo chmod +x "$TARGET_FILE"

    echo "✅ Environment variables updated at $TARGET_FILE"
    echo "🔁 Run: source /etc/profile  (or log out and back in)"
    mark_done "$step"
}



# ===== Clone or update repo ==================================================
clone_repo() {
  [ "${INSTALL_APT}" -ge 0 ] >/dev/null 2>&1 || true  # silence shellcheck re: unused toggles
  local step="clone_repo"
  if already_done "$step"; then warn "Repo step already done; skipping"; return 0; fi

  local REPO_URL="https://github.com/mozi30/TemporalAttentionPlayground.git"
  section "Cloning/updating repository"
  if [ -d "$REPO_DIR/.git" ]; then
    echo "🔄 Repo exists — pulling latest..."
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" pull --rebase
    git -C "$REPO_DIR" submodule update --init --recursive
  else
    echo "⬇️  Cloning TemporalAttentionPlayground (with submodules)..."
    git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
  fi
  ok "Repository ready at $REPO_DIR"
  mark_done "$step"
}

# ===== Helper script for quick env activation ===============================
write_msda_helper() {
  local step="helper_msda"
  if already_done "$step"; then warn "MSDA helper already written; skipping"; return 0; fi

  section "Writing helper: ${HELPER_SCRIPT_MSDA}"
  cat > "${HELPER_SCRIPT_MSDA}" << 'EOF'
set -e
# temporarily relax nounset inside helper since conda hooks expect unset vars
set +u
export ADDR2LINE="$(command -v addr2line || true)"
export NM="$(command -v nm || true)"
export STRINGS="$(command -v strings || true)"
# conda sh hook
if [ -f "$TAP_HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$TAP_HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate msda 2>/dev/null || true
fi
set -u
EOF
  chmod +x "${HELPER_SCRIPT_MSDA}"
  ok "Helper created at ${HELPER_SCRIPT_MSDA}"
  mark_done "$step"
}

write_yolox_helper() {
  local step="helper_yolox"
  if already_done "$step"; then warn "YOLOX helper already written; skipping"; return 0; fi

  section "Writing helper: ${HELPER_SCRIPT_YOLOX}"
  cat > "${HELPER_SCRIPT_YOLOX}" << 'EOF'
set -e
# temporarily relax nounset inside helper since conda hooks expect unset vars
set +u
if [ -f "$TAP_HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$TAP_HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate yolox 2>/dev/null || true
fi
set -u
EOF
  chmod +x "${HELPER_SCRIPT_YOLOX}"
  ok "Helper created at ${HELPER_SCRIPT_YOLOX}"
  mark_done "$step"
}

# ===== APT prerequisites =====================================================
install_apt_prereqs() {
  [ "${INSTALL_APT}" -eq 1 ] || { warn "Skipping apt-get step"; return 0; }
  local step="apt_prereqs"
  if already_done "$step"; then warn "APT prerequisites already installed; skipping"; return 0; fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not available; skipping OS deps"
    mark_done "$step"
    return 0
  fi

  section "Installing OS prerequisites via apt-get"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    build-essential gcc-10 g++-10 pkg-config cmake make \
    ffmpeg curl ca-certificates git wget gnupg
  ok "Compilers installed"
  export CC=/usr/bin/gcc-10
  export CXX=/usr/bin/g++-10
  "$CC" --version || true
  mark_done "$step"
}

# ===== CUDA 11.3 toolkit (optional) =========================================
install_cuda_113() {
  [ "${INSTALL_CUDA}" -eq 1 ] || { warn "Skipping CUDA install"; return 0; }
  local step="cuda_11_3"
  if already_done "$step"; then warn "CUDA 11.3 step already done; skipping"; return 0; fi

  section "Installing CUDA 11.3 (toolkit only)"
  if [ -d "${CUDA_VERSION_DIR}/lib64" ]; then
    ok "CUDA found at ${CUDA_VERSION_DIR}; skipping installer"
  else
    if [ ! -f "${CUDA_RUNFILE}" ]; then
      section "Downloading CUDA runfile"
      wget --progress=dot:giga -t 3 -O "${CUDA_RUNFILE}" "${CUDA_URL}"
    fi
    chmod +x "${CUDA_RUNFILE}"
    section "Running silent CUDA installer"
    sudo ./"${CUDA_RUNFILE}" --silent --toolkit --override
    if [ -d "${CUDA_VERSION_DIR}/lib64" ]; then
      echo "${CUDA_VERSION_DIR}/lib64" | sudo tee /etc/ld.so.conf.d/cuda-11-3.conf >/dev/null
      sudo ldconfig
      ok "CUDA libraries registered"
    else
      err "CUDA install did not produce ${CUDA_VERSION_DIR}/lib64"
      exit 1
    fi
  fi

  # shell env for future shells
  section "Ensuring /etc/profile.d/cuda.sh"
  if [ ! -f /etc/profile.d/cuda.sh ]; then
    sudo bash -c "cat >/etc/profile.d/cuda.sh" <<'EOSH'
if [ -d /usr/local/cuda-11.3 ]; then
  export CUDA_HOME=/usr/local/cuda-11.3
  case ":${PATH}:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$CUDA_HOME/bin:/usr/bin:${PATH:-}";; esac
  if [ -d "$CUDA_HOME/lib64" ]; then
    if [ -z "${LD_LIBRARY_PATH-}" ]; then
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
    else
      case ":${LD_LIBRARY_PATH}:" in *":$CUDA_HOME/lib64:"*) ;; *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}";; esac
    fi
  fi
fi
EOSH
    ok "CUDA profile script installed"
  else
    warn "CUDA profile script already present; skipping"
  fi

  mark_done "$step"
}

# ===== Miniconda bootstrap ===================================================
setup_miniconda() {
  [ "${SETUP_MINICONDA}" -eq 1 ] || { warn "Skipping Miniconda setup"; return 0; }
  local step="miniconda_bootstrap"
  if already_done "$step"; then warn "Miniconda bootstrap already done; skipping"; return 0; fi

  section "Ensuring Miniconda at ${CONDA_HOME}"
  if [ ! -d "${CONDA_HOME}" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_HOME}"
    rm -f /tmp/miniconda.sh
  else
    ok "Miniconda already present"
  fi

  "${CONDA_HOME}/bin/conda" init bash >/dev/null 2>&1 || true

  # make `conda activate` available in this running script too
  set +u
  if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
  else
    # fallback to shell hook
    eval "$(${CONDA_HOME}/bin/conda shell.bash hook)" || true
  fi
  set -u

  ok "Miniconda ready"
  mark_done "$step"
}

# ===== Activate conda inside this script (nounset-safe) ======================
conda_shell_hook() {
  local step="conda_base_init"
  if already_done "$step"; then warn "Conda base init already done; skipping"; return 0; fi

  section "Initializing conda shell (nounset temporarily disabled)"
  # shellcheck disable=SC1091
  set +u
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  conda update -n base -y conda || true
  conda install -y -n base conda-libmamba-solver || true
  conda config --set solver libmamba || true
  conda --version || true
  conda config --show solver | sed -n 's/^solver: /solver: /p' || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  set -u
  mark_done "$step"
}

# ===== MSDA environment: torch 1.8.1/cu111 + OpenMMLab stack ================
setup_env_msda() {
  [ "${SETUP_ENV_MSDA}" -eq 1 ] || { warn "Skipping env 'msda'"; return 0; }
  local step="env_msda"
  if already_done "$step"; then warn "'msda' environment already configured; skipping"; return 0; fi

  section "Creating/activating env: msda (Python 3.9)"
  set +u
  if conda env list | awk '{print $1}' | grep -qx msda; then
    warn "Environment 'msda' already exists"
  else
    conda create -y -n msda python=3.9
  fi
  conda activate msda
  set -u

  section "Upgrading pip & Jupyter kernel"
  set +u
  python -m pip install --upgrade pip
  python -m pip install ipykernel
  python -m ipykernel install --user --name msda --display-name "Python (msda)" || true
  set -u
  ok "Kernel 'Python (msda)' installed/updated"

  section "Compilers for CUDA/cpp extensions (conda)"
  set +u
  conda install -y -n msda -c conda-forge "gcc_linux-64=10.*" "gxx_linux-64=10.*" binutils_linux-64
  conda run -n msda bash -lc '
    set -euo pipefail
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX"
    "$CXX" --version
  '
  set -u

  section "Installing PyTorch 1.8.1 + cu111"
  python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu111 \
    torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1

  section "Core libraries (pinned for compatibility)"
  echo "numpy==1.23.5" > /tmp/constraints.txt
  python -m pip install --no-cache-dir -c /tmp/constraints.txt \
    "numpy==1.23.5" "opencv-python-headless<5" yacs pyyaml tqdm matplotlib pillow cython "packaging<24" "scipy<1.11"

  section "pycocotools"
  python -m pip install --no-cache-dir pycocotools

  section "mmcv-full (Torch 1.8.1/cu111) with ops"
  export MMCV_WITH_OPS=1
  export FORCE_CUDA=1
  export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
  if command -v gcc-10 >/dev/null 2>&1; then export CC=gcc-10 CXX=g++-10; fi
  python -m pip install --no-cache-dir \
    --no-build-isolation \
    "mmcv-full==1.7.2" \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.1/index.html

  section "mmdetection"
  python -m pip install --no-cache-dir -c /tmp/constraints.txt "mmdet==2.28.2"

  section "Sanity check: torch/mmcv/mmdet"
  python - <<'PY'
import torch, mmcv, mmdet
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"mmcv: {mmcv.__version__}, mmdet: {mmdet.__version__}")
PY

  section "CUDA env on conda activate (if CUDA exists)"
  if [ -d /usr/local/cuda ]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    {
      echo 'export CUDA_HOME=/usr/local/cuda'
      echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    } >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
    ok "CUDA vars will be applied when 'msda' is activated"
  else
    warn "No /usr/local/cuda detected"
  fi

  write_msda_helper
  mark_done "$step"
}

# ===== Build TransVOD++ ops (optional) =======================================
build_transvod_ops() {
  [ "${BUILD_TRANSVOD_OPS}" -eq 1 ] || { warn "Skipping TransVOD++ ops build"; return 0; }
  local step="build_transvod_ops"
  if already_done "$step"; then warn "TransVOD++ ops already built; skipping"; return 0; fi

  section "Building TransVOD_plusplus custom ops"
  if [ -d "${OPS_DIR}" ]; then
    pushd "${OPS_DIR}" >/dev/null
    python setup.py install
    popd >/dev/null
    ok "TransVOD++ ops built"
    mark_done "$step"
  else
    warn "Ops dir not found: ${OPS_DIR} (skipping)"
    mark_done "$step"
  fi
}

# ===== VisDrone scripts (optional) ===========================================
run_visdrone_tools() {
  [ "${DOWNLOAD_VISDRONE}" -eq 1 ] || { warn "Skipping VisDrone tools"; return 0; }
  local step="visdrone_tools"
  if already_done "$step"; then warn "VisDrone tools already run; skipping"; return 0; fi

  section "Running VisDrone helpers (download/convert)"
  mkdir -p "${VISDRONE_ROOT}" "${VISDRONE_TRANSVOD}"

  # If folders already exist, skip heavy steps
  if [ -d "${VISDRONE_ROOT}/VisDrone2019-VID-train" ] && [ -d "${VISDRONE_ROOT}/VisDrone2019-VID-val" ]; then
    warn "VisDrone train/val already present; skipping download"
  else
    python3 "${SETUP_CODE_DIR}/visdrone_download.py" --root "${VISDRONE_ROOT}" || warn "visdrone_download failed"
  fi

  # if [ -d "${VISDRONE_TRANSVOD}" ]; then
  #   warn "Converted VisDrone (TransVOD format) already present; skipping conversion"
  # else
  #   python3 "${SETUP_CODE_DIR}/testVisdroneToImageNetVid.py" \
  #     --visdrone-train "${VISDRONE_ROOT}/VisDrone2019-VID-train" \
  #     --visdrone-val   "${VISDRONE_ROOT}/VisDrone2019-VID-val" \
  #     --out-root       "${VISDRONE_TRANSVOD}" \
  #     --link-mode      hardlink || warn "testVisdroneToImageNetVid.py failed"
  # fi

  mark_done "$step"
}

# ===== YOLOX environment (optional track/training) ===========================

setup_env_yolox() {
  [ "${SETUP_ENV_YOLOX}" -eq 1 ] || { warn "Skipping env 'yolox'"; return 0; }
  local step="env_yolox"
  if already_done "$step"; then warn "'yolox' environment already configured; skipping"; return 0; fi

  section "Creating/activating env: yolox (Python 3.9)"
  set +u
  if conda env list | awk '{print $1}' | grep -qx yolox; then
    warn "Environment 'yolox' already exists"
  else
    conda create -y -n yolox python=3.9
  fi
  conda activate yolox
  set -u

  # Global pip constraint file to keep the legacy stack stable
  section "Applying global pip constraints for legacy stack"
  export PIP_CONSTRAINT=/tmp/yolox-legacy-constraints.txt
  cat > "$PIP_CONSTRAINT" << 'EOF'
numpy==1.21.5
scipy==1.7.3
protobuf==3.20.3
EOF
  echo "Using PIP_CONSTRAINT=$PIP_CONSTRAINT"

  section "PyTorch 1.10.0 + cu113"
  python -m pip install --no-cache-dir \
    torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

  section "Core conda deps"
  set +u
  conda install -y -c conda-forge numpy=1.21.5 opencv=4.5.5 scikit-image=0.19.1 \
    pillow=9.0.1 h5py=3.6.0 scikit-learn=1.0.2 scipy=1.7.3 cython
  set -u

  section "Pip extras"
  python -m pip install --no-cache-dir \
    loguru==0.6.0 tqdm==4.62.3 ninja==1.10.2 \
    tabulate==0.8.9 tensorboard==2.8.0 thop==0.0.31-2005241907

  section "Tracking libs (lap/lapx, motmetrics, filterpy) with numpy pin"
  set +u
  if conda install -y -c conda-forge lap "numpy=1.21.5"; then
    PY_NP=$(python -c 'import numpy as n; print(n.__version__)')
    if [[ ! "$PY_NP" =~ ^1\.21\. ]]; then
      warn "lap pulled numpy $PY_NP; removing lap and switching to lapx"
      conda remove -y lap
      LAP_OK=false
    else
      LAP_OK=true
    fi
  else
    LAP_OK=false
  fi

  if [[ "$LAP_OK" != "true" ]]; then
    python -m pip install --no-cache-dir "lapx>=0.5.4" || warn "lapx install failed (tracking may be slower)"
  fi

  python -m pip install --no-cache-dir motmetrics==1.2.5 filterpy==1.4.5
  set -u

  section "COCO tools"
  set +u
  conda install -y -c conda-forge pycocotools "numpy=1.21.5" || \
    python -m pip install --no-cache-dir pycocotools==2.0.6 || \
    python -m pip install --no-cache-dir pycocotools==2.0.4 || warn "pycocotools failed"
  set -u

  section "ONNX + onnxruntime (protobuf pinned to 3.20.3)"
  conda install -y -c conda-forge "protobuf=3.20.3" || true
  python -m pip install --no-cache-dir --upgrade --force-reinstall "protobuf==3.20.3"

  if ! python -m pip install --no-cache-dir --upgrade --force-reinstall "onnxruntime==1.10.0"; then
    conda install -y -c conda-forge "onnxruntime=1.10.0" || warn "onnxruntime install failed (non-critical)"
  fi
  python -m pip install --no-cache-dir --upgrade --force-reinstall "onnx==1.10.2" "onnx-simplifier==0.4.1" || \
    warn "ONNX/onnx-simplifier install failed (non-critical for training)"

  section "Verification"
  python - <<'PY'
import torch, numpy, scipy
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
try:
  import lap; print('lap: OK')
except Exception:
  try:
    import lapx; print('lapx: OK')
  except Exception as e:
    print('lap/lapx: MISSING ->', e)
try:
  import motmetrics, filterpy
  print('motmetrics:', getattr(motmetrics, "__version__", "unknown"))
  print('filterpy:', getattr(filterpy, "__version__", "unknown"))
except Exception as e:
  print('motmetrics/filterpy issue ->', e)
try:
  import pycocotools; print('pycocotools: OK')
except Exception as e:
  print('pycocotools: MISSING ->', e)
try:
  import onnx, google.protobuf
  print('onnx:', onnx.__version__)
  print('protobuf:', google.protobuf.__version__)
  onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1,3,224,224])
  print('onnx import/use: OK')
except Exception as e:
  print('onnx/protobuf issue ->', e)
PY

  write_yolox_helper

  section "YOLOX project wiring (editable path)"
  if [ -d "${REPO_DIR}/YOLOV" ]; then
    pushd "${REPO_DIR}/YOLOV" >/dev/null
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "${REPO_DIR}/YOLOV" | tee "${SITE_PACKAGES}/yolox.pth" >/dev/null
    ok "yolox.pth written to ${SITE_PACKAGES}"
    python - <<'PY'
import yolox, torch
print('YOLOX version:', getattr(yolox, "__version__", "unknown"))
print('YOLOX path:', yolox.__file__)
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
try:
  import onnx; print('ONNX:', onnx.__version__)
except Exception:
  print('ONNX not available (optional)')
PY
    popd >/dev/null
  else
    warn "YOLOV dir not found at ${REPO_DIR}/YOLOV"
  fi

  section "Install missing dep: timm"
  python -m pip install --no-cache-dir timm==0.9.16 || warn "timm install failed"

  section "YOLOX CLI smoke test"
  if [ -f "${REPO_DIR}/YOLOV/tools/train.py" ]; then
    python "${REPO_DIR}/YOLOV/tools/train.py" --help | head -30 || warn "train.py help failed"
  else
    warn "train.py not found"
  fi

  mark_done "$step"
}


# ===== Optional: download datasets over FTP (fill your server creds) =========
download_datasets_ftp() {
  [ "${DOWNLOAD_FTP_DATASETS}" -eq 1 ] || { warn "Skipping FTP downloads"; return 0; }
  local step="ftp_datasets"
  if already_done "$step"; then warn "FTP dataset downloads previously attempted; skipping"; return 0; fi

  section "FTP dataset download (edit placeholders before enabling)"
  python - <<'PY'
from ftplib import FTP, error_perm
from pathlib import Path

FTP_HOST = "minecraftwgwg.hopto.org"   # <-- verify
FTP_USER = "ftpuser"                    # <-- verify
FTP_PASS = "ftpadmin"                   # <-- verify
REMOTE_PATH = "datasets/visdrone"
LOCAL_PATH = Path.home() / "TemporalAttentionPlayground" / "datasets" / "visdrone"
DATASETS = ["VisDrone2019-VID-train", "VisDrone2019-VID-val", "VisDrone2019-VID-test"]

def download_file_ftp(host, username, password, remote_path, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with FTP(host, timeout=30) as ftp:
        ftp.login(user=username, passwd=password)
        print(f"Connected: {host} as {username}")
        ftp.set_pasv(True)
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
        print(f"Downloaded: {remote_path} -> {local_path}")

print("Starting FTP downloads...")
for name in DATASETS:
    remote = f"{REMOTE_PATH}/{name}.zip"
    local  = LOCAL_PATH / f"{name}.zip"
    if local.exists():
        print(f"Skipping existing: {local}")
        continue
    try:
        download_file_ftp(FTP_HOST, FTP_USER, FTP_PASS, remote, local)
    except error_perm as e:
        print(f"FTP error {remote}: {e}")
    except Exception as e:
        print(f"Error {remote}: {e}")
print("All downloads attempted.")
PY
  mark_done "$step"
}

# ===== MEGAcmd installer (apt repo) =========================================
install_megacmd() {
  [ "${INSTALL_MEGACMD}" -eq 1 ] || { warn "Skipping MEGAcmd install"; return 0; }
  local step="install_megacmd"
  if already_done "$step"; then warn "MEGAcmd already installed; skipping"; return 0; fi

  if command -v mega-get >/dev/null 2>&1; then
    ok "MEGAcmd present: $(mega-version 2>/dev/null || echo 'unknown')"
    mark_done "$step"
    return 0
  fi

  # --- Detect Ubuntu release ---
  local CODENAME VERSION_ID
  CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$(lsb_release -cs 2>/dev/null)}")"
  VERSION_ID="$(. /etc/os-release && echo "${VERSION_ID:-}" | sed 's/"//g')"

  # --- Where the .deb lives (override with MEGACMD_DEB_FILE / MEGACMD_DEB_DIR if you want) ---
  local DEB_DIR="${MEGACMD_DEB_DIR:-${REPO_DIR}/setup/deps}"
  local DEB_FILE="${MEGACMD_DEB_FILE:-}"

  # Map codename -> expected .deb name (MEGA’s naming scheme)
  if [ -z "$DEB_FILE" ]; then
    case "$CODENAME" in
      noble) DEB_FILE="megacmd-xUbuntu_24.04_amd64.deb" ;;
      jammy) DEB_FILE="megacmd-xUbuntu_22.04_amd64.deb" ;;
      focal) DEB_FILE="megacmd-xUbuntu_20.04_amd64.deb" ;;
      *)     DEB_FILE="megacmd-xUbuntu_${VERSION_ID}_amd64.deb" ;; # best guess
    esac
  fi

  local DEB_PATH="$DEB_DIR/$DEB_FILE"

  # --- Ensure we don't have a broken MEGA repo messing with apt update ---
  if [ -f /etc/apt/sources.list.d/meganz.list ] || [ -f /usr/share/keyrings/meganz-archive-keyring.gpg ]; then
    section "Cleaning stale MEGA apt repo"
    sudo rm -f /etc/apt/sources.list.d/meganz.list /usr/share/keyrings/meganz-archive-keyring.gpg
  fi

  if command -v apt-get >/dev/null 2>&1; then
    section "Installing MEGAcmd from local .deb ($DEB_FILE)"
    # Preflight: needed basics
    set +e
    sudo apt-get update -y || warn "apt-get update encountered issues; continuing"
    sudo apt-get install -y --no-install-recommends ca-certificates lsb-release wget gnupg || true

    if [ ! -f "$DEB_PATH" ]; then
      err "Local package not found: $DEB_PATH"
      # Helpful hint for common mismatch
      if [ "$CODENAME" = "jammy" ] && [ -f "$DEB_DIR/megacmd-xUbuntu_24.04_amd64.deb" ]; then
        err "You are on Ubuntu 22.04 (jammy). Place megacmd-xUbuntu_22.04_amd64.deb in $DEB_DIR and rerun."
      fi
      return 1
    fi

    # Try install, then fix deps, then re-try
    sudo dpkg -i "$DEB_PATH"
    local dpkg_rc=$?
    if [ $dpkg_rc -ne 0 ]; then
      warn "dpkg reported dependency problems (rc=$dpkg_rc); attempting to fix"
      sudo apt-get -f install -y || { err "Could not fix dependencies automatically"; return 1; }
      sudo dpkg -i "$DEB_PATH" || {
        err "MEGAcmd .deb still failed to configure. Likely distro/version mismatch (you have $CODENAME, package is $DEB_FILE)."
        return 1
      }
    fi
    set -e
  else
    err "apt-get not available; please install MEGAcmd manually"
    return 1
  fi

  if ! command -v mega-get >/dev/null 2>&1; then
    err "MEGAcmd install did not provide mega-get on PATH"
    return 1
  fi

  ok "MEGAcmd installed: $(mega-version 2>/dev/null || echo 'unknown')"
  mark_done "$step"
}

# ===== MEGAcmd login/session ================================================
mega_login() {
  [ "${MEGA_LOGIN_ON_START}" -eq 1 ] || { warn "Skipping MEGA login"; return 0; }
  local step="mega_login"
  # We **don't** mark done, to allow re-login if env changes; just run idempotently.

  section "Logging into MEGA (CLI) if needed"
  if ! command -v mega-get >/dev/null 2>&1; then
    err "mega-get not found; run install_megacmd first"
    return 1
  fi

  # Start server if not up
  pgrep -f mega-cmd-server >/dev/null || (mega-cmd-server >/dev/null 2>&1 & disown || true)

  # Are we already logged in?
  if mega-whoami >/dev/null 2>&1; then
    ok "Already logged into MEGA: $(mega-whoami 2>/dev/null | tr -d '\n')"
    return 0
  fi

  # Need credentials
  if [ -z "${MEGA_EMAIL}" ] || [ -z "${MEGA_PASS}" ]; then
    err "MEGA_EMAIL/MEGA_PASS are not set. Export them before running or set in the script."
    return 1
  fi

  # Clean stale sessions and login
  mega-logout >/dev/null 2>&1 || true
  mega-login "${MEGA_EMAIL}" "${MEGA_PASS}"

  # Validate
  if mega-whoami >/dev/null 2>&1; then
    ok "MEGA login OK: $(mega-whoami 2>/dev/null | tr -d '\n')"
    mega-getquota || true
  else
    err "MEGA login failed"
    return 1
  fi
}

download_datasets_mega() {
  [ "${DOWNLOAD_VISDRONE_MEGA}" -eq 1 ] || { warn "Skipping VisDrone MEGA download"; return 0; }
  local step="visdrone_mega"
  if already_done "$step"; then warn "VisDrone MEGA download already done; skipping"; return 0; fi

  section "Downloading VisDrone via MEGA (Python helper)"

  if ! command -v mega-get >/dev/null 2>&1; then
    err "mega-get (MEGAcmd) not found. Enable INSTALL_MEGACMD=1 or install manually."
    return 1
  fi

  # Optionally ensure we’re logged in to avoid anonymous quota limits
  if [ "${MEGA_LOGIN_ON_START}" -eq 1 ]; then
    mega_login || return 1
  fi

  # Allow override of destination
  local target_root="${VISDRONE_ROOT:-$TAP_HOME/datasets/visdrone}"
  export target_root

  python - <<'PYCODE'
import os
import pathlib
import zipfile
import subprocess
import argparse
import shutil
from typing import List, Optional

# Use env-provided root if present
default_root = os.environ.get("target_root", "datasets/visdrone")
ROOT = pathlib.Path(default_root)
ROOT.mkdir(parents=True, exist_ok=True)

files = {
    "VisDrone2019-VID-train.zip": "https://mega.nz/file/4jwBBAJa#yhtv7GCulkXSqvz269Sw3cecXJUpN_2FBqNBgQ1Cn4M",
    "VisDrone2019-VID-val.zip": "https://mega.nz/file/A7pklJKJ#BhSjtVF-8DeUWlmjtNb5CEZFMkRBSOc6hMHP7pTVarA",
    "VisDrone2019-VID-test-dev.zip": "https://mega.nz/file/FrYzmIxY#OQ6qQLHYgqgHfxUcfSXDtTcjTup1QwN2_Vun6c84Kj4",
}

expected_dirs = {name: pathlib.Path(name[:-4]) for name in files.keys()}

def run(cmd: list[str], cwd: Optional[pathlib.Path] = None):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

for name, url in files.items():
    out = ROOT / name
    exp_dir = expected_dirs[name]

    # Skip if extracted exists here or in parent
    dir_here = ROOT / exp_dir
    dir_parent = ROOT.parent / exp_dir
    if dir_here.is_dir():
        print(f"[skip] already extracted: {dir_here}")
        continue
    if dir_parent.is_dir():
        print(f"[skip] already extracted in parent: {dir_parent}")
        continue

    # If zip already exists, skip download
    if out.exists():
        print(f"[skip-download] zip already exists: {out}")
    else:
        alt = ROOT.parent / name
        if alt.exists():
            print(f"[move] found existing {alt}, moving to {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(alt), str(out))
        else:
            # If you want to proceed despite quota warnings, add --ignore-quota-warn below
            run(["mega-get", url], cwd=ROOT)
            if not out.exists():
                candidate = ROOT / pathlib.Path(url).name
                if candidate.exists() and candidate != out:
                    shutil.move(str(candidate), str(out))
            if not out.exists():
                raise RuntimeError(f"Download failed or unexpected output name for {name}")

    # Unzip if folder not present
    if not (ROOT / exp_dir).exists():
        print(f"[unzip] {out}")
        with zipfile.ZipFile(out) as zz:
            zz.extractall(ROOT)
    else:
        print(f"[skip-unzip] target exists: {ROOT / exp_dir}")

    # Optional: remove zip to save space
    try:
        os.remove(out)
    except OSError:
        pass
print("All done.")
PYCODE
  mark_done "$step"
}

# ===== Convert VisDrone to Transvod format ================================
convert_visdrone_to_imagenetvid() {
  [ "${CONVERT_VISDRONE_TRANSVOD}" -eq 1 ] || { warn "Skipping VisDrone->Transvod conversion"; return 0; }
  local step="convert_visdrone_transvod"
  if already_done "$step"; then
    warn "VisDrone->TransVOD conversion already done; skipping"
    return 0
  fi

  section "Converting VisDrone to Transvod format"

  local train_dir="$VISDRONE_ROOT/VisDrone2019-VID-train"
  local val_dir="$VISDRONE_ROOT/VisDrone2019-VID-val"
  local out_dir="$VISDRONE_TRANSVOD"
  printf 'train_dir="%s"\n' "$train_dir"
  pip install pillow  # Ensure PIL is present for the converter
  # Support alternate naming in case someone used datasets-visdrone
  if [ ! -d "$train_dir" ] && [ -d "$TAP_HOME/datasets-visdrone/VisDrone2019-VID-train" ]; then
    train_dir="$TAP_HOME/datasets-visdrone/VisDrone2019-VID-train"
  fi
  if [ ! -d "$val_dir" ] && [ -d "$TAP_HOME/datasets-visdrone/VisDrone2019-VID-val" ]; then
    val_dir="$TAP_HOME/datasets-visdrone/VisDrone2019-VID-val"
  fi

  if [ ! -d "$train_dir" ] || [ ! -d "$val_dir" ]; then
    err "Missing VisDrone train/val directories. Download them first (MEGA or FTP)."
    return 1
  fi

  mkdir -p "$out_dir"

  # Run the converter script
  if [ -f "$SETUP_CODE_DIR/visdrone_transvod_builder.py" ]; then
    python3 "$SETUP_CODE_DIR/visdrone_transvod_builder.py" \
      --visdrone-train "$train_dir" \
      --visdrone-val "$val_dir" \
      --out-root "$out_dir" \
      --link-mode hardlink
  else
    err "Converter not found: $SETUP_CODE_DIR/visdrone_transvod_builder.py"
    return 1
  fi

  ok "VisDrone conversion completed → $out_dir"
  mark_done "$step"
}

# ===== Convert VisDrone to YoloV format ================================
convert_visdrone_to_yolov() {
  [ "${CONVERT_VISDRONE_YOLOV}" -eq 1 ] || { warn "Skipping VisDrone->YoloV conversion"; return 0; }
  local step="convert_visdrone_yolov"
  if already_done "$step"; then
    warn "VisDrone->YoloV conversion already done; skipping"
    return 0
  fi

  section "Converting VisDrone to YoloV format"

  local train_dir="$VISDRONE_ROOT/VisDrone2019-VID-train"
  local val_dir="$VISDRONE_ROOT/VisDrone2019-VID-val"
  local out_dir="$VISDRONE_YOLOV"
  pip install pillow  # Ensure PIL is present for the converter
  # Support alternate naming in case someone used datasets-visdrone
  if [ ! -d "$train_dir" ] && [ -d "$TAP_HOME/datasets-visdrone/VisDrone2019-VID-train" ]; then
    train_dir="$TAP_HOME/datasets-visdrone/VisDrone2019-VID-train"
  fi
  if [ ! -d "$val_dir" ] && [ -d "$TAP_HOME/datasets-visdrone/VisDrone2019-VID-val" ]; then
    val_dir="$TAP_HOME/datasets-visdrone/VisDrone2019-VID-val"
  fi

  if [ ! -d "$train_dir" ] || [ ! -d "$val_dir" ]; then
    err "Missing VisDrone train/val directories. Download them first (MEGA or FTP)."
    return 1
  fi

  mkdir -p "$out_dir"

  # Run the converter script
  if [ -f "$SETUP_CODE_DIR/visdrone_yolov_builder.py" ]; then
    python3 "$SETUP_CODE_DIR/visdrone_yolov_builder.py" \
      --visdrone-train "$train_dir" \
      --visdrone-val "$val_dir" \
      --out-root "$out_dir" \
      --link-mode hardlink
  else
    err "Converter not found: $SETUP_CODE_DIR/visdrone_yolov_builder.py"
    return 1
  fi

  ok "VisDrone conversion completed → $out_dir"
  mark_done "$step"
}

 create_yolox_annotations() {
  [ "${CREATE_YOLOX_ANNOTS}" -eq 1 ] || { warn "Skipping YOLOX annotation creation"; return 0; }
  local step="create_yolox_annots"
  if already_done "$step"; then
    warn "YOLOX annotation creation already done; skipping"
    return 0
  fi

  section "Creating YOLOX annotations for VisDrone dataset"

  local yolov_dir="$VISDRONE_YOLOV/annotations"
  if [ ! -d "$yolov_dir" ]; then
    err "Missing YoloV dataset directory. Convert VisDrone to YoloV format first."
    return 1
  fi

  if [ -f "${HELPER_SCRIPT_YOLOX:-$TAP_HOME/activate_yolox.sh}" ]; then
    # Activate YOLOX environment (sourcing makes conda activate available)
    # shellcheck disable=SC1091
    source "${HELPER_SCRIPT_YOLOX:-$TAP_HOME/activate_yolox.sh}"
  elif [ -f $TAP_HOME/activate_yolox.sh ]; then
    # In case script runs as root and helper is placed in /root
    source $TAP_HOME/activate_yolox.sh
  else
    err "Missing activate_yolox helper. Expected at ${HELPER_SCRIPT_YOLOX:-$TAP_HOME/activate_yolox.sh} or /root/activate_yolox.sh"
    return 1
  fi


  # Run the annotation creation script
  if [ -f "$SETUP_CODE_DIR/visdrone_yolox_builder.py" ]; then
    python3 "$SETUP_CODE_DIR/visdrone_yolox_builder.py"
  else
    err "Annotation creation script not found: $SETUP_CODE_DIR/visdrone_yolox_builder.py"
    return 1
  fi

  conda deactivate
  set -u

  ok "YOLOX annotation creation completed → $yolov_dir"
  mark_done "$step"
 }


create_visdrone_coco_annotations() {
  [ "${CREATE_VISDRONE_COCO_ANNOTS}" -eq 1 ] || { warn "Skipping COCO annotation creation"; return 0; }
  local step="create_visdrone_coco_annots"
  if already_done "$step"; then
    warn "COCO annotation creation already done; skipping"
    return 0
  fi

  section "Creating COCO annotations for VisDrone dataset"

  local visdrone_dir="$VISDRONE_YOLOV"
  if [ ! -d "$visdrone_dir" ]; then
    err "Missing VisDrone dataset directory. Download VisDrone first."
    return 1
  fi

  # Run the COCO annotation creation script
  if [ -f "$REPO_DIR/code/annotations/flatten-vid-annotation.py" ]; then
    python3 "$REPO_DIR/code/annotations/flatten-vid-annotation.py" "$visdrone_dir/annotations/imagenet_vid_train.json" "$visdrone_dir/annotations/imagenet_vid_train_coco.json"
    python3 "$REPO_DIR/code/annotations/flatten-vid-annotation.py" "$visdrone_dir/annotations/imagenet_vid_val.json" "$visdrone_dir/annotations/imagenet_vid_val_coco.json"
  else
    err "COCO annotation creation script not found: $REPO_DIR/code/annotations/flatten-vid-annotation.py"
    return 1
  fi

  ok "COCO annotation creation completed → $visdrone_dir/annotations"
  mark_done "$step"
}


 download_yolov_weights() {
  [ "${DOWNLOAD_YOLOV_WEIGHTS}" -eq 1 ] || { warn "Skipping YOLOV weights download"; return 0; }
  local step="download_yolov_weights"
  if already_done "$step"; then
    warn "YOLOV weights already downloaded; skipping"
    return 0
  fi

  section "Downloading YOLOV pretrained weights"

  local weights_dir="$TAP_WEIGHTS/yolov"
  local swinb_checkpoint_url="https://mega.nz/file/liBxwDwS#OIE_VxM7i2TvfxeC93C-Ptyh9VwshuGOlTe0v5vtdig"
  local v_swinBase="https://mega.nz/file/Iyx1xYLC#ByppxgdCtkhZGzssrciC74QtvlEheXCtqGUHylgm3lg"

  mkdir -p "$weights_dir"

  # Run the weights download script
  if [ -f "$SETUP_CODE_DIR mega_downloader.py" ]; then
    python3 "$SETUP_CODE_DIR/mega_downloader.py" "$swinb_checkpoint_url" "$weights_dir"/swinb_checkpoint.pth
    python3 "$SETUP_CODE_DIR/mega_downloader.py" "$v_swinBase" "$weights_dir"/v_swinBase.pth
  else
    err "Weights download script not found: $SETUP_CODE_DIR/mega_downloader.py"
    return 1
  fi

  ok "YOLOV weights downloaded → $weights_dir"
  mark_done "$step"
}

# ===== Download Swin Transformer pretrained weights ==========================
download_swin_weights() {
  [ "${DOWNLOAD_SWIN_WEIGHTS}" -eq 1 ] || { warn "Skipping Swin weights download"; return 0; }
  local step="download_swin_weights"
  if already_done "$step"; then
    warn "Swin weights already downloaded; skipping"
    return 0
  fi

  section "Downloading Swin Transformer pretrained weights"

  local weights_dir="$TAP_WEIGHTS/swin"
  mkdir -p "$weights_dir"

  local swin_base_url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
  local swin_base_file="$weights_dir/swin_base_patch4_window7_224_22k.pth"

  if [ ! -f "$swin_base_file" ]; then
    info "Downloading Swin-Base-22K pretrained weights (~418MB)..."
    wget -q --show-progress "$swin_base_url" -O "$swin_base_file"
    ok "Downloaded Swin-Base weights → $swin_base_file"
  else
    ok "Swin-Base weights already exist → $swin_base_file"
  fi

  mark_done "$step"
}

# ===== Download YOLOX COCO pretrained weights =================================
download_yolox_coco_weights() {
  [ "${DOWNLOAD_YOLOX_COCO_WEIGHTS}" -eq 1 ] || { warn "Skipping YOLOX COCO weights download"; return 0; }
  local step="download_yolox_coco_weights"
  if already_done "$step"; then
    warn "YOLOX COCO weights already downloaded; skipping"
    return 0
  fi

  section "Downloading YOLOX COCO pretrained weights"

  local weights_dir="$TAP_WEIGHTS/yolox_coco"
  local yolox_dir="$REPO_DIR/YOLOV/pretrained"
  mkdir -p "$weights_dir"
  mkdir -p "$yolox_dir"

  local yolox_l_url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
  local yolox_l_file="$weights_dir/yolox_l.pth"

  if [ ! -f "$yolox_l_file" ]; then
    info "Downloading YOLOX-L COCO pretrained weights (~415MB)..."
    wget -q --show-progress "$yolox_l_url" -O "$yolox_l_file"
    ok "Downloaded YOLOX-L weights → $yolox_l_file"
  else
    ok "YOLOX-L weights already exist → $yolox_l_file"
  fi

  # Create symlink in YOLOV/pretrained directory
  if [ ! -L "$yolox_dir/yolox_l.pth" ]; then
    ln -sf "$yolox_l_file" "$yolox_dir/yolox_l.pth"
    ok "Created symlink → $yolox_dir/yolox_l.pth"
  fi

  mark_done "$step"
}




# ===== Main ==================================================================
main() {
  section "Starting setup (idempotent). To force redo, run with FORCE_RUN=1"
  set_tap_env
  clone_repo
  write_msda_helper
  install_apt_prereqs
  install_cuda_113
  setup_miniconda
  conda_shell_hook
  setup_env_msda
  build_transvod_ops
  run_visdrone_tools
  setup_env_yolox
  #download_datasets_ftp
  install_megacmd
  mega_login
  download_datasets_mega
  convert_visdrone_to_imagenetvid
  convert_visdrone_to_yolov
  create_yolox_annotations
  download_yolov_weights
  create_visdrone_coco_annotations
  download_swin_weights
  download_yolox_coco_weights
  ok "Setup complete"
  echo "Tip: activate with 'source ${HELPER_SCRIPT_MSDA}' or 'source ${HELPER_SCRIPT_YOLOX}'"
}

main "$@"
