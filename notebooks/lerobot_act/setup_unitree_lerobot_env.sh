#!/usr/bin/env bash
# Automated setup for Conda-based Python 3.10 environment integrating OpenVINO notebooks + Unitree LeRobot fork.
# Usage:
#   bash setup_unitree_lerobot_env.sh
set -euo pipefail

ENV_NAME="unitree_lerobot"
LEROBOT_DIR="unitree_IL_lerobot"
PY_VERSION="3.10"

log() { printf "\n[setup] %s\n" "$*"; }
err() { printf "\n[error] %s\n" "$*" >&2; }

check_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    err "Conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
  fi
}

create_env() {
  if conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    log "Conda env '${ENV_NAME}' already exists (skipping creation)."
  else
    log "Creating conda env '${ENV_NAME}' (python=${PY_VERSION})"
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
  fi
}

activate_env() {
  log "Activating env '${ENV_NAME}'"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

upgrade_tooling() {
  log "Upgrading pip/wheel/setuptools"
  python -m pip install --upgrade pip wheel setuptools
}

install_openvino_requirements() {
  # We are already inside the openvino_notebooks tree;
  # find the root (parent directories) containing requirements.txt.
  if [ -f ../../requirements.txt ]; then
    ROOT_REQ=../../requirements.txt
  elif [ -f ../requirements.txt ]; then
    ROOT_REQ=../requirements.txt
  elif [ -f requirements.txt ]; then
    ROOT_REQ=requirements.txt
  else
    err "Could not locate requirements.txt for OpenVINO notebooks. Ensure script resides under openvino_notebooks/."
    return 1
  fi
  log "Installing OpenVINO notebooks requirements from ${ROOT_REQ}"
  python -m pip install -r "${ROOT_REQ}"
}

register_kernel() {
  if jupyter kernelspec list 2>/dev/null | grep -q "${ENV_NAME}"; then
    log "Jupyter kernel '${ENV_NAME}' already registered."
  else
    log "Registering Jupyter kernel '${ENV_NAME}'"
    python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python ${PY_VERSION} (${ENV_NAME})"
  fi
}

clone_lerobot_repo() {
  if [ -d "${LEROBOT_DIR}" ]; then
    log "LeRobot repo '${LEROBOT_DIR}' already exists (skipping clone)."
  else
    log "Cloning Unitree LeRobot fork with submodules"
    git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git "${LEROBOT_DIR}"
  fi
  pushd "${LEROBOT_DIR}" >/dev/null
  log "Updating submodules"
  git submodule update --init --recursive
  popd >/dev/null
}

install_pinocchio() {
  log "Installing pinocchio via conda-forge (if not installed)"
  if python -c "import pinocchio" 2>/dev/null; then
    log "pinocchio already installed."
  else
    conda install -y pinocchio -c conda-forge || err "pinocchio install failed; continue without if unused."
  fi
}

install_lerobot_editable() {
  pushd "${LEROBOT_DIR}" >/dev/null
  if python -c "import lerobot" 2>/dev/null; then
    log "lerobot already importable (skipping editable install)."
  else
    if [ -d unitree_lerobot/lerobot ]; then
      log "Editable install of upstream component"
      pushd unitree_lerobot/lerobot >/dev/null
      python -m pip install -e . || err "Editable install (lerobot) failed"
      popd >/dev/null
    fi
    log "Editable install of root fork extras"
    python -m pip install -e .[dev] || python -m pip install -e . || err "Editable install root failed"
  fi
  popd >/dev/null
}

install_extra_packages() {
  log "Installing auxiliary packages (if missing)"
  python - <<'EOF'
import importlib, subprocess, sys
packages = [
  'safetensors','onnx','onnxruntime','openvino-dev[pot]','nncf','hydra-core','rich','tqdm','pandas','h5py'
]
for pkg in packages:
    base = pkg.split('[')[0].split('==')[0].split('>=')[0]
    try:
        importlib.import_module(base.replace('-','_'))
        print(f"[extra] OK: {pkg}")
    except Exception:
        print(f"[extra] Installing: {pkg}")
        subprocess.check_call([sys.executable,'-m','pip','install',pkg])
EOF
}

sanity_check() {
  log "Running sanity check"
  python - <<'EOF'
import sys, importlib
print('Python:', sys.version)
for pkg in ['torch','openvino','nncf','lerobot']:
    try:
        importlib.import_module(pkg)
        print('OK:', pkg)
    except Exception as e:
        print('FAIL:', pkg, e)
EOF
}

main() {
  check_conda
  create_env
  activate_env
  upgrade_tooling
  install_openvino_requirements
  register_kernel
  clone_lerobot_repo
  install_pinocchio
  install_lerobot_editable
  install_extra_packages
  sanity_check
  log "Setup complete. Launch notebook with:"
  echo "    conda activate ${ENV_NAME}"
  echo "    jupyter lab notebooks/lerobot_act/lerobot-act.ipynb --NotebookApp.kernel_name=${ENV_NAME}"
}

main
