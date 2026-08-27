#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: MIT
#
# Download the VoxCeleb1 *test* set needed to reproduce the pyannote/embedding
# EER benchmark (see benchmark_eer.py and docs/MODEL_AND_CONVERSION.md).
#
# Downloads two things into a target folder (default: ../test_audio):
#   1. veri_test.txt        - trial pairs list (public, ~1.5 MB, 37,720 pairs)
#   2. vox1_test_wav.zip     - test audio (~1 GB, 4,874 wavs) from a HF mirror
# and extracts the audio to  <target>/vox1/wav/<id>/<clip>/<utt>.wav
#
# Usage:
#   ./download_voxceleb_test.sh [TARGET_DIR]
#
# Requirements: wget, unzip, and the `hf` CLI (from huggingface_hub, already in
# the ov_pyan / ov_xpu envs). A Hugging Face login is recommended:  hf auth login

set -euo pipefail

# Default to the current working directory so a notebook or terminal run can
# download into the folder it is already using, without depending on a parent
# directory like ../test_audio.
TARGET_DIR="${1:-.}"
TRIALS_URL="https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test.txt"
HF_REPO="ProgramComputer/voxceleb"
ZIP_REL="vox1/vox1_test_wav.zip"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"
echo ">> Target directory: $(pwd)"

# --- 1. Trial pairs list ----------------------------------------------------
if [[ -f veri_test.txt ]]; then
  echo ">> veri_test.txt already present, skipping download."
else
  echo ">> Downloading trial list ..."
  wget -q --show-progress -O veri_test.txt "${TRIALS_URL}"
fi
echo ">> Trial pairs: $(wc -l < veri_test.txt)"

# --- 2. Test audio ----------------------------------------------------------
if [[ -d vox1/wav ]]; then
  echo ">> vox1/wav already present, skipping audio download/extract."
else
  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found. Activate an env with huggingface_hub:" >&2
    echo "       source ~/miniforge3/bin/activate ov_pyan" >&2
    exit 1
  fi
  if [[ ! -f "${ZIP_REL}" ]]; then
    echo ">> Downloading test audio (~1 GB) from ${HF_REPO} ..."
    hf download "${HF_REPO}" "${ZIP_REL}" --repo-type dataset --local-dir .
  else
    echo ">> ${ZIP_REL} already downloaded, skipping."
  fi
  echo ">> Extracting audio ..."
  unzip -q "${ZIP_REL}" -d vox1
fi

# --- 3. Summary -------------------------------------------------------------
WAV_COUNT=$(find vox1/wav -name '*.wav' 2>/dev/null | wc -l)
echo ">> Done. WAV files: ${WAV_COUNT}"
echo ">> Layout:"
echo "     $(pwd)/veri_test.txt"
echo "     $(pwd)/vox1/wav/<id>/<clip>/<utt>.wav"
echo ">> Run the benchmark, e.g.:"
echo "     python benchmark_eer.py --backend openvino --device CPU \\"
echo "       --trials ${TARGET_DIR}/veri_test.txt --wav-root ${TARGET_DIR}/vox1/wav"
