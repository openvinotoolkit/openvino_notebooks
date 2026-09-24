#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Prepares the `dfine_repo` for OpenVINO inference: clones the upstream repo
# (pinned to commit 956d170) and copies the OpenVINO-specific files listed in
# copy_files_to_dfine_repo.txt.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/dfine_repo"
UPSTREAM_URL="https://github.com/Peterande/D-FINE"
UPSTREAM_COMMIT="956d170"
COPY_LIST="$HERE/copy_files_to_dfine_repo.txt"
REQUIREMENTS="$HERE/requirements.txt"

cd "$HERE"

# 1. Clone only if missing.
if [ -d "$REPO_DIR/.git" ]; then
  echo "dfine_repo already present, skipping clone."
else
  echo "Cloning $UPSTREAM_URL ..."
  git clone "$UPSTREAM_URL" "$REPO_DIR"
fi

# 2. Pin to the known-good commit.
git -C "$REPO_DIR" checkout "$UPSTREAM_COMMIT"

# 3. Copy new files.
grep -v -E '^[[:space:]]*(#|$)' "$COPY_LIST" | while IFS= read -r f; do
  [ -n "$f" ] || continue
  mkdir -p "$REPO_DIR/$(dirname "$f")"
  cp "$HERE/$f" "$REPO_DIR/$f"
done

# 4. Install torch/torchvision from the XPU wheel index.
python -m pip install --upgrade \
  torch==2.10.0 \
  torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/xpu

# 5. Install the remaining Python requirements.
python -m pip install --upgrade -r "$REQUIREMENTS"
