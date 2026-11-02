#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Prepares the `omnix_repo` for OpenVINO inference: clones the upstream repo
# (pinned to commit 6f2716d) and applies omnix_changes.patch.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/omnix_repo"
UPSTREAM_URL="https://github.com/Perkins729/OmniXtreme"
UPSTREAM_COMMIT="6f2716d"
PATCH="$HERE/omnix_changes.patch"
REQUIREMENTS="$HERE/requirements.txt"

cd "$HERE"

# 1. Clone only if missing.
if [ -d "$REPO_DIR/.git" ]; then
  echo "omnix_repo already present, skipping clone."
else
  echo "Cloning $UPSTREAM_URL ..."
  git clone "$UPSTREAM_URL" "$REPO_DIR"
fi

# 2. Pin to the commit.
git -C "$REPO_DIR" checkout "$UPSTREAM_COMMIT"

# 3. Apply the OpenVINO patch unless it is already applied.
if git -C "$REPO_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "omnix_changes.patch already applied, skipping."
else
  git -C "$REPO_DIR" apply "$PATCH"
fi

# 4. Install torch from the CPU wheel index.
python -m pip install --upgrade \
  torch==2.14.0 \
  --index-url https://download.pytorch.org/whl/cpu

# 5. Install the remaining Python requirements.
python -m pip install --upgrade -r "$REQUIREMENTS"
