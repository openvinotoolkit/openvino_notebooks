#!/usr/bin/env bash
set -euo pipefail

# Download VoxConverse data into a score_der.py-compatible layout.
#
# Resulting layout (default DEST=./voxconverse):
#   DEST/dev/*.rttm
#   DEST/test/*.rttm
#   DEST/voxconverse_dev_wav/**/*.wav
#   DEST/voxconverse_test_wav/**/*.wav
#
# Usage:
#   bash download_voxconverse.sh
#   bash download_voxconverse.sh --dest /path/to/voxconverse
#   bash download_voxconverse.sh --skip-audio
#   bash download_voxconverse.sh --skip-annotations

DEST="$(pwd)/voxconverse"
SKIP_AUDIO=0
SKIP_ANNOTATIONS=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --skip-audio)
      SKIP_AUDIO=1
      shift
      ;;
    --skip-annotations)
      SKIP_ANNOTATIONS=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Download VoxConverse v0.3 annotations + audio.

Options:
  --dest PATH          Destination root (default: ./voxconverse)
  --skip-audio         Download only annotations
  --skip-annotations   Download only audio
  --force              Re-download/re-extract over existing files
  -h, --help           Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DEST"

# Prefer curl, fallback to wget.
download_file() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    echo "# exists, skip: $out"
    return
  fi

  echo "# download: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

extract_zip() {
  local zip_path="$1"
  local out_dir="$2"

  if [[ -d "$out_dir" && "$FORCE" -eq 0 ]]; then
    echo "# exists, skip extract: $out_dir"
    return
  fi

  mkdir -p "$out_dir"
  echo "# extract: $zip_path -> $out_dir"
  unzip -q -o "$zip_path" -d "$out_dir"
}

if [[ "$SKIP_ANNOTATIONS" -eq 0 ]]; then
  TMP_REPO="$DEST/.voxconverse_annotations_repo"

  if [[ -d "$TMP_REPO" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      rm -rf "$TMP_REPO"
    else
      echo "# using existing annotations repo clone: $TMP_REPO"
    fi
  fi

  if [[ ! -d "$TMP_REPO" ]]; then
    echo "# clone annotations repo (master has v0.3 RTTM fixes)"
    git clone --depth 1 https://github.com/joonson/voxconverse.git "$TMP_REPO"
  fi

  mkdir -p "$DEST/dev" "$DEST/test"

  echo "# copy RTTM annotations"
  find "$TMP_REPO/dev" -maxdepth 1 -type f -name '*.rttm' -print0 | xargs -0 -I{} cp -f "{}" "$DEST/dev/"
  find "$TMP_REPO/test" -maxdepth 1 -type f -name '*.rttm' -print0 | xargs -0 -I{} cp -f "{}" "$DEST/test/"

  DEV_COUNT=$(find "$DEST/dev" -maxdepth 1 -type f -name '*.rttm' | wc -l)
  TEST_COUNT=$(find "$DEST/test" -maxdepth 1 -type f -name '*.rttm' | wc -l)
  echo "# annotations ready: dev=$DEV_COUNT test=$TEST_COUNT"
fi

if [[ "$SKIP_AUDIO" -eq 0 ]]; then
  DEV_ZIP="$DEST/voxconverse_dev_wav.zip"
  TEST_ZIP="$DEST/voxconverse_test_wav.zip"

  DEV_URL="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
  TEST_URL="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"

  download_file "$DEV_URL" "$DEV_ZIP"
  download_file "$TEST_URL" "$TEST_ZIP"

  extract_zip "$DEV_ZIP" "$DEST/voxconverse_dev_wav"
  extract_zip "$TEST_ZIP" "$DEST/voxconverse_test_wav"
fi

echo ""
echo "# done"
echo "# dataset root: $DEST"
echo "# score command example:"
echo "python score_der.py --dataset voxconverse --subset test --backend cpu --vox-root '$DEST'"
