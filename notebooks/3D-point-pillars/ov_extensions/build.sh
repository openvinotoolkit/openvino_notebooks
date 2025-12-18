#!/bin/bash
# Build OpenVINO PointPillars extensions

set -e

BUILD_TYPE=${1:-Release}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Find OpenVINO installation
OPENVINO_DIR=$(python -c "import openvino; import os; print(os.path.dirname(openvino.__file__))")

# Configure and build
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DOpenVINO_DIR="${OPENVINO_DIR}/cmake"

if make -j$(nproc); then
    echo "✓ OpenVINO extensions built successfully"
    echo "  Library: ${BUILD_DIR}/libov_pointpillars_extensions.so"
else
    rc=$?
    echo "Failed to build OpenVINO extensions (exit code: ${rc})" >&2
    exit ${rc}
fi
