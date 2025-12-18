#!/bin/bash
# Build script for PointPillars Docker image
# Usage: ./devops/build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building PointPillars Docker image${NC}"

echo -e "${YELLOW}Building OpenVino E2E image ...${NC}"
docker build \
    --build-arg BUILD_TYPE=e2eOV \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg UBUNTU_VERSION=24.04 \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    -t pointpillars:e2eOV \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}" \
    "${@:2}"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully built pointpillars:e2eOV${NC}"
else
    rc=$?
    echo -e "${RED}Failed to build pointpillars:e2eOV (exit code: ${rc})${NC}" >&2
    exit ${rc}
fi
