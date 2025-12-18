#!/bin/bash
# Benchmark OpenVINO PointPillars E2E Pipeline
# Tests voxelization, neural network, and post-processing individually and combined
#
# Prerequisites:
#   - Virtual environment activated (auto-activated if .venv exists)
#   - Extension built against the same OpenVINO version as in venv
#     To rebuild: source .venv/bin/activate && cd ov_extensions && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
#
# Usage:
#   ./benchmark-e2eOV.sh [CONFIG] [DEVICE] [NITER] [NIREQ] [NSTREAMS] [MODE]
#
# Arguments:
#   CONFIG    - Path to config JSON (default: pretrained/pointpillars_full_config.json)
#   DEVICE    - OpenVINO device: CPU, GPU (default: CPU)
#   NITER     - Number of iterations (default: 1000)
#   NIREQ     - Number of async requests (default: 2)
#   NSTREAMS  - Number of inference streams (default: 1)
#   MODE      - Benchmark mode: all, voxel, nn, postproc, combined (default: all)

set -e

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Default arguments
CONFIG="${1:-pretrained/pointpillars_full_config.json}"
DEVICE="${2:-CPU}"
NITER="${3:-1000}"
NIREQ="${4:-2}"
NSTREAMS="${5:-1}"
MODE="${6:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenVINO PointPillars E2E Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Config:       ${CONFIG}"
echo -e "  Device:       ${DEVICE}"
echo -e "  Iterations:   ${NITER}"
echo -e "  Async Reqs:   ${NIREQ}"
echo -e "  Streams:      ${NSTREAMS}"
echo -e "  Mode:         ${MODE}"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Extract paths from config JSON using Python
echo -e "${YELLOW}Parsing configuration...${NC}"
PARSED=$(python -c "
import json
import sys
with open('$CONFIG', 'r') as f:
    config = json.load(f)
print(config['extension_lib'])
print(config['voxel_model'])
print(config['nn_model'])
print(config['postproc_model'])
")

# Read parsed values into variables
EXT_LIB=$(echo "$PARSED" | sed -n '1p')
VOXEL_MODEL=$(echo "$PARSED" | sed -n '2p')
NN_MODEL=$(echo "$PARSED" | sed -n '3p')
POSTPROC_MODEL=$(echo "$PARSED" | sed -n '4p')

echo -e "  Extension:    ${EXT_LIB}"
echo -e "  Voxel Model:  ${VOXEL_MODEL}"
echo -e "  NN Model:     ${NN_MODEL}"
echo -e "  Postproc:     ${POSTPROC_MODEL}"
echo ""

# Check if files exist
if [ ! -f "$EXT_LIB" ]; then
    echo -e "${RED}Error: Extension library not found: $EXT_LIB${NC}"
    echo -e "${YELLOW}Hint: Build the extension first:${NC}"
    echo -e "  cd ov_extensions && rm -rf build && bash build.sh && cd .."
    exit 1
fi

for model in "$VOXEL_MODEL" "$NN_MODEL" "$POSTPROC_MODEL"; do
    if [ ! -f "$model" ]; then
        echo -e "${RED}Error: Model not found: $model${NC}"
        exit 1
    fi
    # Create empty .bin file if it doesn't exist (for custom ops with no weights)
    bin_file="${model%.xml}.bin"
    if [ ! -f "$bin_file" ]; then
        echo -e "${YELLOW}Creating empty weight file: $bin_file${NC}"
        touch "$bin_file"
    fi
done

# Create results directory
RESULTS_DIR="./benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# Common benchmark_app flags
# Note: -hint none is required to use fine-tune options like -nstreams, -nireq in OpenVINO 2025.x
BENCHMARK_FLAGS="-d $DEVICE -niter $NITER -nireq $NIREQ -report_type detailed_counters -report_folder $RESULTS_DIR"

# Function to run benchmark
run_benchmark() {
    local name=$1
    local model=$2
    local use_extension=$3
    local shape_override=$4  # Optional: input shape override for dynamic models

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Benchmarking: ${name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Build shape argument if provided
    local shape_arg=""
    if [ -n "$shape_override" ]; then
        shape_arg="-shape $shape_override"
    fi

    # Run benchmark with both throughput and latency hints
    # "throughput" "latency"
    for hint in "none -nstreams $NSTREAMS"; do
        echo -e "${YELLOW}Running with -hint ${hint}${NC}"

        if [ "$use_extension" = "true" ]; then
            echo -e "Command: benchmark_app -m $model $BENCHMARK_FLAGS -hint $hint -extensions $EXT_LIB $shape_arg"
            echo ""
            benchmark_app -m "$model" $BENCHMARK_FLAGS -hint $hint -extensions "$EXT_LIB" $shape_arg | tee -a "$RESULTS_DIR/${name}_output.txt"
        else
            echo -e "Command: benchmark_app -m $model $BENCHMARK_FLAGS -hint $hint $shape_arg"
            echo ""
            benchmark_app -m "$model" $BENCHMARK_FLAGS -hint $hint $shape_arg | tee -a "$RESULTS_DIR/${name}_output.txt"
        fi
        echo ""
    done

    echo -e "${GREEN}✓ ${name} benchmark complete${NC}"
    echo ""
}

# Run benchmarks based on mode
case "$MODE" in
    "voxel")
        # Voxelization has dynamic input [?,4], use typical point cloud size
        run_benchmark "voxelization" "$VOXEL_MODEL" "true" "points_input[20000,4]"
        ;;
    "nn")
        # NN has dynamic pillar inputs
        run_benchmark "neural_network" "$NN_MODEL" "false" "pillars[16000,32,4],coors[16000,4],npoints[16000]"
        ;;
    "postproc")
        # Postprocessing expects 3D inputs: [n_anchors*nclasses, H, W]
        # For PointPillars: 6 anchors (3 classes * 2 rotations), 3 classes
        #   cls_preds:     [6*3, H, W] = [18, 248, 216]
        #   box_preds:     [6*7, H, W] = [42, 248, 216]
        #   dir_cls_preds: [6*2, H, W] = [12, 248, 216]
        # H=248 (y: 79.36/0.16/2), W=216 (x: 69.12/0.16/2)
        run_benchmark "postprocessing" "$POSTPROC_MODEL" "true" "bbox_cls_pred[18,248,216],bbox_pred[42,248,216],bbox_dir_cls_pred[12,248,216]"
        ;;
    "combined")
        echo -e "${YELLOW}Note: benchmark_app runs models individually.${NC}"
        echo -e "${YELLOW}For true E2E latency, use profile-e2eOV.py instead.${NC}"
        echo ""
        run_benchmark "1_voxelization" "$VOXEL_MODEL" "true" "points_input[20000,4]"
        run_benchmark "2_neural_network" "$NN_MODEL" "false" "pillars[16000,32,4],coors[16000,4],npoints[16000]"
        run_benchmark "3_postprocessing" "$POSTPROC_MODEL" "true" "bbox_cls_pred[18,248,216],bbox_pred[42,248,216],bbox_dir_cls_pred[12,248,216]"
        ;;
    "all")
        run_benchmark "1_voxelization" "$VOXEL_MODEL" "true" "points_input[20000,4]"
        run_benchmark "2_neural_network" "$NN_MODEL" "false" "pillars[16000,32,4],coors[16000,4],npoints[16000]"
        run_benchmark "3_postprocessing" "$POSTPROC_MODEL" "true" "bbox_cls_pred[18,248,216],bbox_pred[42,248,216],bbox_dir_cls_pred[12,248,216]"
        ;;
    *)
        echo -e "${RED}Error: Unknown mode: $MODE${NC}"
        echo -e "Valid modes: all, voxel, nn, postproc, combined"
        exit 1
        ;;
esac

# Generate summary report
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Benchmark Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Results saved to: ${RESULTS_DIR}"
echo ""
echo -e "Quick summary (extract from outputs):"
total_median_latency=0
for output_file in "$RESULTS_DIR"/*_output.txt; do
    if [ -f "$output_file" ]; then
        name=$(basename "$output_file" _output.txt)
        echo -e "${YELLOW}${name}:${NC}"

        # Extract Throughput from last match (final benchmark run)
        throughput=$(grep "Throughput:" "$output_file" 2>/dev/null | tail -1 | sed 's/.*Throughput:/Throughput:/')
        if [ -n "$throughput" ]; then
            echo "  $throughput"
        fi

        # Extract Latency block from last occurrence (final benchmark run)
        # Get line number of last "Latency:" occurrence and print next 4 lines
        latency_line=$(grep -n "Latency:" "$output_file" 2>/dev/null | tail -1 | cut -d: -f1)
        if [ -n "$latency_line" ]; then
            echo "  Latency:"
            # Extract lines and remove "[ INFO ]" prefix
            sed -n "$((latency_line+1)),$((latency_line+4))p" "$output_file" | sed 's/\[ INFO \]//g' | sed 's/^/   /'

            # Extract median latency value for E2E calculation
            median_latency=$(sed -n "$((latency_line+1))p" "$output_file" | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [ -n "$median_latency" ]; then
                total_median_latency=$(python -c "print($total_median_latency + $median_latency)")
            fi
        fi
        echo ""
    fi
done

# Calculate and display E2E metrics
if [ "$MODE" = "all" ] || [ "$MODE" = "combined" ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}End-to-End Performance (Sum of Median Latencies)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    is_valid=$(python -c "print(1 if $total_median_latency > 0 else 0)")
    if [ "$is_valid" -eq 1 ]; then
        e2e_fps=$(python -c "print(f'{1000 / $total_median_latency:.2f}')")
        echo -e "  Total Latency: ${GREEN}${total_median_latency} ms${NC}"
        echo -e "  E2E Throughput: ${GREEN}${e2e_fps} FPS${NC}"
    else
        echo -e "  ${RED}Could not calculate E2E latency${NC}"
    fi
    echo ""
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "View detailed reports:"
echo -e "  ls -lh $RESULTS_DIR"
echo ""
