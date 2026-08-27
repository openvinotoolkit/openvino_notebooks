#!/usr/bin/env bash
# One-shot per-file benchmark matching the OV_README "80-100s files" table.
# Each backend/file combination is a fresh process (cold start), matching the
# methodology of the previous benchmark run.
#
# Target files: the same 6 VoxConverse test files used in OV_README.md
#   lubpm (81s), eucfa (81s), jxydp (86s), tnjoh (87s), eguui (99s), nqcpi (100s)
#
# Backends on this machine (Core Ultra X7 358H + Arc B390 iGPU):
#   cpu       -> diar_cpu env
#   ov-cpu    -> diar_ov env
#   ov-gpu0   -> diar_ov env  (Arc B390 iGPU)
#
# Usage (from pyannote-audio/usage/):
#   bash run_file_benchmark.sh
#   bash run_file_benchmark.sh --files "lubpm eucfa"          # subset of files
#   bash run_file_benchmark.sh --backends "ov-gpu0"           # single backend
#   bash run_file_benchmark.sh --vox-root /path/to/voxconverse

set -euo pipefail

FILES=(lubpm eucfa jxydp tnjoh eguui nqcpi)
BACKENDS=(cpu ov-cpu ov-gpu0)
VOX_ROOT="./voxconverse"
LOG="file_benchmark.log"
LOG_EXPLICIT=0
NUM_SPEAKERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --files)    read -ra FILES    <<< "$2"; shift 2 ;;
    --backends) read -ra BACKENDS <<< "$2"; shift 2 ;;
    --vox-root) VOX_ROOT="$2"; shift 2 ;;
    --log)           LOG="$2"; LOG_EXPLICIT=1; shift 2 ;;
    --num-speakers) NUM_SPEAKERS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# If --log is not provided, encode backend/device name(s) into log filename.
if [[ "$LOG_EXPLICIT" -eq 0 ]]; then
  backend_tag="$(printf "%s_" "${BACKENDS[@]}")"
  backend_tag="${backend_tag%_}"
  backend_tag="${backend_tag//-/_}"
  LOG="file_benchmark_${backend_tag}.log"
fi

source ~/miniforge3/etc/profile.d/conda.sh

backend_to_env() {
  local backend="$1"
  if [[ "$backend" == "cpu" ]]; then
    echo "diar_cpu"
  elif [[ "$backend" == "xpu" ]]; then
    echo "diar_xpu"
  else
    echo "diar_ov"
  fi
}

build_env_list() {
  local envs=()
  local backend
  for backend in "$@"; do
    local env
    env="$(backend_to_env "$backend")"
    if [[ " ${envs[*]} " != *" $env "* ]]; then
      envs+=("$env")
    fi
  done
  echo "${envs[*]}"
}

get_ov_gpu_name() {
  local name="N/A"
  set +u
  conda activate diar_ov >/dev/null 2>&1 || true
  set -u
  name=$(python - <<'PY' 2>/dev/null || echo "N/A"
try:
    import openvino as ov
    core = ov.Core()
    devices = [d for d in core.available_devices if d.startswith("GPU")]
    if devices:
      print(core.get_property(devices[0], "FULL_DEVICE_NAME"))
    else:
      print("N/A")
except Exception:
    print("N/A")
PY
)
  set +u
  conda deactivate >/dev/null 2>&1 || true
  set -u
  echo "$name"
}

get_xpu_gpu_name() {
  local name="N/A"
  set +u
  conda activate diar_xpu >/dev/null 2>&1 || true
  set -u
  name=$(python - <<'PY' 2>/dev/null || echo "N/A"
try:
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() > 0:
      print(torch.xpu.get_device_name(0))
    else:
      print("N/A")
except Exception:
    print("N/A")
PY
)
  set +u
  conda deactivate >/dev/null 2>&1 || true
  set -u
  echo "$name"
}

OV_GPU_NAME="$(get_ov_gpu_name)"
XPU_GPU_NAME="$(get_xpu_gpu_name)"
GPU_NAME="$OV_GPU_NAME"
if [[ "$GPU_NAME" == "N/A" ]]; then
  GPU_NAME="$XPU_GPU_NAME"
fi

# find WAV path for a given file stem
find_wav() {
  local stem="$1"
  find "$VOX_ROOT" -name "${stem}.wav" 2>/dev/null | head -1
}

# extract time (seconds, float) from a "diarization for … (backend, Xs)" line
extract_time() {
  grep -oP '\d+\.\d+(?=s\))' | tail -1
}

# extract counted speakers from the final summary line: "# N speaker(s): ..."
extract_counted_spk() {
  grep -oP '^#\s+\K\d+(?=\s+speaker\(s\):)' | tail -1
}

# robust WAV duration reader that does not depend on conda env packages
get_duration() {
  local wav_path="$1"
  python - "$wav_path" <<'PY' 2>/dev/null || echo "?"
import sys

path = sys.argv[1]

# Try stdlib first (works for PCM WAVs and is environment-independent).
try:
  import wave
  with wave.open(path, "rb") as f:
    frames = f.getnframes()
    rate = f.getframerate()
    if rate > 0:
      print(round(frames / rate, 1))
      raise SystemExit(0)
except Exception:
  pass

# Fallback: soundfile if available in the active env.
try:
  import soundfile as sf
  f = sf.SoundFile(path)
  print(round(len(f) / f.samplerate, 1))
except Exception:
  raise SystemExit(1)
PY
}

# duration from RTTM annotation support (matches README table methodology)
get_rttm_duration() {
  local rttm_path="$1"
  python - "$rttm_path" <<'PY' 2>/dev/null || echo "?"
import sys
from pathlib import Path

path = Path(sys.argv[1])
end = 0.0
for line in path.read_text().splitlines():
  parts = line.split()
  if len(parts) < 5:
    continue
  try:
    end = max(end, float(parts[3]) + float(parts[4]))
  except Exception:
    pass
print(round(end, 1) if end > 0 else "?")
PY
}

echo "# File benchmark — $(date)" | tee "$LOG"
echo "# conda_env: $(build_env_list "${BACKENDS[@]}")" | tee -a "$LOG"
echo "# cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)" | tee -a "$LOG"
echo "# gpu: $GPU_NAME" | tee -a "$LOG"
echo "# backends: ${BACKENDS[*]}" | tee -a "$LOG"
echo "# files: ${FILES[*]}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# header
printf "%-10s %8s %8s %12s  " "file" "dur(s)" "GT_spk" "counted_spk" | tee -a "$LOG"
for b in "${BACKENDS[@]}"; do printf "%14s  " "$b"; done | tee -a "$LOG"
echo "" | tee -a "$LOG"
printf "%-10s %8s %8s %12s  " "----------" "--------" "--------" "------------" | tee -a "$LOG"
for b in "${BACKENDS[@]}"; do printf "%14s  " "--------------"; done | tee -a "$LOG"
echo "" | tee -a "$LOG"

declare -A TIMES  # TIMES[file,backend] = time
declare -A COUNTED  # COUNTED[file,backend] = predicted speaker count

for file in "${FILES[@]}"; do
  wav=$(find_wav "$file")
  if [[ -z "$wav" ]]; then
    echo "# WARNING: WAV not found for $file, skipping" | tee -a "$LOG"
    continue
  fi

  # get speaker count from RTTM
  rttm=$(find "$VOX_ROOT/test" -name "${file}.rttm" 2>/dev/null | head -1)
  spk="?"
  dur="?"
  if [[ -n "$rttm" ]]; then
    spk=$(awk '{print $8}' "$rttm" | sort -u | wc -l)
    dur=$(get_rttm_duration "$rttm")
  fi
  # Fallback to WAV duration when RTTM is missing/unreadable.
  if [[ "$dur" == "?" ]]; then
    dur=$(get_duration "$wav")
  fi

  for backend in "${BACKENDS[@]}"; do
    if [[ "$backend" == "cpu" ]]; then
      env="diar_cpu"
    elif [[ "$backend" == "xpu" ]]; then
      env="diar_xpu"
    else
      env="diar_ov"
    fi

    echo "# running: $backend on $file ($dur s)" >&2
    set +u; conda activate "$env"; set -u

    run_output=""

    spk_arg=()
    [[ -n "$NUM_SPEAKERS" ]] && spk_arg=(--num-speakers "$NUM_SPEAKERS")

    if [[ "$backend" == "cpu" ]]; then
      run_output=$(python run_diarization.py "$wav" "${spk_arg[@]}" 2>&1)
    elif [[ "$backend" == "xpu" ]]; then
      run_output=$(python run_diarization.py "$wav" --device xpu "${spk_arg[@]}" 2>&1)
    elif [[ "$backend" == "ov-cpu" ]]; then
      run_output=$(python run_diarization_ov.py "$wav" --device CPU "${spk_arg[@]}" 2>&1)
    elif [[ "$backend" == "ov-gpu" ]]; then
      run_output=$(python run_diarization_ov.py "$wav" --device GPU.0 "${spk_arg[@]}" 2>&1)
    elif [[ "$backend" == "ov-gpu0" ]]; then
      run_output=$(python run_diarization_ov.py "$wav" --device GPU.0 "${spk_arg[@]}" 2>&1)
    elif [[ "$backend" == "ov-gpu1" ]]; then
      run_output=$(python run_diarization_ov.py "$wav" --device GPU.1 "${spk_arg[@]}" 2>&1)
    fi

    set +u; conda deactivate; set -u
    t=$(printf "%s\n" "$run_output" | grep "diarization for" | extract_time)
    counted=$(printf "%s\n" "$run_output" | extract_counted_spk)
    TIMES["$file,$backend"]="${t:-?}"
    COUNTED["$file,$backend"]="${counted:-?}"
  done

  counted_spk="${COUNTED["$file,${BACKENDS[0]}"]:-?}"

  # print row
  printf "%-10s %8s %8s %12s  " "$file" "$dur" "$spk" "$counted_spk" | tee -a "$LOG"
  for b in "${BACKENDS[@]}"; do
    printf "%13ss  " "${TIMES[$file,$b]}" | tee -a "$LOG"
  done
  echo "" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "# done: $(date)" | tee -a "$LOG"
echo "# log: $LOG"
