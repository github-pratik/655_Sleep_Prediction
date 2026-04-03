#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR_DEFAULT="$ROOT/logs/phases_6_9/$TIMESTAMP"

SEARCH_ROOT="$ROOT"
BOOTSTRAP=500
SKIP_LATENCY=0
SKIP_PLOTS=1
PPG_MAX_SUBJECTS=3
LOG_DIR="$LOG_DIR_DEFAULT"

SLEEPACCEL_ROOTS=()
PPG_DALIA_ROOTS=()

usage() {
  cat <<'EOF'
Run Phases 6->9 in one command with per-step logs.

Usage:
  scripts/run_phases_6_9.sh [options]

Options:
  --sleepaccel-root PATH   Repeatable. Local SleepAccel root.
  --ppg-dalia-root PATH    Repeatable. Local PPG-DaLiA root.
  --search-root PATH       Search root for step6 auto-discovery (default: repo root).
  --bootstrap N            Bootstrap samples for step9 (default: 500).
  --skip-latency           Pass --skip-latency to step7.
  --with-plots             Do not pass --skip-plots to step8.
  --ppg-max-subjects N     Max PPG-DaLiA subjects in step6 (default: 3).
  --log-dir PATH           Override log output directory.
  -h, --help               Show this help.

Environment:
  PYTHON_BIN               Python interpreter to use (default: .venv/bin/python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sleepaccel-root)
      SLEEPACCEL_ROOTS+=("$2")
      shift 2
      ;;
    --ppg-dalia-root)
      PPG_DALIA_ROOTS+=("$2")
      shift 2
      ;;
    --search-root)
      SEARCH_ROOT="$2"
      shift 2
      ;;
    --bootstrap)
      BOOTSTRAP="$2"
      shift 2
      ;;
    --skip-latency)
      SKIP_LATENCY=1
      shift
      ;;
    --with-plots)
      SKIP_PLOTS=0
      shift
      ;;
    --ppg-max-subjects)
      PPG_MAX_SUBJECTS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_step() {
  local step_name="$1"
  shift
  local log_file="$LOG_DIR/${step_name}.log"

  echo "[$(date +"%Y-%m-%d %H:%M:%S")] START ${step_name}"
  (
    cd "$ROOT"
    "$@"
  ) 2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAIL  ${step_name} (exit ${status})"
    echo "Log: $log_file"
    exit "$status"
  fi
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE  ${step_name}"
}

echo "Pipeline root: $ROOT" | tee "$LOG_DIR/summary.log"
echo "Python: $PYTHON_BIN" | tee -a "$LOG_DIR/summary.log"
echo "Logs: $LOG_DIR" | tee -a "$LOG_DIR/summary.log"

has_sleepaccel=0
has_ppg_dalia=0
if [[ ${SLEEPACCEL_ROOTS+set} == "set" ]] && [[ ${#SLEEPACCEL_ROOTS[@]} -gt 0 ]]; then
  has_sleepaccel=1
fi
if [[ ${PPG_DALIA_ROOTS+set} == "set" ]] && [[ ${#PPG_DALIA_ROOTS[@]} -gt 0 ]]; then
  has_ppg_dalia=1
fi

if [[ $has_sleepaccel -eq 1 || $has_ppg_dalia -eq 1 ]]; then
  step6_cmd=(
    "$PYTHON_BIN" "scripts/step6_public_pretrain.py"
    "--search-root" "$SEARCH_ROOT"
    "--output-dir" "artifacts/public_pretrain"
    "--ppg-max-subjects" "$PPG_MAX_SUBJECTS"
    "--save-window-table"
  )
  for root in ${SLEEPACCEL_ROOTS+"${SLEEPACCEL_ROOTS[@]}"}; do
    step6_cmd+=("--sleepaccel-root" "$root")
  done
  for root in ${PPG_DALIA_ROOTS+"${PPG_DALIA_ROOTS[@]}"}; do
    step6_cmd+=("--ppg-dalia-root" "$root")
  done
  run_step "step6_public_pretrain" "${step6_cmd[@]}"
else
  {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] SKIP step6_public_pretrain"
    echo "Reason: no --sleepaccel-root/--ppg-dalia-root provided."
    echo "Provide dataset roots to execute Phase 6 with real public data."
  } | tee "$LOG_DIR/step6_public_pretrain.log"
fi

step7_cmd=(
  "$PYTHON_BIN" "scripts/step7_transfer_finetune.py"
  "--model-data" "dataset/model_data.csv"
  "--encoder" "artifacts/public_pretrain/public_encoder.pkl"
)
if [[ $SKIP_LATENCY -eq 1 ]]; then
  step7_cmd+=("--skip-latency")
fi
run_step "step7_transfer_finetune" "${step7_cmd[@]}"

step8_cmd=("$PYTHON_BIN" "scripts/step8_distill_mobile.py")
if [[ $SKIP_PLOTS -eq 1 ]]; then
  step8_cmd+=("--skip-plots")
fi
run_step "step8_distill_mobile" "${step8_cmd[@]}"

run_step "step9_research_eval" \
  "$PYTHON_BIN" "scripts/step9_research_eval.py" \
  "--bootstrap" "$BOOTSTRAP"

{
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] PIPELINE COMPLETED"
  echo "Key outputs:"
  echo "  - artifacts/public_pretrain/public_encoder.pkl (if step6 ran)"
  echo "  - reports/public_transfer_report.json"
  echo "  - reports/distillation_report.json"
  echo "  - reports/robustness_metrics.json"
} | tee -a "$LOG_DIR/summary.log"
