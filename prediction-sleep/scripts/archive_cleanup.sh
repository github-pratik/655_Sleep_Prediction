#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +"%Y%m%d_%H%M%S")"
ARCHIVE_DIR="$ROOT/archives"
ARCHIVE_PATH="$ARCHIVE_DIR/unwanted_files_$TS.zip"
DO_DELETE=0

usage() {
  cat <<'EOF'
Archive and optionally delete unwanted temporary files.

Usage:
  scripts/archive_cleanup.sh [--delete]

Actions:
  1) Create zip archive under prediction-sleep/archives/
  2) If --delete is passed, remove archived files after successful zip.

Currently targets:
  - logs/phases_6_9/*
  - dataset/public/PPG-DaLiA/ppg+dalia/data.zip (redundant after extraction)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      DO_DELETE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$ARCHIVE_DIR"

declare -a TARGETS=()

if [[ -d "$ROOT/logs/phases_6_9" ]]; then
  while IFS= read -r -d '' f; do
    TARGETS+=("${f#$ROOT/}")
  done < <(find "$ROOT/logs/phases_6_9" -type f -print0)
fi

if [[ -f "$ROOT/dataset/public/PPG-DaLiA/ppg+dalia/data.zip" ]]; then
  TARGETS+=("dataset/public/PPG-DaLiA/ppg+dalia/data.zip")
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No cleanup targets found."
  exit 0
fi

(
  cd "$ROOT"
  zip -q "$ARCHIVE_PATH" "${TARGETS[@]}"
)

echo "Archive created: $ARCHIVE_PATH"
echo "Archived files:"
printf '  - %s\n' "${TARGETS[@]}"

if [[ $DO_DELETE -eq 1 ]]; then
  (
    cd "$ROOT"
    rm -f "${TARGETS[@]}"
  )
  echo "Deleted archived files."
else
  echo "No deletion performed. Re-run with --delete to remove archived files."
fi
