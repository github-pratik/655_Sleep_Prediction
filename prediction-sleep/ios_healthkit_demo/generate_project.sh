#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "xcodegen not found. Install with: brew install xcodegen"
  exit 1
fi

xcodegen generate
echo "Generated: $SCRIPT_DIR/SleepFatigueEdgeDemo.xcodeproj"

if command -v open >/dev/null 2>&1; then
  open "$SCRIPT_DIR/SleepFatigueEdgeDemo.xcodeproj" || true
fi
