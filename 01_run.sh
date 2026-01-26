#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function print_usage {
  echo "Usage: $0 <benchmark> [args...]"
  echo "  <benchmark> maps to a script named run_<benchmark>.shm or run_<benchmark>.sh"
  echo "  Remaining args are forwarded to the benchmark script."
}

if [ $# -lt 1 ]; then
  print_usage
  exit 1
fi

benchmark="$1"
shift

script_candidates=(
  "$SCRIPT_DIR/scripts/run_${benchmark}.shm"
  "$SCRIPT_DIR/scripts/run_${benchmark}.sh"
)

target_script=""
for candidate in "${script_candidates[@]}"; do
  if [ -f "$candidate" ]; then
    target_script="$candidate"
    break
  fi
done

if [ -z "$target_script" ]; then
  echo "Unknown benchmark '$benchmark'. Expected a script named run_${benchmark}.shm or run_${benchmark}.sh."
  print_usage
  exit 1
fi

bash "$target_script" "$@"
