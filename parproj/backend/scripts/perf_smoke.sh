#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
USE_GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_DIR="$2"; shift 2;;
        --gpu)
            USE_GPU=1; shift;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT_DIR/$BUILD_DIR/graph_ml"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build first: cmake -B build -DENABLE_OPENMP=ON && cmake --build build"
    exit 1
fi

THREADS=$(nproc)
if [[ "$THREADS" -gt 4 ]]; then
    THREADS=4
fi

args=(--vertices 1000 --trees 10 --bc-samples 50 --threads "$THREADS")
if [[ "$USE_GPU" -eq 1 ]]; then
    args+=(--gpu)
fi

result_line=$(
    "$BINARY" "${args[@]}" 2>&1 | grep '^\[RESULT\]' || true
)

if [[ -z "$result_line" ]]; then
    echo "ERROR: No [RESULT] line captured"
    exit 1
fi

echo "Smoke test OK"
echo "$result_line"
