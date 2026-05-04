#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BINARY="$BUILD_DIR/graph_ml"
OUTPUT="benchmark_results.csv"

THREAD_COUNTS=(1 2 4 8)
GRAPH_SIZES=(1000 5000 10000 20000)
TREES=30
BC_SAMPLES=100

if [ ! -x "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Build first: cmake -B build -DENABLE_OPENMP=ON && cmake --build build"
    exit 1
fi

echo "threads,vertices,edges,ingest_ms,features_ms,preproc_ms,train_ms,infer_ms,total_ms,accuracy,f1" > "$OUTPUT"

total_runs=$(( ${#THREAD_COUNTS[@]} * ${#GRAPH_SIZES[@]} ))
run=0

echo "═══════════════════════════════════════════════════════"
echo "  Benchmark: $total_runs runs"
echo "  Threads: ${THREAD_COUNTS[*]}"
echo "  Sizes:   ${GRAPH_SIZES[*]}"
echo "  Trees:   $TREES   BC samples: $BC_SAMPLES"
echo "═══════════════════════════════════════════════════════"
echo ""

for size in "${GRAPH_SIZES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        run=$((run + 1))
        echo "[$run/$total_runs] vertices=$size threads=$threads ..."

        result_line=$("$BINARY" \
            --vertices "$size" \
            --trees "$TREES" \
            --bc-samples "$BC_SAMPLES" \
            --threads "$threads" \
            2>&1 | grep '^\[RESULT\]' || echo "")

        if [ -z "$result_line" ]; then
            echo "  WARNING: No [RESULT] line captured, skipping"
            continue
        fi

        csv_line=$(echo "$result_line" \
            | sed 's/^\[RESULT\] //' \
            | tr ',' '\n' \
            | sed 's/^[^=]*=//' \
            | paste -sd, -)
        echo "$csv_line" >> "$OUTPUT"
        echo "  Done: $(echo "$result_line" | sed 's/\[RESULT\] //')"
    done
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Benchmark complete. Results: $OUTPUT"
echo "  Visualize: python3 scripts/visualize_metrics.py --csv $OUTPUT"
echo "═══════════════════════════════════════════════════════"
