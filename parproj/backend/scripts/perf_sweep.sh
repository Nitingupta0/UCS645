#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
RUNS=3
USE_GPU=0
TREES=30
BC_SAMPLES=100
THREADS_OVERRIDE="${THREADS:-}"
SIZES_OVERRIDE="${SIZES:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_DIR="$2"; shift 2;;
        --runs)
            RUNS="$2"; shift 2;;
        --gpu)
            USE_GPU=1; shift;;
        --threads)
            THREADS_OVERRIDE="$2"; shift 2;;
        --sizes)
            SIZES_OVERRIDE="$2"; shift 2;;
        --trees)
            TREES="$2"; shift 2;;
        --bc-samples)
            BC_SAMPLES="$2"; shift 2;;
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

if [[ ! "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
    echo "ERROR: --runs must be a positive integer"
    exit 1
fi

THREAD_COUNTS=(1 2 4 8 16 32)
GRAPH_SIZES=(1000 5000 10000 20000)

if [[ -n "$THREADS_OVERRIDE" ]]; then
    IFS=',' read -r -a THREAD_COUNTS <<< "$THREADS_OVERRIDE"
fi
if [[ -n "$SIZES_OVERRIDE" ]]; then
    IFS=',' read -r -a GRAPH_SIZES <<< "$SIZES_OVERRIDE"
fi

MAX_THREADS=$(nproc)
FILTERED_THREADS=()
for t in "${THREAD_COUNTS[@]}"; do
    if [[ "$t" -le "$MAX_THREADS" ]]; then
        FILTERED_THREADS+=("$t")
    fi
done
if [[ ${#FILTERED_THREADS[@]} -eq 0 ]]; then
    FILTERED_THREADS=(1)
fi
THREAD_COUNTS=("${FILTERED_THREADS[@]}")

RAW_OUT="$ROOT_DIR/perf_raw.csv"
SUMMARY_OUT="$ROOT_DIR/perf_summary.csv"
EFF_OUT="$ROOT_DIR/perf_efficiency.csv"

printf "run,threads,vertices,edges,total_ms,accuracy,f1\n" > "$RAW_OUT"

TOTAL_RUNS=$(( ${#THREAD_COUNTS[@]} * ${#GRAPH_SIZES[@]} * RUNS ))
RUN_ID=0

echo "═══════════════════════════════════════════════════════"
echo "  Perf Sweep: $TOTAL_RUNS runs"
echo "  Threads: ${THREAD_COUNTS[*]}"
echo "  Sizes:   ${GRAPH_SIZES[*]}"
echo "  Runs:    $RUNS"
echo "  Trees:   $TREES   BC samples: $BC_SAMPLES"
if [[ "$USE_GPU" -eq 1 ]]; then
    echo "  GPU:     enabled"
else
    echo "  GPU:     disabled"
fi
echo "═══════════════════════════════════════════════════════"
echo ""

for size in "${GRAPH_SIZES[@]}"; do
    for threads in "${THREAD_COUNTS[@]}"; do
        for r in $(seq 1 "$RUNS"); do
            RUN_ID=$((RUN_ID + 1))
            echo "[$RUN_ID/$TOTAL_RUNS] v=$size t=$threads run=$r ..."

            args=(--vertices "$size" --trees "$TREES" --bc-samples "$BC_SAMPLES" --threads "$threads")
            if [[ "$USE_GPU" -eq 1 ]]; then
                args+=(--gpu)
            fi

            result_line=$(
                "$BINARY" "${args[@]}" 2>&1 | grep '^\[RESULT\]' || true
            )

            if [[ -z "$result_line" ]]; then
                echo "  WARNING: No [RESULT] line captured, skipping"
                continue
            fi

            clean=$(echo "$result_line" | sed 's/^\[RESULT\] //')
            edges=$(echo "$clean" | tr ',' '\n' | awk -F= '$1=="edges" {print $2}')
            total_ms=$(echo "$clean" | tr ',' '\n' | awk -F= '$1=="total_ms" {print $2}')
            accuracy=$(echo "$clean" | tr ',' '\n' | awk -F= '$1=="accuracy" {print $2}')
            f1=$(echo "$clean" | tr ',' '\n' | awk -F= '$1=="f1" {print $2}')

            if [[ -z "$edges" || -z "$total_ms" ]]; then
                echo "  WARNING: Missing metrics in [RESULT], skipping"
                continue
            fi

            printf "%s,%s,%s,%s,%s,%s,%s\n" \
                "$r" "$threads" "$size" "$edges" "$total_ms" "$accuracy" "$f1" \
                >> "$RAW_OUT"
        done
    done
done

echo ""
echo "Generating summary CSV..."
printf "threads,vertices,edges,avg_total_ms,best_total_ms,avg_accuracy,avg_f1,throughput_edges_per_ms\n" > "$SUMMARY_OUT"

awk -F',' 'NR>1 {
    key=$2","$3
    count[key]++
    sum_total[key]+=$5
    sum_acc[key]+=$6
    sum_f1[key]+=$7
    edges[key]=$4
    if (best[key]=="" || $5 < best[key]) best[key]=$5
}
END {
    for (k in count) {
        split(k,a,",")
        threads=a[1]
        vertices=a[2]
        avg_total=sum_total[k]/count[k]
        avg_acc=sum_acc[k]/count[k]
        avg_f1=sum_f1[k]/count[k]
        throughput=(avg_total>0 ? edges[k]/avg_total : 0)
        printf "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.6f\n", \
            threads, vertices, edges[k], avg_total, best[k], avg_acc, avg_f1, throughput
    }
}' "$RAW_OUT" | sort -t, -k2,2n -k1,1n >> "$SUMMARY_OUT"

printf "threads,vertices,avg_total_ms,speedup,efficiency\n" > "$EFF_OUT"

awk -F',' 'NR>1 {
    key=$1","$2
    avg_total[key]=$4
    if ($1==1) base[$2]=$4
}
END {
    for (k in avg_total) {
        split(k,a,",")
        threads=a[1]
        vertices=a[2]
        if (base[vertices] > 0) {
            speedup=base[vertices]/avg_total[k]
            efficiency=speedup/threads
        } else {
            speedup=0
            efficiency=0
        }
        printf "%s,%s,%.4f,%.4f,%.4f\n", threads, vertices, avg_total[k], speedup, efficiency
    }
}' "$SUMMARY_OUT" | sort -t, -k2,2n -k1,1n >> "$EFF_OUT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Raw results:     $RAW_OUT"
echo "  Summary results: $SUMMARY_OUT"
echo "  Efficiency:      $EFF_OUT"
echo "═══════════════════════════════════════════════════════"
