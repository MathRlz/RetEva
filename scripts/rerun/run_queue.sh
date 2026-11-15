#!/bin/bash
# Queue worker: several workers (one per GPU group) drain one config queue in priority order.
# Usage: run_queue.sh <queue_dir> <cuda_visible_devices>
# Queue = one empty file per config, named "<NN>_<config>"; `mv` claims it atomically.
cd "$(dirname "$0")/../.." && source scripts/rerun/env_rerun.sh && pqa_en
Q=$1; export CUDA_VISIBLE_DEVICES=$2
mkdir -p "$Q/running" "$Q/done"
while true; do
  job=$(ls "$Q" | grep -E '^[0-9]+_' | sort | head -1)
  [ -z "$job" ] && break
  mv "$Q/$job" "$Q/running/$job" 2>/dev/null || continue   # another worker took it
  cfg=${job#*_}
  echo "=== $(date +%T) START $cfg (GPUs $2)"
  start=$(date +%s)
  evaluator run --config configs/$cfg.yaml || echo "=== FAILED $cfg"
  # A run whose nodes all failed (e.g. OOM) still exits 0 and writes only failed_nodes — catch that.
  res=$(find evaluation_results -maxdepth 1 -name "results_${cfg}_*.json" -newermt "@$start" | head -1)
  python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('variants') else 1)" "$res" 2>/dev/null \
    || echo "=== FAILED $cfg (no variants in result: ${res:-none})"
  echo "=== $(date +%T) END $cfg"
  mv "$Q/running/$job" "$Q/done/$job"
done
echo "=== WORKER DONE $(date +%T) (GPUs $2)"
