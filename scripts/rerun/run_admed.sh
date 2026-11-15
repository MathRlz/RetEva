#!/bin/bash
# ADMEDVoice re-runs (GPUs 0-2).
cd "$(dirname "$0")/../.." && source scripts/rerun/env_rerun.sh && export CUDA_VISIBLE_DEVICES=0,1,2
run() { echo "=== $(date +%T) START $1"; evaluator run --config configs/$1.yaml || echo "=== FAILED $1"; echo "=== $(date +%T) END $1"; }
run admed_audioemb_cmp
run admed_asr_embed_jina_only_cmp
echo "=== ALL DONE $(date +%T)"
