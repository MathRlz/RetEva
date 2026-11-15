#!/bin/bash
# PubMedQA re-runs for per-query stats (2026-11-15). Slide pairs first.
cd "$(dirname "$0")/../.." && source scripts/rerun/env_rerun.sh
run() { echo "=== $(date +%T) START $1"; evaluator run --config configs/$1.yaml || echo "=== FAILED $1"; echo "=== $(date +%T) END $1"; }
pqa_en
run pubmed_qa_tts_audioemb_cmp
run pubmed_qa_tts_asr_embed_cmp
pqa_pl
run pubmed_qa_tts_audioemb_cmp_pl
run pubmed_qa_tts_asr_embed_cmp_pl
pqa_plen
run pubmed_qa_tts_audioemb_cmp_plen
run pubmed_qa_tts_asr_embed_cmp_plen
echo "=== ALL DONE $(date +%T)"
