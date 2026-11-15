#!/bin/bash
# Paired stats for the 2026-11-15 re-runs. Usage: ./run_stats.sh [primary|voices|text|pl]...
cd "$(dirname "$0")/../.." && source scripts/rerun/env_rerun.sh >/dev/null 2>&1
L() { ls -t evaluation_results/results_$1_$2_*.json | head -1; }
A=pubmed_qa_tts_asr_embed_cmp; E=pubmed_qa_tts_audioemb_cmp; S=evaluation_results/stats
mkdir -p $S
for what in "$@"; do case $what in
primary) echo "## Primary family (Holm m=3)"
  python scripts/paired_stats.py --csv $S/primary.csv \
    --pair "P1 Whisper+Jina vs SONAR speech" "$(L $A pubmed_qa)@finalize_xtts_whisper_jina"  "$(L $E pubmed_qa)@finalize_xtts_sonar" \
    --pair "P2 Whisper+SONAR vs SONAR speech" "$(L $A pubmed_qa)@finalize_xtts_whisper_sonar" "$(L $E pubmed_qa)@finalize_xtts_sonar" \
    --pair "P3 APM vs Whisper+Jina (ADMED)" "$(L admed_audioemb_cmp admed_voice)@finalize_apm_whisper" \
                                         "$(L admed_asr_embed_jina_only_cmp admed_voice)@finalize_whisper_jina" ;;
voices) echo "## Other voices (Holm m=4)"
  python scripts/paired_stats.py --csv $S/voices.csv \
    --pair "P1-m4t"   "$(L $A pubmed_qa)@finalize_m4t_whisper_jina"    "$(L $E pubmed_qa)@finalize_m4t_sonar" \
    --pair "P2-m4t"   "$(L $A pubmed_qa)@finalize_m4t_whisper_sonar"   "$(L $E pubmed_qa)@finalize_m4t_sonar" \
    --pair "P1-piper" "$(L $A pubmed_qa)@finalize_piper_whisper_jina"  "$(L $E pubmed_qa)@finalize_piper_sonar" \
    --pair "P2-piper" "$(L $A pubmed_qa)@finalize_piper_whisper_sonar" "$(L $E pubmed_qa)@finalize_piper_sonar" ;;
text) echo "## Typed vs spoken (equivalence: read the delta CI, ignore p)"
  python scripts/paired_stats.py --csv $S/text_vs_cascade.csv \
    --pair "typed Jina vs xtts Whisper+Jina"  "$(L pubmed_qa_text_embed_baseline pubmed_qa)@finalize_jina"  "$(L $A pubmed_qa)@finalize_xtts_whisper_jina" \
    --pair "typed SONAR vs xtts SONAR speech" "$(L pubmed_qa_text_embed_baseline pubmed_qa)@finalize_sonar" "$(L $E pubmed_qa)@finalize_xtts_sonar" ;;
pl) echo "## PL-100 (indicative only, Holm m=4)"
  python scripts/paired_stats.py --csv $S/pl100.csv \
    --pair "P1-pl"   "$(L ${A}_pl pubmed_qa)@finalize_xtts_whisper_jina"    "$(L ${E}_pl pubmed_qa)@finalize_xtts_sonar" \
    --pair "P2-pl"   "$(L ${A}_pl pubmed_qa)@finalize_xtts_whisper_sonar"   "$(L ${E}_pl pubmed_qa)@finalize_xtts_sonar" \
    --pair "P1-plen" "$(L ${A}_plen pubmed_qa)@finalize_xtts_whisper_jina"  "$(L ${E}_plen pubmed_qa)@finalize_xtts_sonar" \
    --pair "P2-plen" "$(L ${A}_plen pubmed_qa)@finalize_xtts_whisper_sonar" "$(L ${E}_plen pubmed_qa)@finalize_xtts_sonar" ;;
esac; echo; done
