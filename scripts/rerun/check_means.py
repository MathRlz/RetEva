"""Print Recall@1 per variant of the newest result file per config, next to August's value."""
# Usage (from repo root): python3 scripts/rerun/check_means.py <config_name>...
import glob
import json
import sys

AUG = {  # August Recall@1 (local copies of the Aug 29-31 results)
    "admed_audioemb_cmp": {"finalize_apm_whisper": 0.885, "finalize_apm_m4t": 0.0, "finalize_eam": 0.821, "finalize_sonar": 0.9},
    "admed_asr_embed_jina_only_cmp": {"finalize_whisper_jina": 0.899, "finalize_m4t_jina": 0.881,
                                      "finalize_wav2vec2_jina": 0.75, "finalize_whisper_lora_jina": 0.971},
    "pubmed_qa_tts_audioemb_cmp": {"finalize_xtts_sonar": 0.524, "finalize_m4t_sonar": 0.491, "finalize_piper_sonar": 0.495,
                                   "finalize_xtts_apm_whisper": 0.061, "finalize_xtts_eam": 0.06},
    "pubmed_qa_tts_asr_embed_cmp": {"finalize_xtts_whisper_jina": 0.989, "finalize_xtts_whisper_sonar": 0.551,
                                    "finalize_piper_whisper_jina": 0.99, "finalize_m4t_m4t_jina": 0.838},
}
for cfg in sys.argv[1:]:
    files = sorted(glob.glob(f"evaluation_results/results_{cfg}_*.json"))
    files = [f for f in files if f"results_{cfg}_" in f and "_pl_" not in f[len(f"evaluation_results/results_{cfg}"):][:4]]
    if not files:
        print(cfg, "no result yet"); continue
    d = json.load(open(files[-1]))
    print("==", cfg, files[-1][-30:])
    for k, v in d["variants"].items():
        has = "item_scores" in v.get("report", {})
        aug = AUG.get(cfg, {}).get(k)
        flag = "" if aug is None else ("OK" if abs(v["Recall@1"] - aug) < 0.0015 else f"DIFF (Aug {aug})")
        print(f"  {k:32s} R@1 {v['Recall@1']:.3f}  item_scores={has}  {flag}")
