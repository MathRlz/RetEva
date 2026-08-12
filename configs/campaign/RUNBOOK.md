# Campaign runbook — thesis experiments

Every config here (and every `configs/apm_tests/**`) takes machine-specific paths from the
**environment**, so one committed config runs anywhere. Set these on the experiment machine:

```bash
export ADMED_VOICE_PATH=/path/to/admed_voice          # dir holding corpus_summary_all.csv
export APM_CHECKPOINT_DIR=/path/to/attention_pool_model   # the apm_*.pt files
export ASR_ADAPTER_DIR=/path/to/medical-transcribe-pl     # LoRA adapters (admed ASR arm)
export CLAP_CHECKPOINT_DIR=/path/to/clap_style_model      # best_model.pt
export EVALUATOR_ALLOW_UNSAFE_CHECKPOINTS=1               # required only for the CLAP arm
export PYTHONHASHSEED=0                                   # recorded in provenance; set it explicitly
```

An **unset** variable is left literal in the path, so the pre-flight error names the variable
you forgot. Install `pytrec-eval-terrier` (dev extra) so the IR reference cross-check
(`tests/test_ir_reference_impl.py`, 55 cases) actually runs there instead of skipping.

**Dataset pinning.** `hani_medical` loads a pinned `repo@revision`
(`datasets/builtins/hani_medical.py:_HANI_DATASET`) so an upstream re-upload cannot move the
data under a rerun; override with `data.huggingface_dataset` if you deliberately want another
snapshot. The pubmed campaign set is committed, so it needs no pin.

**LLM reproducibility.** `llm.seed` is forwarded to the server (Ollama honors it) and
inherited by judge / answer_gen / query_optimization / query_correction — `temperature: 0`
alone does *not* pin sampling. The server's own model build/quantization is still outside the
report, so LLM-backed numbers are reproducible *per server*, not absolutely.

> **Set `trace_limit` to the dataset size, not 0.** It caps the dataset *and* gates
> per-query traces: `0` yields the full set with **no** traces, silently dropping the
> failure analysis, the per-speaker breakdown, the per-query UI panels, and the judge
> (which raises without them). The campaign configs use `trace_limit: 200`.

## Arm 1 — spoken-retrieval branch comparison (data included, run first)

| Config | What it measures |
|---|---|
| `pubmed200_3branch.yaml` | ref (oracle transcript) vs asr (Whisper) vs corr (rule correction): the ASR-degradation claim, paired deltas + CIs at n=200 |
| `pubmed200_c7_correction.yaml` | the five correctors (rule / kb / phonetic / clinical vs raw asr) on WER/CER + retrieval |

```bash
python3 scripts/build_pubmed_campaign.py -n 200      # writes examples/data/pubmed_qa_campaign
evaluator run --config configs/campaign/pubmed200_3branch.yaml
```

**Caveat to state in the chapter:** PubMedQA questions carry essentially no drug/dose
language, so `ceer` is ~0 for every branch and `ceer_rx` fires on a small minority of items.
The CEER safety story needs medication speech — Arms 2/3.

## Arm 2 — APM / cross-modal embedding variants (needs admed or hani + checkpoints)

Self-retrieval: the audio query is embedded into the text embedder's space and matched
against the transcription corpus. The interesting axes are **encoder** (Whisper-large-v3 vs
M4T), **post-pool transform** (raw attention pooling vs whitening vs ABTT), and **dataset**.

| Config | Encoder | Post-pool | Dataset | Checkpoint (`$APM_CHECKPOINT_DIR/`) |
|---|---|---|---|---|
| `apm_tests/admed/apm_admed_whisper_attention.yaml` | Whisper-large-v3 | attention | admed | `apm_whisper_jina_admed_voice.pt` |
| `apm_tests/admed/apm_admed_whisper_whiten.yaml` | Whisper-large-v3 | whitening | admed | `apm_whisper_jina_admed_voice_whitened_none.pt` |
| `apm_tests/admed/apm_admed_whisper_abtt.yaml` | Whisper-large-v3 | ABTT | admed | `apm_whisper_jina_admed_voice_abtt_none.pt` |
| `apm_tests/admed/apm_admed_m4t_attention.yaml` | M4T | attention | admed | `apm_m4t_jina_admed_voice_attention_whitening.pt` |
| `apm_tests/admed/apm_admed_m4t_whiten.yaml` | M4T | whitening | admed | `apm_m4t_jina_admed_voice_whitened_none.pt` |
| `apm_tests/admed/apm_admed_m4t_abtt.yaml` | M4T | ABTT | admed | `apm_m4t_jina_admed_voice_abtt_none.pt` |
| `apm_tests/hani/apm_hani_whisper_{attention,whiten,abtt}.yaml`, `apm_hani_m4t_whiten.yaml` | both | all three | hani | **admed-trained weights** — cross-dataset transfer; say so explicitly in the write-up |

Baselines to run alongside (no APM checkpoint needed), so the APM numbers have something to
beat:

| Config | Baseline |
|---|---|
| `evaluation_config_hani_selfretr_asr_text.yaml` | ASR → text embedding (the pipeline the APM replaces) |
| `evaluation_config_hani_selfretr_audio_emb.yaml` | audio embedding alone |
| `evaluation_config_hani_selfretr_fusion.yaml` | audio ⊕ text fusion |
| `evaluation_config_sonar_crossmodal.yaml` | SONAR speech+text (a shared-space model, no training of ours) |
| `evaluation_config_clap_admed.yaml` | CLAP-style contrastive audio-text |

## Arm 3 — LLM arms (need a local server)

```bash
ollama serve &                       # in the container/machine running the eval
ollama pull mistral:7b-instruct      # the model the campaign configs name
```

| Config | What it adds |
|---|---|
| `pubmed200_c7_llm.yaml` | the C7 `llm` corrector branch (constrained decode) beside rule/kb/phonetic/clinical |
| `pubmed200_judge.yaml` | LLM-as-judge over retrieval + answers (needs `trace_limit > 0`) |
| `pubmed200_rag_answer.yaml` | retrieval → answer generation → answer metrics (+ judge) |

Without a reachable endpoint the **judge and correction** arms fail loudly, but
**query optimization** falls back to the original query per item — the run still completes
and the report is silently unoptimized. Check `report.provenance.optimization_fallbacks`
(and the stage WARNING) before trusting any query-optimization result.

## After every run

```bash
R=evaluation_results/campaign/results_<name>.json
evaluator branch-report "$R" --out-dir figures/ --plot-format pdf   # LaTeX table + vector delta plot
evaluator export "$R" -f provenance -o figures/provenance.tex       # reproducibility table
evaluator export "$R" -f metrics-table -o figures/metrics.csv       # tidy branch×metric rows
evaluator graph --config <cfg> --format dot -o figures/dag.dot && dot -Tpdf figures/dag.dot -o figures/dag.pdf
```

Reports land in the config's `experiment.output_dir` with a `…config_resolved.yaml` sidecar
(the executed DAG) and are ingested into `leaderboard.sqlite` there; `evaluator leaderboard`
and `/ui` query across runs.
