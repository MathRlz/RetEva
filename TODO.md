# TODO — article + presentation (after supervisor review, 2026-11-15)

Order: section 1 first. It is the one hard requirement for a bdb grade, and it is independent
of the APM-M4T and EAM bugs in sections 4–5, because neither variant is in a key pair.

## 1. Statistical analysis  [MUST: slide + article Section 5]

### Supervisor spec
- 95% CI for Recall@1 and MRR. Paired bootstrap: resample questions with replacement, 10,000 resamples.
  An interval on the main comparison bars is enough.
- Paired test cascade vs latent: exact McNemar on Recall@1 hits (each question hit/miss under both paths).
- Holm correction across the comparisons (at least mention it).
- PL-100 set: show the wide CIs (about ±8–10 pts at R@1 ≈ 0.5) openly → supports "indicative only".

### Why a re-run is needed
Result JSONs in `evaluation_results/` hold only `{mean, n}` per metric. Per-query scores exist
in memory during a run (`item_scores` in `evaluator/evaluation/aggregate.py`) but are never written.
No `answers.jsonl` from those runs survived. So: persist per-query scores, re-run, then compute offline.

### Key pairs → result variants (match TTS voice on both sides!)
TTS has 3 voices: `xtts`, `m4t`, `piper`. A pair is valid only on the same audio.
Headline voice = `xtts` (abstract's 0.524 is `finalize_xtts_sonar`). Other two voices go to an appendix table.

| # | Pair (A vs B) | Dataset | A variant (config) | B variant (config) | Current R@1 A / B |
|---|---|---|---|---|---|
| P1 | Whisper+Jina v4 vs SONAR speech | PubMedQA 1000 | `finalize_xtts_whisper_jina` (`pubmed_qa_tts_asr_embed_cmp`) | `finalize_xtts_sonar` (`pubmed_qa_tts_audioemb_cmp`) | 0.989 / 0.524 |
| P2 | SONAR-space cascade vs SONAR speech | PubMedQA 1000 | `finalize_xtts_whisper_sonar` (`pubmed_qa_tts_asr_embed_cmp`) | `finalize_xtts_sonar` (`pubmed_qa_tts_audioemb_cmp`) | 0.551 / 0.524 |
| P3 | APM vs Whisper+Jina v4 | ADMEDVoice (1626) | `finalize_apm_whisper` (`admed_audioemb_cmp`) | `finalize_whisper_jina` (`admed_asr_embed_jina_only_cmp`) | 0.885 / 0.899 |
| ctx | typed text vs cascade (0.988 vs 0.990 claim) | PubMedQA 1000 | `finalize_jina` (`pubmed_qa_text_embed_baseline`) | `finalize_piper_whisper_jina` | 0.988 / 0.990 |
| PL | P1, P2 on Polish | PL-100 | `*_pl` / `*_plen` configs | same | CIs only, flagged |

P2 is the cleanest test of "skip the transcript": same SONAR space, only the query path differs.
P1 mixes path and embedding space, so the gap there is mostly Jina vs SONAR.

### Holm family
- Primary family = P1, P2, P3 on headline voice → Holm over m = 3. That's the slide.
- Secondary = same pairs on `m4t` + `piper` voices, plus PL-100 → separate Holm family, appendix.
- Text vs cascade (ctx) is an equivalence claim, not a difference → report ΔR@1 CI only. Equal = CI contains 0 and is narrow.

### Steps
- [ ] **1a. Persist per-query scores.** In `build_report` (`evaluator/evaluation/aggregate.py`),
      write `report["item_scores"] = {branch: {metric: {query_id: score}}}`, after variant rollup.
      Saved automatically into results JSON + per-variant `metrics.json` (`results_io.py` dumps the report whole).
      Size: about 1000 q × 15 metrics × 30 variants, a few MB per file. Acceptable.
      *Draft already in working tree (uncommitted, untested), see "State" below.*
- [ ] **1b. Stats helpers** in `evaluator/analysis/significance.py`, next to `benjamini_hochberg`:
      `mcnemar_exact(hits_a, hits_b) -> (b, c, p)` via `scipy.stats.binomtest(b, b+c, 0.5)`
      (exact, because discordant counts can be small), and `holm(p_values)`.
      Bootstrap: reuse `bootstrap_confidence_interval(values, n_bootstrap=10_000, random_state=0)`.
      Paired bootstrap of a difference = that function applied to per-query diffs `B_i − A_i`.
- [ ] **1c. Offline script `scripts/paired_stats.py`.** Input: `--pair LABEL A_SPEC B_SPEC` (repeatable),
      spec = `PATH@VARIANT/BRANCH`. Join on shared query ids, warn on dropped ids.
      Per pair output: n, R@1 A/B with CI, ΔR@1 with paired CI, b/c discordant counts,
      McNemar p, Holm p, MRR A/B with CI, ΔMRR with paired CI. Writes a markdown table + CSV.
      Hit definition: `recall@1 > 0`, meaning the top-1 doc is relevant, which also works with several relevant docs.
- [ ] **1d. Test** `tests/test_paired_stats.py`: McNemar known value (b=8, c=2 → p = 112/1024),
      Holm known values, `build_report` carries `item_scores`, script end to end on 2 synthetic JSONs.
- [ ] **1e. Re-run 4 configs + PL-100 ones** (`pubmed_qa_tts_asr_embed_cmp`, `pubmed_qa_tts_audioemb_cmp`,
      `admed_audioemb_cmp`, `admed_asr_embed_jina_only_cmp`, `pubmed_qa_text_embed_baseline`, `*_pl`, `*_plen`).
      Sanity: re-run means must equal the Aug 29–31 means. Any drift → find out why before trusting the stats.
      ⚠ This machine is CPU-only. ASR over 3 voices × 1000 q is slow. Run on the GPU box used in August, or trim configs to the needed variants.
- [ ] **1f. Run script**, 2 calls: the primary family (P1–P3, xtts), then the secondary family. Keep the CSVs in `evaluation_results/stats/`.
- [ ] **1g. Slide**: main comparison bars (R@1, MRR) with 95% CI whiskers. Table P1–P3 with Δ [CI], McNemar p, Holm p.
      Footnote: "paired bootstrap, 10,000 resamples; exact McNemar on R@1 hits; Holm, m = 3".
      PL-100 bars on the same slide or the next one, with visibly wide whiskers and an "indicative only" label.
- [ ] **1h. Article**: fill the Section 5 `[TODO: CI]` slots. Add "(p < 0.001, McNemar)" after 0.524 in the abstract,
      only if P1 confirms it. One methods paragraph on the tests and the Holm family.

### Effort per step
GPU times come from the Aug 29–31 run logs, on a multi-GPU box (`cuda:0..3`). "Your time" means hands-on time, not waiting.

| Step | Code | Tests | Machine time | Your time |
|---|---|---|---|---|
| 0. venv for Python 3.14 (`pip install -e .[dev]`, torch) | none | full suite, 143 files, about 30 s offline | about 15 min download | 15 min |
| 1a. persist `item_scores` | 10 lines, draft done | full suite, because the report shape changes (streaming-equivalence, judge, golden tests) | 30 s | 15 min review |
| 1b. McNemar + Holm | 27 lines, draft done | 2 unit tests | <1 s | 10 min review |
| 1c. `paired_stats.py` | 130 lines, draft done | 1 end-to-end test, plus one dry run on a real re-run file | <1 s | 30 min review |
| 1d. tests | 65 lines, 4 tests, draft done | itself | <5 s | covered above |
| 1e. re-runs | 0 lines, configs used as-is | sanity check of means against August, one short one-off command | about 2 h on GPU, see below | 30–60 min setup and copying |
| 1f. run stats | 0 lines, 2 commands | — | about 1 min (10k bootstrap, 6 CIs per pair) | 10 min |
| 1g. slide chart | about 40–60 lines, extend `evaluation_results/make_paper_charts.py` (CI whiskers) | visual check | seconds | 1–2 h with slide |
| 1h. article text | none | — | — | 1–2 h |

**1e machine time, from the August logs:**

| Config | GPU time |
|---|---|
| `pubmed_qa_tts_asr_embed_cmp` (27 variants) | 45 min |
| `pubmed_qa_tts_audioemb_cmp` | 9 min |
| `pubmed_qa_text_embed_baseline` | 3 min |
| `admed_asr_embed_jina_only_cmp` | not logged; `admed_asr_embed_cmp` took 35 min, so expect 15–20 min |
| `admed_audioemb_cmp` | 5 min |
| PL + PLEN: text 2×1.5, audioemb 2×4, asr 2×12 | 35 min |
| **Total, sequential** | **about 2 h** |

On this CPU-only machine the time was not measured. Expect many hours or more, so use the GPU box.
Trimming the ASR config to xtts plus the needed variants would cut the 45 min, but it breaks the appendix grid. Not worth it.

**Section 1 total:** about 6–8 h hands-on plus about 2 h of GPU time. That fits 1–2 working days.
The critical path is GPU box access for 1e. Steps 0 and 1a–1d can be finished before then.

### Rough effort, other sections (less certain)

| Section | Code | Machine time | Your time |
|---|---|---|---|
| 4. APM-M4T bug | 0–20 lines of fix, plus a 1 h sanity script | 5 min re-run of `admed_audioemb_cmp` | 2–6 h, open-ended debugging |
| 5. EAM Polish lang code | likely 1–5 lines | about 10 min PL re-runs | 1–3 h |
| 6. LoRA leakage | about 20-line id-overlap check | seconds | 1 h; a re-split means retraining LoRA, days |
| 7. Presentation edits | — | — | 1–3 h |
| 8. Article: training details, number fixes, refs, highlights | — | — | 1–2 days |
| 8. Real spoken queries | recording plus a dataset loader of about 50 lines | about 30 min of runs | 1–2 weeks, depends on speakers |

If 4 or 5 changes numbers, re-run only the affected config, not all of 1e.

### Runbook: local → push → server
Remote: `origin` = github.com/MathRlz/RetEva. `origin/clean` = local `clean` (e88370b), so a push is a fast-forward.

**A. Local: what is left to code before the push**
- [ ] Review the 4 drafts (1a–1d). No other code is needed for the runs: configs are used as-is and CIs come from the offline script.
- [ ] The slide chart (1g) waits until the stats CSVs exist. It is not part of this push.
- [ ] Tests cannot run locally (Python 3.14 env broken), so they run on the server first (step B).
- [ ] Commit on `clean` with `git commit -F msg.txt`, then `git push origin clean`.

**B. Server: pull + test (about 5 min). Stop if anything fails.**
```bash
cd <repo> && git fetch origin && git checkout clean && git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # only if models are cached there
python -m pytest -q -p no:cacheprovider tests/test_paired_stats.py      # new tests
python -m pytest -q -p no:cacheprovider                                 # full suite: report shape changed
```

**C. Server: smoke run (5 min). Confirms item_scores land on disk before spending 2 h.**
```bash
evaluator run --config configs/admed_audioemb_cmp.yaml
python -c "import json,glob; f=sorted(glob.glob('evaluation_results/results_admed_audioemb_cmp_*.json'))[-1]; v=json.load(open(f))['variants']['finalize_apm_whisper']; print(f, list(v['report']['item_scores']), round(v['Recall@1'],3))"
# expect: a branch list (not KeyError) and 0.885
```

**D. Server: remaining re-runs (about 2 h). Run inside tmux.**
```bash
for c in pubmed_qa_text_embed_baseline pubmed_qa_tts_audioemb_cmp pubmed_qa_tts_asr_embed_cmp \
         admed_asr_embed_jina_only_cmp \
         pubmed_qa_tts_audioemb_cmp_pl pubmed_qa_tts_asr_embed_cmp_pl \
         pubmed_qa_tts_audioemb_cmp_plen pubmed_qa_tts_asr_embed_cmp_plen; do
  evaluator run --config configs/$c.yaml || echo "FAILED $c"
done 2>&1 | tee rerun_$(date +%F).log
```
Sanity: each Recall@1 must equal the "Current R@1" column in the key-pairs table above. On drift, stop and find the cause.

**E. Server: stats (about 1 min).** `L` picks the newest result file for config `$1` and dataset `$2`.
The dataset part keeps `..._cmp_` from also matching `..._cmp_pl_`.
```bash
L() { ls -t evaluation_results/results_$1_$2_*.json | head -1; }
A=pubmed_qa_tts_asr_embed_cmp; E=pubmed_qa_tts_audioemb_cmp; mkdir -p evaluation_results/stats

# 1. primary family (slide), Holm m=3
python scripts/paired_stats.py --csv evaluation_results/stats/primary.csv \
  --pair "P1 Whisper+Jina vs SONAR"    "$(L $A pubmed_qa)@finalize_xtts_whisper_jina"  "$(L $E pubmed_qa)@finalize_xtts_sonar" \
  --pair "P2 SONAR cascade vs SONAR"   "$(L $A pubmed_qa)@finalize_xtts_whisper_sonar" "$(L $E pubmed_qa)@finalize_xtts_sonar" \
  --pair "P3 APM vs Whisper+Jina (AD)" "$(L admed_audioemb_cmp admed_voice)@finalize_apm_whisper" \
                                       "$(L admed_asr_embed_jina_only_cmp admed_voice)@finalize_whisper_jina"

# 2. other voices, appendix, Holm m=4
python scripts/paired_stats.py --csv evaluation_results/stats/voices.csv \
  $(for v in m4t piper; do echo --pair "P1-$v" "$(L $A pubmed_qa)@finalize_${v}_whisper_jina"  "$(L $E pubmed_qa)@finalize_${v}_sonar" \
                                 --pair "P2-$v" "$(L $A pubmed_qa)@finalize_${v}_whisper_sonar" "$(L $E pubmed_qa)@finalize_${v}_sonar"; done)

# 3. PL-100 (indicative only), Holm m=4
python scripts/paired_stats.py --csv evaluation_results/stats/pl100.csv \
  $(for s in pl plen; do echo --pair "P1-$s" "$(L ${A}_$s pubmed_qa)@finalize_xtts_whisper_jina"  "$(L ${E}_$s pubmed_qa)@finalize_xtts_sonar" \
                              --pair "P2-$s" "$(L ${A}_$s pubmed_qa)@finalize_xtts_whisper_sonar" "$(L ${E}_$s pubmed_qa)@finalize_xtts_sonar"; done)

# 4. typed vs spoken: equivalence, read the ΔR@1 CI only and ignore p
python scripts/paired_stats.py --csv evaluation_results/stats/text_vs_cascade.csv \
  --pair "typed vs cascade" "$(L pubmed_qa_text_embed_baseline pubmed_qa)@finalize_jina" "$(L $A pubmed_qa)@finalize_piper_whisper_jina"
```
⚠ The `$(for …)` lines word-split on spaces in labels. That is why labels 2–3 have no spaces.

**F. Copy back (after EVERY finished run, not just at the end).** Done so far: `evaluation_results/rerun_2026-11-15/`.
```bash
D=evaluation_results/rerun_$(date +%F); H=student5@172.20.2.204:scripts/RetEva
rsync -a --prune-empty-dirs --include='*/' --include='results_*.json' --include='results_*.yaml' \
      --include='stats/***' --include='variants/***' --exclude='*' $H/evaluation_results/ $D/
rsync -a $H/logs $H/rerun_*.log $H/env_rerun.sh $H/run_*.sh $D/
```
Old note: Copy `evaluation_results/stats/` plus the new result JSONs to local (rsync/scp).
Charts (1g) can run on the server too, because the local env has no matplotlib until the venv is fixed.

### State (2026-11-15, 17:45) — steps 1a–1f DONE
- Scripts in repo: `scripts/rerun/` (env_rerun.sh, run_pubmed.sh, run_admed.sh, run_stats.sh, check_means.py) and
  `scripts/make_stats_charts.py` (slide charts → `evaluation_results/rerun_2026-11-15/charts/`).
- Code committed + pushed (8875cbf); tests green on server (py3.12 conda `ai-env`).
- All re-runs finished on the server (EN 1000 TTS, ADMED, PL/PLEN 100, text baseline). Results + logs + stats copied to
  `evaluation_results/rerun_2026-11-15/` (tables: `stats/summary_en.md`, `stats/summary_pl.md`; CSVs alongside).
- Reproduction vs August: xtts + m4t voices **exact**; piper drifts 0.002–0.04 (voice model re-downloaded) → headline uses xtts.
- Results (Recall@1, xtts, Holm m=3):
  - P1 Whisper+Jina 0.989 vs SONAR speech 0.524: Δ −0.465 [−0.495, −0.434], b/c 465/0, p<0.001
  - P2 Whisper+SONAR 0.551 vs SONAR speech 0.524: Δ −0.027 [−0.048, −0.006], b/c 69/42, Holm p 0.026 (m4t/piper: −0.06, p<0.001)
  - P3 ADMED APM 0.885 vs Whisper+Jina 0.898: Δ +0.013 [−0.006, 0.031], b/c 99/120, p 0.18 → not significant
  - typed vs cascade (Jina): Δ +0.001 [−0.002, 0.005], 3 discordant → equivalent. SONAR speech vs typed SONAR: −0.066, p<0.001
  - ADMED SONAR speech 0.900 vs Whisper+Jina 0.898: Δ −0.002 [−0.020, 0.017] → equal on real speech
  - PL-100: P1 Δ −0.31 (pl) / −0.22 (plen), p<0.001; P2 Δ −0.02 / −0.05, n.s.; CIs ±9 pts ("indicative only")
- Left: 1g slide charts, 1h article text. Abstract: "(p < 0.001, McNemar)" after 0.524 is confirmed.
- Local test env still broken (py3.14); tests run on the server.

## 4. APM-SeamlessM4T at chance (Recall@1 0.000 on ADMEDVoice)
Ch.4 says stable (cosine 0.951 test), Ch.5 says collapsed → inference bug suspected.
- [ ] Check checkpoint actually loaded (path, variant: mean/attention × none/ABTT/whitening).
- [ ] Check pooling + post-processing at inference match training (ABTT/whitening stats reused, not refit).
- [ ] Check encoder layer / hidden-state index matches training export.
- [ ] Sanity: embed a few training utterances, cosine vs their text target should be ~0.95.
- [ ] Fix or drop variant from paper; update Ch.4/Ch.5 wording to agree.

## 5. EAM on Polish queries (0.21 → 0.01)
**Likely cause found 2026-11-15: wrong checkpoint, not language code.** August runs all used the same path
`model_ema/eam_alignment.pt`, but the file changed between Aug 29 and Aug 30. Re-run with `model_base_final`
(which reproduces Aug 29 EN exactly: text EAM 0.069, latent xtts 0.060) gives PL-100 latent EAM R@1 0.34/0.28/0.25
(pl, xtts/m4t/piper) and 0.32/0.29/0.31 (plen), vs August 0.01 and the PL text branch 0.32.
The Aug 30 file was probably ADMED-trained: ADMED EAM was 0.821 then, but 0.001 with `model_base_final` now.
- [ ] Find the Aug 30 (ADMED-trained) EAM checkpoint, or retrain; until then ADMED EAM 0.821 is not reproducible.
- [ ] Article/thesis: report EAM per checkpoint (general vs ADMED-trained), not as "fails on Polish".
- [ ] Grep language code passed to SONAR/SeamlessM4T speech encoder for `language: "pl"` queries
      (`evaluator/models/`, `test_eam_alignment.py`). Suspect `eng` hardcoded.
- [ ] Re-run PL-100 EAM after fix; compare with text branch (0.32).

## 6. Leakage: Whisper-LoRA test split vs adapter training data
- [ ] Compare ADMEDVoice ids in `../data_med_pl_dataset` split used by
      `whisper_adapter_natural_anonym_synth_cv_fl` vs eval set. Re-split or state + exclude row.

## 7. Presentation  (`Magisterka obrona.pptx`, 14 slides, Polish; charts are PNG images)

Charts come from one script (server, matplotlib), reading the stats CSVs + per-query result JSONs.
Whiskers = 95% bootstrap CI on every marker. Global rules:
- One color meaning across all slides: **query path** (text green, cascade blue, latent orange). Today chart 1 uses green = text while chart 2 uses green = Jina, which conflicts.
- Labels horizontal; label only the numbers you talk about. No "najlepszy": name the exact config.
- Title = conclusion, e.g. "Kaskada nic nie traci; ścieżka latentna zależy od przestrzeni".

- [ ] **New slide 11a "Czy różnice są istotne?"** (MUST, bdb condition). Forest plot of ΔR@1 (B−A) with 95% CI, line at 0.
      Rows: P1 Whisper+Jina→SONAR speech, P2 Whisper+SONAR→SONAR speech (cleanest test: same space), P3 ADMED APM→Whisper+Jina;
      below a separator: typed→cascade (equivalence, no p). Table to the right: Δ [CI], Holm p.
      Footer: "bootstrap parowany 10 000 prób · test McNemara (dokładny) na trafieniach R@1 · korekta Holma, m = 3".
      Headline sentence written after the numbers are in. b/c counts go in the speaker notes.
      Speaker note (~20 s): pairing → only discordant questions count → McNemar → bootstrap → Holm for 3 comparisons.
- [ ] **Backup slide**: full table (R@1, MRR both sides + CIs, b/c, raw + Holm p), plus the same plot for the other voices and PL-100.
- [ ] **Slide 10, chart 1 (R@1/5/10 per model, `image1.png`)** → dot plot, one row per embedding space, one marker per path:
      ```
      Jina v4   ▲────────────────────────●■   tekst 0.99 · kaskada 0.99 · APM 0.06
      SONAR     ──────────────▲─■─●            0.59 · 0.55 · 0.52
      LaBSE     ───────────────■●              0.63 · 0.61 · —
      ```
      Shows the thesis in one glance: Jina's cascade is lossless and its latent path collapses; in SONAR all three paths are close.
      R@5/R@10 → backup table, or R@10 as a faint tick. Drops 24 bars, 24 rotated labels and the "odcień słupka" legend.
- [ ] **Slide 10, chart 2 (synthesizer effect, `image3.png`)** → dot plot: row = ASR × embedder, 3 colored dots = TTS voice (XTTS/M4T/Piper).
      Tight dots = voice doesn't matter; spread shows voice effect, compared against the CI (±3 pts at n=1000).
      Annotate the outlier M4T TTS → M4T ASR (Jina 0.84 vs ≥0.95 elsewhere): the same model hearing its own synthetic speech.
      Replaces the cryptic `m4t-m4t` labels.
- [ ] **Slide 11 (PL-100, `image5/6.png`)**: add whiskers (wide, ±8–10 pts) + corner label "n = 100, orientacyjnie".
      APM (SeamlessM4T) 0.00 bars: remove the series, or label "błąd potoku – w trakcie wyjaśniania" (see §4). Removing it is cleaner.
- [ ] Editorial notes from supervisor (separate list, end of her message).

## 8. Article (outside repo, tracked here for completeness)
- [ ] Fill Section 3 `[TODO]`: LR, batch size, epochs, τ schedule, loss weights, data2vec λ, ADMEDVoice split sizes.
- [ ] Fix thesis/table mismatches: M4T µ/σ 0.98 vs 0.91; HuBERT FLEURS 64.9 vs 64.5; Eq. 3.2 L2 norm squared.
- [ ] Figures redrawn in English from thesis plots (`evaluation_results/make_*_charts.py`).
- [ ] References: add lab's Polish medical ASR papers; verify marked entries.
- [ ] Highlights file (name contains "highlights"); graphical abstract from Fig. 1.
- [ ] Decide: keep or drop RAG paragraph.
- [ ] Real spoken queries: ~100 questions × 5–10 speakers PL/EN if feasible; else keep ADMEDVoice prominent.
