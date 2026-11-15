#!/usr/bin/env python3
"""Clean vs degraded audio, per query (re-run 2026-11-15). Markdown on stdout.

Usage: python3 scripts/noise_stats.py <rerun_dir> [condition ...]   (default condition: noisy)

For each system: R@1 clean, R@1 degraded, paired Δ [95% CI], McNemar p, share of quality retained.
Then cascade vs latent under the same degradation. Picks the newest result file that has variants
(aborted runs leave files with only failed_nodes).
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extra_stats import boot_ci, fci, fp, mcnemar, paired, scores  # noqa: E402


def newest_valid(d, prefix):
    for f in sorted(glob.glob(f"{d}/results_{prefix}_*.json"), reverse=True):
        v = json.load(open(f)).get("variants")
        if v:
            return v
    return None


def section(title, pairs):
    out = [f"### {title}", "",
           "| System | R@1 czysty | R@1 zakłócony | Δ [95% CI] | b / c | McNemar p | zachowane |",
           "|---|---|---|---|---|---|---|"]
    for name, clean, noisy in pairs:
        if clean is None or noisy is None:
            out.append(f"| {name} | — | — | brak wyników | — | — | — |")
            continue
        _, hc, hn = paired(scores(*clean), scores(*noisy))
        d = hn - hc
        b, c, p = mcnemar(hc > 0, hn > 0)
        out.append(f"| {name} | {hc.mean():.3f} | {hn.mean():.3f} | {d.mean():+.3f} {fci(*boot_ci(d))} | "
                   f"{b} / {c} | {fp(p)} | {hn.mean() / hc.mean():.0%} |")
    return out + [""]


def vs_section(title, pairs):
    out = [f"### {title}", "", "| Para (A → B) | R@1 A | R@1 B | Δ [95% CI] | b / c | McNemar p |",
           "|---|---|---|---|---|---|"]
    for name, a, b_ in pairs:
        if a is None or b_ is None:
            out.append(f"| {name} | — | — | brak wyników | — | — |")
            continue
        _, ha, hb = paired(scores(*a), scores(*b_))
        d = hb - ha
        b, c, p = mcnemar(ha > 0, hb > 0)
        out.append(f"| {name} | {ha.mean():.3f} | {hb.mean():.3f} | {d.mean():+.3f} {fci(*boot_ci(d))} | {b} / {c} | {fp(p)} |")
    return out + [""]


def run(d, cond):
    pa, pl = newest_valid(d, "pubmed_qa_tts_asr_embed_cmp_pubmed_qa"), newest_valid(d, "pubmed_qa_tts_audioemb_cmp_pubmed_qa")
    na, nl = newest_valid(d, f"pubmed_qa_tts_asr_embed_{cond}_cmp_pubmed_qa"), newest_valid(d, f"pubmed_qa_tts_audioemb_{cond}_cmp_pubmed_qa")
    aa, al = newest_valid(d, "admed_asr_embed_cmp_admed_voice"), newest_valid(d, "admed_audioemb_cmp_admed_voice")
    naa, nal = newest_valid(d, f"admed_asr_embed_{cond}_cmp_admed_voice"), newest_valid(d, f"admed_audioemb_{cond}_cmp_admed_voice")
    k = lambda v, key: (v, key) if v and key in v else None  # noqa: E731
    L = [f"## Warunek: `{cond}`", ""]
    L += section("PubMedQA 1000, głos XTTS: czysty → zakłócony", [
        ("kaskada Whisper + Jina", k(pa, "finalize_xtts_whisper_jina"), k(na, "finalize_xtts_whisper_jina")),
        ("kaskada Whisper + SONAR", k(pa, "finalize_xtts_whisper_sonar"), k(na, "finalize_xtts_whisper_sonar")),
        ("latentny SONAR speech", k(pl, "finalize_xtts_sonar"), k(nl, "finalize_xtts_sonar")),
    ])
    L += vs_section("PubMedQA, sygnał zakłócony: kaskada → latentny", [
        ("Whisper + Jina → SONAR speech", k(na, "finalize_xtts_whisper_jina"), k(nl, "finalize_xtts_sonar")),
        ("Whisper + SONAR → SONAR speech", k(na, "finalize_xtts_whisper_sonar"), k(nl, "finalize_xtts_sonar")),
    ])
    L += section("ADMEDVoice (prawdziwa mowa): czysty → zakłócony", [
        ("kaskada Whisper + SONAR", k(aa, "finalize_whisper_sonar"), k(naa, "finalize_whisper_sonar")),
        ("kaskada Whisper + LaBSE", k(aa, "finalize_whisper_labse"), k(naa, "finalize_whisper_labse")),
        ("latentny APM-Whisper", k(al, "finalize_apm_whisper"), k(nal, "finalize_apm_whisper")),
        ("latentny SONAR speech", k(al, "finalize_sonar"), k(nal, "finalize_sonar")),
    ])
    L += vs_section("ADMEDVoice, sygnał zakłócony: kaskada → latentny", [
        ("Whisper + SONAR → SONAR speech (ta sama przestrzeń)", k(naa, "finalize_whisper_sonar"), k(nal, "finalize_sonar")),
        ("Whisper + LaBSE → APM-Whisper", k(naa, "finalize_whisper_labse"), k(nal, "finalize_apm_whisper")),
    ])
    return L


if __name__ == "__main__":
    d = sys.argv[1]
    conds = sys.argv[2:] or ["noisy"]
    out = ["# Odporność na zakłócenia — porównanie sparowane (re-run 2026-11-15)", "",
           "`noisy` = szum biały 15 dB + zmiana tempa; `speedonly` = sama zmiana tempa; `noise_snrN` = szum biały N dB.",
           "Te same pytania w obu warunkach (augmentacja z ziarnem per pytanie, 1 wariant). Bootstrap 10 000, McNemar dokładny.", ""]
    for c in conds:
        out += run(d, c)
    print("\n".join(out))
