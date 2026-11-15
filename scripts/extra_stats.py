#!/usr/bin/env python3
"""Dodatkowe analizy na wynikach per pytanie (re-run 2026-11-15). Pisze Markdown na stdout.

Użycie: python3 scripts/extra_stats.py <katalog_rerun> > <katalog_rerun>/stats/extra.md

1. WER a opłacalność ścieżki latentnej (RQ3): przedziały WER kaskady, w każdym R@1 kaskady
   i ścieżki latentnej na tych samych pytaniach + trend (regresja różnicy trafień na WER).
2. Bootstrap po mówcach (ADMEDVoice): losujemy mówców, nie pytania — nagrania jednego
   mówcy są skorelowane, więc bootstrap po pytaniach daje za wąskie przedziały.
3a. TOST: formalna równoważność tekst vs kaskada (margines ±2 pp.).
3b. Cochran Q: czy głos TTS wpływa na trafienia R@1 (osobno w każdym wierszu ASR × embedder).
"""
import glob
import json
import sys

import numpy as np
from scipy import stats

N_BOOT, SEED, MARGIN = 10_000, 0, 0.02


def newest(d, prefix):
    return json.load(open(sorted(glob.glob(f"{d}/results_{prefix}_*.json"))[-1]))["variants"]


def scores(variants, key, metric="recall@1"):
    s = variants[key]["report"]["item_scores"]
    return next(iter(s.values()))[metric]


def paired(a, b):
    ids = sorted(set(a) & set(b))
    return ids, np.array([a[i] for i in ids], float), np.array([b[i] for i in ids], float)


def boot_ci(x, level=0.95):
    idx = np.random.default_rng(SEED).integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2 * 100
    return np.percentile(m, a), np.percentile(m, 100 - a)


def mcnemar(ha, hb):
    b, c = int(np.sum(ha & ~hb)), int(np.sum(~ha & hb))
    return b, c, (1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"


# ---------- 1. WER ----------
BINS = [(0.0, 0.0, "WER = 0"), (0.0, 0.1, "0–10%"), (0.1, 0.2, "10–20%"), (0.2, 0.4, "20–40%"), (0.4, 99, "> 40%")]


def wer_section(title, casc, lat):
    """casc/lat: (variants, key). WER bierzemy z kaskady (ścieżka latentna nie ma transkrypcji)."""
    wer = scores(*casc, "wer")
    ids, hc, hl = paired(scores(*casc), scores(*lat))
    w = np.array([wer[i] for i in ids])
    hc, hl = hc > 0, hl > 0
    out = [f"### {title}", "",
           "| WER kaskady | n | R@1 kaskada | R@1 latentny | Δ latentny − kaskada [95% CI] | b / c | McNemar p |",
           "|---|---|---|---|---|---|---|"]
    for lo, hi, name in BINS:
        m = (w == 0) if hi == 0 else ((w > lo) & (w <= hi))
        if m.sum() < 5:
            out.append(f"| {name} | {int(m.sum())} | — | — | za mało pytań | — | — |")
            continue
        d = hl[m].astype(float) - hc[m].astype(float)
        b, c, p = mcnemar(hc[m], hl[m])
        out.append(f"| {name} | {int(m.sum())} | {hc[m].mean():.3f} | {hl[m].mean():.3f} | "
                   f"{d.mean():+.3f} {fci(*boot_ci(d))} | {b} / {c} | {fp(p)} |")
    d = hl.astype(float) - hc.astype(float)
    lr = stats.linregress(w, d)
    out += ["", f"Trend: regresja (trafienie latentny − trafienie kaskada) na WER: nachylenie {lr.slope:+.3f} "
                f"na jednostkę WER (p = {fp(lr.pvalue)}), n = {len(w)}. "
                "Dodatnie nachylenie = im gorsza transkrypcja, tym bardziej opłaca się ścieżka latentna.", ""]
    return out


# ---------- 2. bootstrap po mówcach ----------
def speaker(qid):
    p = qid.split("_")
    return f"{p[2]}-{p[3]}"  # placówka-mówca, np. GU001-AMV011


def cluster_section(rows):
    out = ["| Para (A → B) | n pytań | n mówców | Δ R@1 | CI po pytaniach | CI po mówcach | p (bootstrap po mówcach) |",
           "|---|---|---|---|---|---|---|"]
    for name, a, b in rows:
        ids, ha, hb = paired(scores(*a), scores(*b))
        d = hb - ha
        spk = np.array([speaker(i) for i in ids])
        uniq = np.unique(spk)
        sums = np.array([d[spk == s].sum() for s in uniq])
        cnts = np.array([(spk == s).sum() for s in uniq])
        idx = np.random.default_rng(SEED).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
        means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)  # ratio estimator: pytania ważone równo
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
        out.append(f"| {name} | {len(ids)} | {len(uniq)} | {d.mean():+.3f} | {fci(*boot_ci(d))} | "
                   f"{fci(lo, hi)} | {fp(p)} |")
    return out


# ---------- 3a. TOST ----------
def tost_section(rows):
    out = [f"Margines równoważności ±{MARGIN:.2f} (±{MARGIN*100:.0f} pp. R@1). Równoważne, gdy oba testy jednostronne "
           "odrzucają (p_TOST = większe z dwóch p < 0.05) — tożsame z 90% CI różnicy w całości wewnątrz marginesu.", "",
           "| Para (A → B) | n | Δ R@1 | 90% CI | p_TOST | Równoważne? |", "|---|---|---|---|---|---|"]
    for name, a, b in rows:
        _, ha, hb = paired(scores(*a), scores(*b))
        d = hb - ha
        p_lo = stats.ttest_1samp(d, -MARGIN, alternative="greater").pvalue
        p_hi = stats.ttest_1samp(d, MARGIN, alternative="less").pvalue
        p = max(p_lo, p_hi)
        lo, hi = boot_ci(d, 0.90)
        out.append(f"| {name} | {len(d)} | {d.mean():+.3f} | {fci(lo, hi)} | {fp(p)} | {'tak' if p < 0.05 else 'nie'} |")
    return out


# ---------- 3b. Cochran Q ----------
def cochran_q(hits):
    """hits: (N pytań, k warunków) 0/1. Zwraca (Q, p) — chi² z k−1 stopniami swobody."""
    k = hits.shape[1]
    col, row = hits.sum(axis=0), hits.sum(axis=1)
    den = k * row.sum() - (row ** 2).sum()
    q = (k - 1) * (k * (col ** 2).sum() - row.sum() ** 2) / den if den else 0.0
    return q, stats.chi2.sf(q, k - 1)


def cochran_section(asr):
    rows, ps = [], []
    for emb in ("jina", "labse", "sonar"):
        for a in ("whisper", "m4t", "wav2vec2"):
            per = [scores(asr, f"finalize_{g}_{a}_{emb}") for g in ("xtts", "m4t", "piper")]
            ids = sorted(set(per[0]) & set(per[1]) & set(per[2]))
            h = np.array([[p[i] > 0 for p in per] for i in ids], float)
            q, p = cochran_q(h)
            rows.append((emb, a, len(ids), h.mean(axis=0), q))
            ps.append(p)
    adj = holm(np.array(ps))
    out = ["| Embedder · ASR | n | R@1 XTTS / M4T / Piper | Q (df=2) | p (Holm, m=9) |", "|---|---|---|---|---|"]
    for (emb, a, n, m, q), pa in zip(rows, adj):
        out.append(f"| {emb} · {a} | {n} | {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f} | {q:.1f} | {fp(pa)} |")
    return out


def main(d):
    asr = newest(d, "pubmed_qa_tts_asr_embed_cmp_pubmed_qa")
    lat = newest(d, "pubmed_qa_tts_audioemb_cmp_pubmed_qa")
    txt = newest(d, "pubmed_qa_text_embed_baseline_pubmed_qa")
    a_asr = newest(d, "admed_asr_embed_jina_only_cmp_admed_voice")
    a_lat = newest(d, "admed_audioemb_cmp_admed_voice")
    L = ["# Dodatkowe analizy statystyczne (re-run 2026-11-15)", "",
         f"Bootstrap {N_BOOT} prób, ziarno {SEED}. Trafienie = R@1 > 0. Generator: `scripts/extra_stats.py`.", "",
         "## 1. Przy jakim WER opłaca się pominąć transkrypcję (RQ3)", "",
         "Pytania dzielone według WER kaskady (Whisper). W każdym przedziale porównanie na tych samych pytaniach.", ""]
    L += wer_section("ADMEDVoice (prawdziwa mowa): Whisper+Jina vs APM (przestrzeń Jina)",
                     (a_asr, "finalize_whisper_jina"), (a_lat, "finalize_apm_whisper"))
    L += wer_section("ADMEDVoice: Whisper+Jina vs SONAR mowa",
                     (a_asr, "finalize_whisper_jina"), (a_lat, "finalize_sonar"))
    L += wer_section("PubMedQA 1000, XTTS: Whisper+SONAR vs SONAR mowa (ta sama przestrzeń)",
                     (asr, "finalize_xtts_whisper_sonar"), (lat, "finalize_xtts_sonar"))
    L += ["## 2. ADMEDVoice: bootstrap po mówcach", "",
          "Losujemy mówców ze zwracaniem (z wszystkimi ich pytaniami). Szerszy przedział = korelacja w obrębie mówcy.", ""]
    L += cluster_section([
        ("APM → Whisper+Jina (para 3)", (a_lat, "finalize_apm_whisper"), (a_asr, "finalize_whisper_jina")),
        ("SONAR mowa → Whisper+Jina", (a_lat, "finalize_sonar"), (a_asr, "finalize_whisper_jina")),
    ])
    L += ["", "## 3a. Równoważność tekst → kaskada (TOST)", ""]
    L += tost_section([
        ("tekst Jina → Whisper+Jina (XTTS)", (txt, "finalize_jina"), (asr, "finalize_xtts_whisper_jina")),
        ("tekst LaBSE → Whisper+LaBSE (XTTS)", (txt, "finalize_labse"), (asr, "finalize_xtts_whisper_labse")),
        ("tekst SONAR → Whisper+SONAR (XTTS)", (txt, "finalize_sonar"), (asr, "finalize_xtts_whisper_sonar")),
    ])
    L += ["", "## 3b. Wpływ głosu TTS (test Q Cochrana)", "",
          "Czy trafienia R@1 tej samej kaskady różnią się między trzema głosami (te same 1000 pytań).", ""]
    L += cochran_section(asr)
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
