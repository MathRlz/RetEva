#!/usr/bin/env python3
"""Wykresy do prezentacji z przedziałami ufności (re-run 2026-11-15, wyniki per pytanie).

Użycie: python3 scripts/make_stats_charts.py <katalog_rerun> <katalog_100_en> <katalog_100_pl_plen> <katalog_wyjsciowy>

1. istotnosc.png   — forest plot różnic ΔR@1 dla par kluczowych (stats/primary.csv, text_vs_cascade.csv)
2. sciezki.png     — R@1 w każdej przestrzeni: tekst / kaskada / latentny (xtts, 95% CI)
3. syntezatory.png — wpływ głosu TTS na R@1: wiersz = ASR × embedder, punkt = głos
4. jezyki.png      — EN/PL/PL→EN, 100 pytań, słupki z 95% CI ("orientacyjnie")
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta zwalidowana (dataviz validate_palette.js, tło #ffffff): ścieżka zapytania = kolor.
SCIEZKA = {"tekst": ("#008300", "o"), "kaskada": ("#2a78d6", "s"), "latentny": ("#eb6834", "^")}
GLOS = {"xtts": ("#1baf7a", "o", "XTTS"), "m4t": ("#4a3aa7", "s", "SeamlessM4T"), "piper": ("#eda100", "D", "Piper")}
INK, INK2, SIATKA = "#0b0b0b", "#52514e", "#e6e5e0"
N_BOOT, SEED = 10_000, 0

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def ci(values):
    """Percentylowy bootstrap średniej (10 000 prób), ten sam co w scripts/paired_stats.py."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_hits(p, n):
    """CI dla R@1 z samej liczby trafień — dla zmiennej 0/1 bootstrap zależy tylko od (trafienia, n)."""
    k = round(p * n)
    return ci([1.0] * k + [0.0] * (n - k))


def newest(d, prefix):
    files = sorted(glob.glob(f"{d}/results_{prefix}_*.json"))
    if not files:
        raise SystemExit(f"brak wyników: {d}/results_{prefix}_*")
    return json.load(open(files[-1]))["variants"]


def r1(variants, key):
    """(średnia, CI) Recall@1 z wyników per pytanie."""
    s = variants[key]["report"]["item_scores"]
    vals = list(next(iter(s.values()))["recall@1"].values())
    return float(np.mean(vals)), ci(vals)


def podpisz(ax, x, y, tekst, **kw):
    ax.annotate(tekst, (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=10, color=INK, **kw)


# ---------- 1. forest plot ----------
def istotnosc(rerun, out):
    rows = list(csv.DictReader(open(f"{rerun}/stats/primary.csv")))
    eq = list(csv.DictReader(open(f"{rerun}/stats/text_vs_cascade.csv")))[0]
    nazwy = {
        "P1": "Whisper+Jina  →  SONAR speech\n(PubMedQA)",
        "P2": "Whisper+SONAR  →  SONAR speech\n(PubMedQA)",
        "P3": "APM  →  Whisper+Jina\n(ADMEDVoice)",
    }
    pozycje = [(nazwy[r["pair"][:2]], r, True) for r in rows] + [("tekst Jina  →  Whisper+Jina\n(PubMedQA)", eq, False)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ys = [4, 3, 2, 0.6]
    for y, (nazwa, r, test) in zip(ys, pozycje):
        d = float(r["delta_recall@1"])
        lo, hi = map(float, r["delta_recall@1_ci"].split(";"))
        ax.plot([lo, hi], [y, y], color=INK, lw=2, solid_capstyle="round")
        ax.plot(d, y, "o", color=INK, ms=9, mec="white", mew=2)
        p = float(r["p_holm"])
        ptxt = ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") if test else "brak testu"
        ax.text(1.02, y, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]    {ptxt}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=11, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.axhline(1.3, color=SIATKA, lw=1)
    ax.set_yticks(ys, [p[0] for p in pozycje])
    ax.set_xlim(-0.55, 0.1)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("różnica Recall@1 (B − A), 95% przedział ufności")
    ax.grid(axis="x", color=SIATKA, lw=0.8)
    ax.text(1.02, 4.45, "Δ [95% CI]               p (Holm)", transform=ax.get_yaxis_transform(),
            fontsize=10, color=INK2, va="bottom")
    ax.tick_params(axis="y", length=0)
    fig.savefig(f"{out}/istotnosc.png")
    plt.close(fig)


# ---------- 2. ścieżki w przestrzeniach ----------
def sciezki(rerun, out):
    txt = newest(rerun, "pubmed_qa_text_embed_baseline_pubmed_qa")
    asr = newest(rerun, "pubmed_qa_tts_asr_embed_cmp_pubmed_qa")
    lat = newest(rerun, "pubmed_qa_tts_audioemb_cmp_pubmed_qa")
    przestrzenie = [  # (nazwa, tekst, kaskada, latentny)
        ("Jina v4", (txt, "finalize_jina"), (asr, "finalize_xtts_whisper_jina"), (lat, "finalize_xtts_apm_whisper")),
        ("LaBSE", (txt, "finalize_labse"), (asr, "finalize_xtts_whisper_labse"), None),
        ("SONAR", (txt, "finalize_sonar"), (asr, "finalize_xtts_whisper_sonar"), (lat, "finalize_xtts_sonar")),
        ("EAM", (txt, "finalize_eam"), None, (lat, "finalize_xtts_eam")),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    off = {"tekst": 0.22, "kaskada": 0.0, "latentny": -0.22}
    for i, (nazwa, *zrodla) in enumerate(przestrzenie):
        y = len(przestrzenie) - 1 - i
        ax.axhline(y, color=SIATKA, lw=0.8, zorder=0)
        for sc, zr in zip(("tekst", "kaskada", "latentny"), zrodla):
            if zr is None:
                continue
            m, (lo, hi) = r1(*zr)
            kol, mk = SCIEZKA[sc]
            yy = y + off[sc]
            ax.plot([lo, hi], [yy, yy], color=kol, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(m, yy, mk, color=kol, ms=10, mec="white", mew=2, zorder=3)
            lewo = hi > 0.9  # przy prawej krawędzi etykieta idzie za lewy koniec wąsa
            etykieta = f"{m:.2f}" + (" (APM)" if nazwa == "Jina v4" and sc == "latentny" else "")
            ax.annotate(etykieta, (lo if lewo else hi, yy), xytext=(-6 if lewo else 6, 0), textcoords="offset points",
                        ha="right" if lewo else "left", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(przestrzenie)), [p[0] for p in reversed(przestrzenie)])
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.6, len(przestrzenie) - 0.4)
    ax.set_xlabel("Recall@1 (1000 pytań PubMedQA, głos XTTS, 95% CI)")
    ax.grid(axis="x", color=SIATKA, lw=0.8)
    ax.tick_params(axis="y", length=0)
    for sc, (kol, mk) in SCIEZKA.items():
        ax.plot([], [], mk, color=kol, ms=9, label={"tekst": "tekst (zapytanie pisane)",
                "kaskada": "kaskada (Whisper → tekst)", "latentny": "latentny (mowa → wektor)"}[sc])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=11)
    fig.savefig(f"{out}/sciezki.png")
    plt.close(fig)


# ---------- 3. wpływ syntezatora ----------
def syntezatory(rerun, out):
    asr = newest(rerun, "pubmed_qa_tts_asr_embed_cmp_pubmed_qa")
    wiersze = [(emb, a) for emb in ("jina", "labse", "sonar") for a in ("whisper", "m4t", "wav2vec2")]
    asr_nazwa = {"whisper": "Whisper", "m4t": "SeamlessM4T", "wav2vec2": "wav2vec2"}
    emb_nazwa = {"jina": "Jina v4", "labse": "LaBSE", "sonar": "SONAR"}
    fig, ax = plt.subplots(figsize=(11, 6.2))
    off = {"xtts": 0.2, "m4t": 0.0, "piper": -0.2}
    etyk = []
    for i, (emb, a) in enumerate(wiersze):
        y = len(wiersze) - 1 - i
        ax.axhline(y, color=SIATKA, lw=0.8, zorder=0)
        ms = {}
        for g, (kol, mk, _) in GLOS.items():
            m, (lo, hi) = r1(asr, f"finalize_{g}_{a}_{emb}")
            ms[g] = m
            ax.plot([lo, hi], [y + off[g]] * 2, color=kol, lw=1.5, alpha=0.6, zorder=2)
            ax.plot(m, y + off[g], mk, color=kol, ms=8, mec="white", mew=1.5, zorder=3)
        rozrzut = max(ms.values()) - min(ms.values())
        ax.text(1.02, y, f"rozrzut {rozrzut:.2f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=10, color=INK if rozrzut > 0.06 else INK2, weight="bold" if rozrzut > 0.06 else "normal")
        etyk.append(f"{emb_nazwa[emb]} · {asr_nazwa[a]}")
    m4t = r1(asr, "finalize_m4t_m4t_jina")[0]
    ax.annotate("SeamlessM4T słyszy własną\nmowę syntetyczną", (m4t, len(wiersze) - 2), xytext=(0.62, len(wiersze) - 1.6),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_yticks(range(len(wiersze)), list(reversed(etyk)))
    for y in (2.5, 5.5):
        ax.axhline(y, color=INK2, lw=0.6)
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(-0.6, len(wiersze) - 0.4)
    ax.set_xlabel("Recall@1 kaskady (1000 pytań PubMedQA, 95% CI)")
    ax.grid(axis="x", color=SIATKA, lw=0.8)
    ax.tick_params(axis="y", length=0)
    for g, (kol, mk, nazwa) in GLOS.items():
        ax.plot([], [], mk, color=kol, ms=8, label=f"syntezator {nazwa}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=11)
    fig.savefig(f"{out}/syntezatory.png")
    plt.close(fig)


# ---------- 4. warianty językowe, 100 pytań ----------
def jezyki(en_dir, pl_dir, out, n=100):
    zr = {
        "EN→EN": {"text": newest(en_dir, "pubmed_qa_text_embed_baseline_pubmed_qa"),
                  "asr": newest(en_dir, "pubmed_qa_tts_asr_embed_cmp_pubmed_qa"),
                  "audio": newest(en_dir, "pubmed_qa_tts_audioemb_cmp_pubmed_qa")},
        "PL→PL": {"text": newest(pl_dir, "pubmed_qa_text_embed_baseline_pl_pubmed_qa"),
                  "asr": newest(pl_dir, "pubmed_qa_tts_asr_embed_cmp_pl_pubmed_qa"),
                  "audio": newest(pl_dir, "pubmed_qa_tts_audioemb_cmp_pl_pubmed_qa")},
        "PL→EN": {"text": newest(pl_dir, "pubmed_qa_text_embed_baseline_plen_pubmed_qa"),
                  "asr": newest(pl_dir, "pubmed_qa_tts_asr_embed_cmp_plen_pubmed_qa"),
                  "audio": newest(pl_dir, "pubmed_qa_tts_audioemb_cmp_plen_pubmed_qa")},
    }
    # tekst SONAR = morski (aqua) z palety: pełny słupek, zostaje w rodzinie „zielony = tekst”;
    # przechodzi walidację CVD (jaśniejszy zielony nie przechodził vs pomarańcz), kontrast <3:1 → wartości nad słupkami
    serie = [("tekst, Jina", "text", "finalize_jina", "#008300", None),
             ("kaskada Whisper + Jina", "asr", "finalize_xtts_whisper_jina", "#2a78d6", None),
             ("tekst, SONAR", "text", "finalize_sonar", "#1baf7a", None),
             ("latentny SONAR", "audio", "finalize_xtts_sonar", "#eb6834", None)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.2
    for j, (nazwa, tryb, klucz, kol, wzor) in enumerate(serie):
        for i, wariant in enumerate(zr):
            m = zr[wariant][tryb][klucz]["Recall@1"]
            lo, hi = ci_hits(m, n)
            x = i + (j - 1.5) * w
            ax.bar(x, m, w - 0.02, color="white" if wzor else kol, edgecolor=kol, hatch=wzor, lw=1.5,
                   label=nazwa if i == 0 else None, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.4, zorder=3)
            ax.text(x, hi + 0.015, f"{m:.2f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(3), list(zr))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall@1 (95% CI)")
    ax.grid(axis="y", color=SIATKA, lw=0.8, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=11)
    ax.text(1.0, -0.13, "n = 100 pytań — wyniki orientacyjne (CI ±8–10 pp.)", transform=ax.transAxes,
            ha="right", fontsize=10, color=INK2, style="italic")
    fig.savefig(f"{out}/jezyki.png")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    rerun, en100, pl100, out = sys.argv[1:]
    Path(out).mkdir(parents=True, exist_ok=True)
    istotnosc(rerun, out)
    sciezki(rerun, out)
    syntezatory(rerun, out)
    jezyki(en100, pl100, out)
    print("zapisano:", sorted(p.name for p in Path(out).glob("*.png")))
