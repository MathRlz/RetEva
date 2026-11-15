#!/usr/bin/env python3
"""English figures for the Speech Communication article.

Usage: python3 scripts/make_article_figures.py <rerun_dir> <thesis_semantics_png_dir> <out_dir>

fig2_cone.png  — 3x3 panel (encoder x {raw, ABTT, whitening}) assembled from the thesis cone-effect
                 histograms (source data not available): Polish title/axis labels cropped, English added.
fig4_wer.png   — ADMEDVoice: R@1 of the cascade and the latent paths per cascade-WER bin, 95% CIs.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_stats import BINS, boot_ci, newest, paired, scores  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CASCADE, LATENT, LATENT2 = "#2a78d6", "#eb6834", "#4a3aa7"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def fig2(sem_dir, out):
    rows = [("SeamlessM4T", "m4t"), ("Whisper", "whisper"), ("mHuBERT-147", "mhubert")]
    cols = [("raw", "_cone.png"), ("ABTT", "_cone_abtt.png"), ("whitening", ("_cone_wnitened.png", "_cone_whitened.png"))]
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.2))
    for i, (rname, rkey) in enumerate(rows):
        for j, (cname, suf) in enumerate(cols):
            ax = axes[i, j]
            sufs = suf if isinstance(suf, tuple) else (suf,)
            path = next(p for p in (Path(sem_dir) / f"{rkey}{x}" for x in sufs) if p.exists())
            im = Image.open(path).convert("RGB")
            w, h = im.size
            # crop the Polish title (top), y-label (left) and x-label (bottom); tick labels stay
            im = im.crop((int(0.017 * w), int(0.045 * h), w, int(0.965 * h)))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(cname, fontsize=12)
            if j == 0:
                ax.set_ylabel(rname, fontsize=12)
    fig.supxlabel("cosine similarity between mean-pooled utterance embeddings", fontsize=11)
    fig.supylabel("density", fontsize=11)
    fig.legend(handles=[plt.Line2D([], [], color="#1f77b4", lw=2, label="within English"),
                        plt.Line2D([], [], color="#ff7f0e", lw=2, label="within Polish"),
                        plt.Line2D([], [], color="#2ca02c", lw=1.2, ls="--", label="English–Polish")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}/fig2_cone.png")
    plt.close(fig)


def fig4(rerun, out):
    casc = newest(rerun, "admed_asr_embed_jina_only_cmp_admed_voice")
    lat = newest(rerun, "admed_audioemb_cmp_admed_voice")
    wer = scores(casc, "finalize_whisper_jina", "wer")
    series = [("Whisper + Jina v4 (cascade)", CASCADE, "s", None),
              ("APM-Whisper (latent)", LATENT, "^", (lat, "finalize_apm_whisper")),
              ("SONAR speech (latent)", LATENT2, "o", (lat, "finalize_sonar"))]
    labels = [b[2].replace("WER = 0", "0").replace("–", "–") for b in BINS]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(BINS))
    for k, (name, col, mk, src) in enumerate(series):
        other = scores(*src) if src else scores(casc, "finalize_whisper_jina")
        ids, hc, hl = paired(scores(casc, "finalize_whisper_jina"), other)
        h = hc if src is None else hl
        w = np.array([wer[i] for i in ids])
        ms, los, his, ns = [], [], [], []
        for lo, hi, _ in BINS:
            m = (w == 0) if hi == 0 else ((w > lo) & (w <= hi))
            v = h[m].astype(float)
            ms.append(v.mean()); lw, hw = boot_ci(v); los.append(lw); his.append(hw); ns.append(int(m.sum()))
        off = (k - 1) * 0.12
        ax.errorbar(x + off, ms, yerr=[np.array(ms) - los, np.array(his) - ms], fmt=mk + "-", color=col,
                    lw=1.8, ms=8, capsize=3, label=name, mfc=col, mec="white")
    ax.set_xticks(x, [f"{l}\n(n = {n})" for l, n in zip(labels, ns)])
    ax.set_xlabel("WER of the cascade (Whisper) on the same recording")
    ax.set_ylabel("Recall@1 (95% CI)")
    ax.set_ylim(0.6, 1.01)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(f"{out}/fig4_wer.png")
    plt.close(fig)


if __name__ == "__main__":
    rerun, sem, out = sys.argv[1:4]
    Path(out).mkdir(parents=True, exist_ok=True)
    fig2(sem, out)
    fig4(rerun, out)
    print("written:", sorted(p.name for p in Path(out).glob("fig*.png")))
