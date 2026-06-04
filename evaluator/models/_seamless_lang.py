"""Shared SeamlessM4T language-code aliases.

SeamlessM4T uses ISO 639-3 (3-letter) codes. This maps the common 2-letter
ISO 639-1 codes onto them so callers can pass either. Used by both the ASR model
(``models/asr/seamless_m4t.py``) and the TTS provider (``models/tts/m4t_tts.py``).
"""
from typing import Dict

# ISO 639-1 -> SeamlessM4T 3-letter codes (common subset).
SEAMLESS_LANG_ALIASES: Dict[str, str] = {
    "en": "eng", "pl": "pol", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus", "uk": "ukr",
    "cs": "ces", "zh": "cmn", "ja": "jpn", "ko": "kor", "ar": "arb",
}
