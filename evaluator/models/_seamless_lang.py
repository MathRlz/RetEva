"""Shared SeamlessM4T language-code aliases.

SeamlessM4T uses ISO 639-3 codes; this maps common ISO 639-1 codes onto them
so callers can pass either.
"""
from typing import Dict

SEAMLESS_LANG_ALIASES: Dict[str, str] = {
    "en": "eng", "pl": "pol", "de": "deu", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus", "uk": "ukr",
    "cs": "ces", "zh": "cmn", "ja": "jpn", "ko": "kor", "ar": "arb",
}
