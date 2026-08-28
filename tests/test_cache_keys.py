"""M4: cache-key hashing is collision-safe for non-JSON-native args, and unchanged for the
JSON-native args every current caller passes (no cache invalidation)."""

import hashlib
import json
from enum import Enum

from evaluator.storage import cache_keys as ck


def test_json_native_keys_unchanged_vs_str_default():
    # the new typed default must NOT change keys for str/int/float/None/list/dict args
    for args in [("model", 5, None), ("a", ["x", "y"], {"k": 1}),
                 ("whisper", "openai/whisper-base", "cpu", 32)]:
        old = hashlib.md5(
            json.dumps(args, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert ck._compute_hash(*args) == old


def test_enum_and_its_value_string_do_not_collide():
    class Mode(Enum):
        A = "A"

    assert ck._compute_hash(Mode.A) != ck._compute_hash("A")
