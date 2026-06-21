"""Stage-handler subpackage: one module per DAG stage family.

Each submodule registers its handlers via ``@register_stage_handler`` at import time, so the
executor discovers them by name. Importing this package imports every submodule, which
**fires all registrations** — so any code path that needs the handlers available only has
to ``import evaluator.evaluation.handlers`` (or anything that does, e.g. ``executor.run``).
"""

from . import sinks  # noqa: F401
from . import source  # noqa: F401
from . import audio  # noqa: F401
from . import asr  # noqa: F401
from . import embedding  # noqa: F401
from . import retrieval  # noqa: F401
from . import query  # noqa: F401
from . import rag  # noqa: F401
from . import metrics  # noqa: F401
