"""HTTP API entrypoint for evaluator runtime.

Requires the ``webapi`` extra: ``pip install evaluator[webapi]``
"""

try:
    from .app import create_app
except ImportError:  # fastapi not installed
    def create_app(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "WebAPI dependencies not installed. "
            "Install with: pip install evaluator[webapi]"
        )

__all__ = ["create_app"]
