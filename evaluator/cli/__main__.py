"""Module entry point: ``python -m evaluator.cli --config <yaml>``.

The packaged equivalent of the repo-root ``evaluate.py`` wrapper — the webapi job
runner launches this (the repo script is not shipped in the wheel, so resolving it
file-relative breaks for pip-installed servers).
"""

import os
import sys

# Dump a Python+C traceback on native crashes (SIGSEGV) when debugging.
if os.environ.get("EVALUATOR_FAULTHANDLER") == "1":
    import faulthandler

    faulthandler.enable()


def main() -> int:
    try:
        from evaluator.cli import parse_args, run_evaluation

        args = parse_args()
        run_evaluation(args)
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 — CLI surface: report and exit non-zero
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
