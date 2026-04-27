#!/usr/bin/env python3
"""Main evaluation script entry point.

This module serves as the CLI entry point for the evaluation tool.
All logic is delegated to the evaluator.cli package.
"""

import sys


def main() -> int:
    """Main entry point for the evaluation CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    try:
        from evaluator.cli import parse_args, run_evaluation
        
        args = parse_args()
        run_evaluation(args)
        return 0
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
