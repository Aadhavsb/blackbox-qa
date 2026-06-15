"""Command-line entry point. Wired up in phase 3 (agent loop)."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: blackbox-qa "your question here"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    print(f"[not yet implemented] would answer: {question!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
