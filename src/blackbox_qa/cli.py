"""Command-line front door: parse a question, run the agent, print the answer."""

from __future__ import annotations

import sys

from blackbox_qa import agent


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: blackbox-qa "your question here"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    result = agent.run(question)
    print(result.answer)
    if result.citations:
        print("\nsources: " + ", ".join(result.citations))
    print(f"\n[turns={result.turns} confidence={result.confidence}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
