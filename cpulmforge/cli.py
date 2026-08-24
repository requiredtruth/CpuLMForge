from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from .profiles import Sample, select_profile

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a CPU local-model profile from measured JSONL runs.")
    parser.add_argument("samples")
    parser.add_argument("--memory-gib", type=float, required=True)
    parser.add_argument("--minimum-tps", type=float, default=0.0)
    parser.add_argument("--executable", default="llama-server")
    parser.add_argument("--fail-if-none", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = [json.loads(line) for line in Path(args.samples).read_text(encoding="utf-8").splitlines() if line.strip()]
        selection = select_profile([Sample(**row) for row in rows], memory_limit_bytes=int(args.memory_gib * 1024**3), minimum_tps=args.minimum_tps, executable=args.executable)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"cpulmforge: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(selection.to_dict(), indent=2, sort_keys=True))
    return 1 if args.fail_if_none and selection.selected is None else 0
