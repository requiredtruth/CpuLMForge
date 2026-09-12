#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -gt 1 ]; then
    echo "Usage: ./demo.sh [memory-gib]" >&2
    exit 2
fi
MEMORY_GIB=${1:-4}
exec "$ROOT/cli.sh" "$ROOT/examples/samples.jsonl" --memory-gib "$MEMORY_GIB" --minimum-tps 10
