#!/usr/bin/env bash
# Launch parallel vision extraction shards (2 pages/pair, resume).
set -euo pipefail
cd "$(dirname "$0")/.."
SHARDS="${1:-6}"
mkdir -p data/byzantine

# Stop any prior single-threaded extractor
pkill -f "extract_byzantine_training_data.py" 2>/dev/null || true
sleep 1

for i in $(seq 0 $((SHARDS - 1))); do
  PYTHONPATH=. .venv/bin/python scripts/extract_byzantine_training_data.py \
    --download --resume --max-pages 2 --fragments-per-page 2 \
    --shard "${i}/${SHARDS}" \
    >> "data/byzantine/extract_shard${i}.log" 2>&1 &
  echo "shard ${i}/${SHARDS} → pid $! (log: data/byzantine/extract_shard${i}.log)"
done

echo "Launched ${SHARDS} workers. Monitor: tail -f data/byzantine/extract_shard*.log"
