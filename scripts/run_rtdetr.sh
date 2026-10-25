#!/usr/bin/env bash
cd "$(dirname "$0")/.."
mkdir -p results/logs
for half in "" "--half"; do
  echo "=== rtdetr-l $half"
  .venv/bin/python -m src.benchmark.eval_ultralytics --weights rtdetr-l.pt $half > results/logs/rtdetr-l${half// /}.log 2>&1 \
    && echo "ok rtdetr-l $half" || echo "FALHOU rtdetr-l $half"
done
echo FIM
