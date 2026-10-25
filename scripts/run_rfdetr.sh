#!/usr/bin/env bash
cd "$(dirname "$0")/.."
mkdir -p results/logs
for v in nano small medium base; do
  for half in "" "--half"; do
    echo "=== rfdetr-$v $half"
    .venv/bin/python -m src.benchmark.eval_rfdetr --variant $v $half > results/logs/rfdetr-${v}${half// /}.log 2>&1 \
      && echo "ok rfdetr-$v $half" || echo "FALHOU rfdetr-$v $half (ver results/logs)"
  done
done
echo FIM
