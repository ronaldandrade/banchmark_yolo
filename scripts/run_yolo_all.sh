#!/usr/bin/env bash
# Roda a família YOLO (Ultralytics) no COCO val2017: FP32 e FP16. Continua se um modelo falhar.
cd "$(dirname "$0")/.."
MODELS="yolov8n yolov8s yolov8m yolov9t yolov9s yolov9m yolov10n yolov10s yolov10m yolo11n yolo11s yolo11m yolo12n yolo12s yolo12m"
mkdir -p results/logs
for m in $MODELS; do
  for half in "" "--half"; do
    echo "=== $m $half"
    .venv/bin/python -m src.benchmark.eval_ultralytics --weights $m.pt $half > results/logs/${m}${half// /}.log 2>&1 \
      && echo "ok $m $half" || echo "FALHOU $m $half (ver results/logs)"
  done
done
echo "FIM"
