#!/usr/bin/env bash
# Segmentação de instância no COCO val2017 (mask AP + box AP), FP32 e FP16. Continua se um falhar.
cd "$(dirname "$0")/.."
MODELS="yolov8n-seg yolov8s-seg yolov8m-seg yolo11n-seg yolo11s-seg yolo11m-seg rfdetr-seg-nano rfdetr-seg-small rfdetr-seg-medium rfdetr-seg-large maskrcnn_resnet50_fpn maskrcnn_resnet50_fpn_v2"
mkdir -p results/logs
for m in $MODELS; do
  for half in "" "--half"; do
    echo "=== $m $half"
    .venv/bin/python -m src.benchmark.eval_seg --model $m $half > results/logs/${m}${half// /}.log 2>&1 \
      && echo "ok $m $half" || echo "FALHOU $m $half (ver results/logs)"
  done
done
echo FIM
