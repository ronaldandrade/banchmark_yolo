#!/usr/bin/env bash
# Detectores clássicos (torchvision + DETR) no COCO val2017, FP32 e FP16 (autocast). Continua se um falhar.
cd "$(dirname "$0")/.."
MODELS="ssdlite320_mobilenet_v3_large ssd300_vgg16 detr_resnet50 fasterrcnn_resnet50_fpn retinanet_resnet50_fpn fcos_resnet50_fpn fasterrcnn_resnet50_fpn_v2 retinanet_resnet50_fpn_v2"
mkdir -p results/logs
for m in $MODELS; do
  for half in "" "--half"; do
    echo "=== $m $half"
    .venv/bin/python -m src.benchmark.eval_classic --model $m $half > results/logs/${m}${half// /}.log 2>&1 \
      && echo "ok $m $half" || echo "FALHOU $m $half (ver results/logs)"
  done
done
echo FIM
