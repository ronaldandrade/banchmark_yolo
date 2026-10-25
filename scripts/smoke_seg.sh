#!/usr/bin/env bash
# Espera os clássicos terminarem (GPU livre) e testa cada backend de segmentação com 20 imagens.
cd "$(dirname "$0")/.."
until grep -q "^FIM" "$1"; do sleep 15; done
for m in yolo11n-seg rfdetr-seg-nano maskrcnn_resnet50_fpn; do
  echo "## $m"
  .venv/bin/python -m src.benchmark.eval_seg --model $m --limit 20 --warmup 3 --out "$2" 2>&1 \
    | grep -E '"mask_mAP50-95"|"box_mAP50-95"|"median"|peak_gpu|Traceback|Error' | head -8
done
echo SMOKE_FIM
