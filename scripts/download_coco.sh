#!/usr/bin/env bash
# Baixa COCO 2017 val (imagens + anotações) para data/coco/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/coco && cd data/coco
[ -f val2017.zip ] || curl -L -C - -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
[ -f annotations_trainval2017.zip ] || curl -L -C - -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
[ -d val2017 ] || unzip -q val2017.zip
[ -f annotations/instances_val2017.json ] || unzip -q -o annotations_trainval2017.zip 'annotations/instances_val2017.json'
echo "OK: $(ls val2017 | wc -l) imagens"
