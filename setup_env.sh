#!/usr/bin/env bash
# Cria o ambiente isolado do projeto em ./.venv (nada é instalado globalmente).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
PY=.venv/bin/python   # usar "python -m pip": o caminho do projeto contém espaços

$PY -m pip install --upgrade pip
# Driver 535 (CUDA 12.2): wheels cu124 funcionam via compatibilidade de versão menor.
# torch>=2.6 é exigido por transformers 5.x (usado pelo rfdetr)
$PY -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install -r requirements.txt
$PY -m ipykernel install --user --name banchmark_yolo --display-name "banchmark_yolo"

$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
