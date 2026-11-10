# Benchmark de Modelos de Detecção de Objetos (COCO val2017)

Benchmark reprodutível que compara acurácia (mAP) e eficiência computacional (latência ponta a ponta, pico de VRAM e parâmetros) de 28 modelos de detecção de objetos em hardware de consumidor (GPU de 4 GB).

## Como rodar

Clone o repositório e configure o ambiente:

```bash
git clone https://github.com/ronaldandrade/banchmark_yolo.git
cd banchmark_yolo
bash setup_env.sh
```

Baixe o conjunto de validação COCO 2017 (imagens e anotações):

```bash
bash scripts/download_coco.sh
```

Para executar o benchmark completo por família (FP32 e FP16):

```bash
bash scripts/run_yolo_all.sh   # YOLOv8, v9, v10, 11 e 12 (portes n/t, s, m)
bash scripts/run_rtdetr.sh     # RT-DETR-L
bash scripts/run_rfdetr.sh     # RF-DETR (nano, small, medium, base)
bash scripts/run_classic.sh    # Faster R-CNN, RetinaNet, FCOS, SSD300, SSDLite320, DETR-R50
```

Para avaliar um modelo individualmente:

```bash
# YOLO / RT-DETR (Ultralytics)
.venv/bin/python -m src.benchmark.eval_ultralytics --weights yolo11n.pt
.venv/bin/python -m src.benchmark.eval_ultralytics --weights yolo11n.pt --half

# RF-DETR
.venv/bin/python -m src.benchmark.eval_rfdetr --variant nano
.venv/bin/python -m src.benchmark.eval_rfdetr --variant nano --half

# Detectores clássicos (torchvision / HuggingFace)
.venv/bin/python -m src.benchmark.eval_classic --model fcos_resnet50_fpn
.venv/bin/python -m src.benchmark.eval_classic --model fcos_resnet50_fpn --half
```

Para consolidar as métricas em tabelas e gerar as curvas da fronteira de Pareto:

```bash
.venv/bin/python -m src.benchmark.make_table
.venv/bin/python -m src.benchmark.plot_pareto
```

## Estrutura

```
banchmark_yolo/
├── PROJETO.md               # Especificação metodológica completa e notas experimentais.
├── setup_env.sh             # Criação do ambiente isolado (.venv) com PyTorch 2.6.0 e CUDA 12.4.
├── requirements.txt         # Pacotes Python necessários para execução e gráficos.
├── requirements.lock.txt    # Versões congeladas das dependências utilizadas nas medições.
├── data/                    # Imagens e anotações do COCO 2017 val (ignorado no Git).
├── scripts/
│   ├── download_coco.sh     # Download e extração do val2017.zip e anotações de instâncias.
│   ├── run_yolo_all.sh      # Bateria de testes dos modelos YOLOv8/v9/v10/11/12 em FP32 e FP16.
│   ├── run_rtdetr.sh        # Bateria de testes do RT-DETR-L em FP32 e FP16.
│   ├── run_rfdetr.sh        # Bateria de testes do RF-DETR (nano, small, medium, base).
│   ├── run_classic.sh       # Bateria de testes para modelos torchvision e DETR do HuggingFace.
│   ├── run_seg.sh           # Roteiro de segmentação de instância (não executado).
│   ├── smoke_seg.sh         # Teste rápido de fumaça para o módulo de segmentação.
│   ├── diag_fcos_fp16.py    # Diagnóstico da queda de mAP do FCOS com autocast FP16.
│   └── diag_fcos_decode.py  # Análise de discrepâncias na decodificação de caixas do FCOS.
├── src/benchmark/
│   ├── common.py            # Utilitários de parsing de nomes de modelos e consolidação dos JSONs.
│   ├── eval_ultralytics.py  # Avaliador unificado para YOLO e RT-DETR (Ultralytics).
│   ├── eval_rfdetr.py       # Avaliador para modelos RF-DETR (backbone DINOv2).
│   ├── eval_classic.py      # Avaliador para torchvision (Faster R-CNN, RetinaNet, FCOS, SSD) e DETR.
│   ├── eval_seg.py          # Avaliador de segmentação de instância (rascunho pendente de validação).
│   ├── make_table.py        # Compilador dos JSONs em results/tabela_benchmark.csv e .md.
│   └── plot_pareto.py       # Gerador dos gráficos de fronteira de Pareto em paper/figures/.
├── results/                 # Saída dos benchmarks: 56 arquivos JSON (FP32/FP16), logs e tabelas.
└── paper/                   # Rascunho do artigo científico (artigo.md) e figuras geradas.
```

## Como funciona

O benchmark unifica a avaliação de 28 arquiteturas de detecção sobre as 5.000 imagens do conjunto de validação do COCO 2017:

1. **Avaliador COCO unificado:** Cada script de avaliação carrega pesos pré-treinados oficiais e processa imagens individualmente (`batch=1`). As caixas delimitadoras e pontuações de confiança são convertidas para o formato COCO e enviadas para o `pycocotools.cocoeval.COCOeval`, padronizando o cálculo de mAP@[.5:.95], mAP@.5 e métricas por escala (AP_S, AP_M, AP_L).
2. **Medição de latência:** Cada medição descarta as primeiras iterações para aquecimento da GPU (warm-up de 30 a 50 imagens) e utiliza `torch.cuda.synchronize()` antes e após cada leitura de tempo. A latência reportada como total corresponde à mediana por imagem de todo o ciclo: leitura do arquivo em disco, pré-processamento, inferência do modelo e pós-processamento/NMS.
3. **Consumo de memória:** A VRAM registrada corresponde ao pico de memória explicitamente alocado pelo PyTorch durante a inferência (`torch.cuda.max_memory_allocated()`), desconsiderando o contexto inicial do driver CUDA.
4. **Tratamento de precisão numérica (FP16):**
   - Nos modelos Ultralytics e RF-DETR, a rede é convertida diretamente para meia precisão via `.half()`.
   - Nos modelos torchvision e DETR, a inferência roda com precisão mista via `torch.autocast`.
   - No FCOS, as caixas são convertidas para FP32 antes de entrar no `torchvision.ops.batched_nms` para evitar o estouro de limite numérico (`inf`) provocado pelo deslocamento de coordenadas por classe em FP16.
5. **Fronteira de Pareto:** O script `plot_pareto.py` lê os resultados de todos os JSONs em `results/`, ordena os modelos por tempo total e identifica os pontos não-dominados (onde nenhum modelo com menor latência obtém mAP superior), gerando gráficos comparativos para FP32 e FP16.

## Limitações

- **Hardware e memória:** Executado exclusivamente em GPU de consumidor (NVIDIA GeForce RTX 3050 Laptop, 4 GB de VRAM). Modelos de grande porte (como YOLOv8/9/10/11/12 nos tamanhos `l` e `x`, bem como RT-DETR-X) não foram avaliados por estouro de memória (OOM).
- **Segmentação não executada:** O código para segmentação de instância (`eval_seg.py` e `run_seg.sh`) foi estruturado, mas não foi testado nem executado. Segmentação semântica não foi implementada.
- **Resoluções heterogêneas:** Modelos Ultralytics rodam em 640×640 px, enquanto RF-DETR opera em suas resoluções nativas (384 a 576 px) e detectores clássicos usam lado menor 800 px (SSD usa 300/320 px). Não foi realizado estudo de sensibilidade variando a resolução de entrada.
- **Medição de latência:** Cada modelo foi avaliado em rodada única de 5.000 imagens; não foram calculadas barras de erro para dispersão térmica entre execuções consecutivas.
- **Exportação e aceleração:** O projeto avalia apenas inferência nativa em PyTorch; exportações para ONNX ou engines TensorRT não foram executadas.
